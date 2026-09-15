import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from openpyxl.styles import PatternFill, Border, Side, Font, Alignment
from openpyxl.utils import get_column_letter


# ============================================================
# 1. Load model data from Excel
# ============================================================

FILE_PATH = "TR665_model_var.xlsx"

raw = pd.read_excel(FILE_PATH, sheet_name=0)

coef = raw["coef_workday"].dropna().to_numpy(dtype=float)
mean_resi = raw["mean_resi_workday"].dropna().to_numpy(dtype=float)
std_resi = raw["std_resi_workday"].dropna().to_numpy(dtype=float)

goal = int(raw["goal_param"].dropna().iloc[0])
model_order = int(raw["model_order"].dropna().iloc[0])      # n = 5
lpv_order = int(raw["LPV_order"].dropna().iloc[0])          # P = 3
num_blocks = int(raw["number_of_inputs"].dropna().iloc[0])  # 4 = 1 a-block + 3 message blocks

T = len(mean_resi)

print("Goal:", goal)
print("Model order:", model_order)
print("LPV order:", lpv_order)
print("Number of coefficient blocks:", num_blocks)
print("T:", T)


# ============================================================
# 2. Message constraints
# ============================================================

MESSAGE_LIMIT = 4
MAX_MESSAGES_PER_BLOCK = 2


# ============================================================
# 3. Parse LPV coefficients
# ============================================================

def parse_coefficients(coef, n, lpv_order, num_blocks):
    """
    Coefficient vector structure:

        96 = 4 * 6 * 4

    where:
        4 = coefficient groups:
            group 0: a coefficients
            groups 1,2,3: b coefficients for message inputs

        6 = n + 1:
            index 0: baseline/current term
            indices 1,...,5: lag terms

        4 = LPV polynomial coefficients:
            p = 0,1,2,3
    """

    block_size = lpv_order + 1
    expected = num_blocks * (n + 1) * block_size

    if len(coef) != expected:
        raise ValueError(
            f"Expected {expected} coefficients, but found {len(coef)}."
        )

    blocks = coef.reshape(num_blocks, n + 1, block_size)

    a_blocks = blocks[0]
    b_blocks = blocks[1:]

    return a_blocks, b_blocks


a_blocks, b_blocks = parse_coefficients(
    coef=coef,
    n=model_order,
    lpv_order=lpv_order,
    num_blocks=num_blocks
)


# ============================================================
# 4. Evaluate time-varying LPV coefficients
# ============================================================

USE_NORMALIZED_TIME = True


def poly_value(poly_coeffs: np.ndarray, k: int, T: int) -> float:
    """
    Evaluate LPV coefficient.

    If USE_NORMALIZED_TIME = True:

        tau_k = k / (T - 1)

    and:

        c_k = c0 + c1*tau_k + c2*tau_k^2 + c3*tau_k^3
    """

    if USE_NORMALIZED_TIME:
        time_var = k / (T - 1)
    else:
        time_var = k

    value = 0.0
    for power, coefficient in enumerate(poly_coeffs):
        value += coefficient * (time_var ** power)

    return value


# ============================================================
# 5. Reduced weekday model
# ============================================================

MOVE_MORE_INDEX = 0


def simulate_one_step(
    k: int,
    y_history: List[float],
    u_history: List[int],
    current_action: int,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """
    Simulate one 15-minute step count.

    Reduced model:

        raw_y_k =
            a_k^(0)
            + sum_{i=1}^{5} a_k^(i) y_{k-i}
            + b_k^(0) u_k
            + sum_{i=1}^{5} b_k^(i) u_{k-i}
            + w_k

    where:
        u_k = 0: no message
        u_k = 1: move more

    Then:
        y_k = max(raw_y_k, 0)
    """

    n = model_order

    y_k = poly_value(a_blocks[0], k, T)

    for i in range(1, n + 1):
        a_i_k = poly_value(a_blocks[i], k, T)
        y_k += a_i_k * y_history[-i]

    b_move = b_blocks[MOVE_MORE_INDEX]

    b_0_k = poly_value(b_move[0], k, T)
    y_k += b_0_k * current_action

    for i in range(1, n + 1):
        b_i_k = poly_value(b_move[i], k, T)
        y_k += b_i_k * u_history[-i]

    w_k = rng.normal(mean_resi[k], std_resi[k])
    y_k += w_k

    raw_y_k = y_k
    y_k = max(raw_y_k, 0.0)

    return y_k, raw_y_k


# ============================================================
# 6. DES time blocks
# ============================================================

"""
Excel row = Python index + 2

M = [7:00, 12:00)
A = [12:00, 15:30)
E = [15:30, 16:45]
O = outside / terminal state after evening
"""

TIME_BLOCKS = {
    "M": list(range(14, 32)),   # Excel rows 16-33: 7:00 to 11:45
    "A": list(range(32, 46)),   # Excel rows 34-47: 12:00 to 15:15
    "E": list(range(46, 52)),   # Excel rows 48-53: 15:30 to 16:45
}

TIME_INTERVALS = {
    "M": "[7:00, 12:00)",
    "A": "[12:00, 15:30)",
    "E": "[15:30, 16:45]",
    "O": "outside / terminal",
}

NEXT_TIME = {
    "M": "A",
    "A": "E",
    "E": "O",
    "O": None,
}

activity_times = raw["Activity info weekday"].dropna().to_list()


def check_time_blocks(time_blocks):
    print("\nChecking TIME_BLOCKS against Excel rows/time labels:\n")

    for block, indices in time_blocks.items():
        print(f"{block}:")
        print(f"  Python indices: {indices[0]} to {indices[-1]}")
        print(f"  Excel rows:     {indices[0] + 2} to {indices[-1] + 2}")
        print(f"  Time labels:    {activity_times[indices[0]]} to {activity_times[indices[-1]]}")
        print(f"  Number samples: {len(indices)}")
        print()


check_time_blocks(TIME_BLOCKS)


# ============================================================
# 7. DES state and hidden state definitions
# ============================================================

@dataclass
class DESState:
    """
    Abstract DES state:

        (time_block, step_status)

    Example:
        (M, NAG)
        (A, AG)
        (O, AG)
    """
    time_block: str
    step_status: str


@dataclass
class HiddenState:
    """
    Hidden state needed to continue the LPV model.

    Contains:
        last 5 y values
        last 5 u values
        cumulative steps
        number of messages already sent
    """
    time_block: str
    step_status: str
    y_history: List[float]
    u_history: List[int]
    cumulative_steps: float
    message_count: int


def status_from_cumulative_steps(cumulative_steps: float, goal: float) -> str:
    if cumulative_steps >= goal:
        return "AG"
    return "NAG"


# ============================================================
# 8. Sample constrained block actions
# ============================================================

def sample_block_actions(
    k_values: List[int],
    rng: np.random.Generator,
    message_count: int
) -> Tuple[Dict[int, int], int, int]:
    """
    Randomly choose 0, 1, or 2 messages for this block, while respecting:

        total daily messages <= 4
        messages per block <= 2
    """

    remaining_messages = MESSAGE_LIMIT - message_count
    max_allowed_this_block = min(MAX_MESSAGES_PER_BLOCK, remaining_messages)

    if max_allowed_this_block <= 0:
        actions = {k: 0 for k in k_values}
        return actions, message_count, 0

    num_messages_this_block = int(rng.integers(0, max_allowed_this_block + 1))

    if num_messages_this_block == 0:
        message_times = set()
    else:
        message_times = set(
            rng.choice(k_values, size=num_messages_this_block, replace=False)
        )

    actions = {
        k: 1 if k in message_times else 0
        for k in k_values
    }

    message_count += num_messages_this_block

    return actions, message_count, num_messages_this_block


# ============================================================
# 9. Simulate one block from a hidden state
# ============================================================

def simulate_block(
    block_name: str,
    y_history: List[float],
    u_history: List[int],
    cumulative_steps: float,
    message_count: int,
    rng: np.random.Generator,
    record_details: bool = False
):
    """
    Simulate one block only.

    Example:
        block_name = "A" simulates k = 32,...,45.

    This uses the incoming hidden state:
        y_history
        u_history
        cumulative_steps
        message_count
    """

    n = model_order
    detailed_rows = []
    k_values = TIME_BLOCKS[block_name]

    actions, message_count, num_messages_this_block = sample_block_actions(
        k_values=k_values,
        rng=rng,
        message_count=message_count
    )

    for k in k_values:

        u_k = actions[k]

        y_k, raw_y_k = simulate_one_step(
            k=k,
            y_history=y_history,
            u_history=u_history,
            current_action=u_k,
            rng=rng
        )

        cumulative_steps += y_k

        if record_details:
            detailed_rows.append({
                "k": k,
                "excel_row": k + 2,
                "time_label": activity_times[k],
                "block": block_name,
                "action": u_k,
                "action_name": "move_more" if u_k == 1 else "no_message",
                "raw_step_count": raw_y_k,
                "step_count": y_k,
                "cumulative_steps": cumulative_steps,
                "message_count_so_far": message_count,
                "block_message_count": num_messages_this_block,
                "daily_message_limit": MESSAGE_LIMIT,
                "max_messages_per_block": MAX_MESSAGES_PER_BLOCK
            })

        y_history.append(y_k)
        y_history = y_history[-n:]

        u_history.append(u_k)
        u_history = u_history[-n:]

    return (
        y_history,
        u_history,
        cumulative_steps,
        message_count,
        num_messages_this_block,
        detailed_rows
    )


# ============================================================
# 10. Simulate one full DES trajectory from zero
# ============================================================

def simulate_des_day(
    seed: int = 1,
    initial_y_history: List[float] = None
):
    """
    Simulate one full trajectory:

        M -> A -> E -> O

    AG/NAG is checked only at the end of M, A, E.
    O is terminal and inherits the final E status.

    Example:
        (M,NAG) -> (A,NAG) -> (E,AG) -> (O,AG)
    """

    rng = np.random.default_rng(seed)
    n = model_order

    if initial_y_history is None:
        y_history = [0.0] * n
    else:
        if len(initial_y_history) != n:
            raise ValueError(f"initial_y_history must have length {n}.")
        y_history = list(initial_y_history)

    u_history = [0] * n
    cumulative_steps = 0.0
    message_count = 0

    des_trajectory = []
    detailed_steps = []
    block_summary = []

    for block_name in ["M", "A", "E"]:

        block_start_cumulative = cumulative_steps
        block_start_message_count = message_count

        (
            y_history,
            u_history,
            cumulative_steps,
            message_count,
            num_messages_this_block,
            rows
        ) = simulate_block(
            block_name=block_name,
            y_history=y_history,
            u_history=u_history,
            cumulative_steps=cumulative_steps,
            message_count=message_count,
            rng=rng,
            record_details=True
        )

        detailed_steps.extend(rows)

        step_status = status_from_cumulative_steps(cumulative_steps, goal)

        des_trajectory.append(
            DESState(
                time_block=block_name,
                step_status=step_status
            )
        )

        block_summary.append({
            "block": block_name,
            "time_interval": TIME_INTERVALS[block_name],
            "block_steps": cumulative_steps - block_start_cumulative,
            "cumulative_steps_at_block_end": cumulative_steps,
            "messages_in_block": num_messages_this_block,
            "message_count_at_block_start": block_start_message_count,
            "message_count_at_block_end": message_count,
            "AG_NAG": step_status
        })

    # Terminal outside state inherits the evening AG/NAG status.
    final_status = des_trajectory[-1].step_status

    des_trajectory.append(
        DESState(
            time_block="O",
            step_status=final_status
        )
    )

    block_summary = pd.DataFrame(block_summary)

    terminal_row = pd.DataFrame([{
        "block": "O",
        "time_interval": TIME_INTERVALS["O"],
        "block_steps": 0.0,
        "cumulative_steps_at_block_end": cumulative_steps,
        "messages_in_block": 0,
        "message_count_at_block_start": message_count,
        "message_count_at_block_end": message_count,
        "AG_NAG": final_status
    }])

    block_summary = pd.concat([block_summary, terminal_row], ignore_index=True)

    return (
        des_trajectory,
        pd.DataFrame(detailed_steps),
        block_summary
    )


# ============================================================
# 11. Build hidden-state pools
# ============================================================

def build_hidden_state_pools(
    num_pool_samples: int = 15000,
    seed: int = None
) -> Dict[Tuple[str, str], List[HiddenState]]:
    """
    Build hidden-state pools at the end of Morning and Afternoon.

    Pools:
        (M, AG), (M, NAG)
        (A, AG), (A, NAG)

    These are used to estimate:
        M -> A
        A -> E

    E -> O is deterministic and does not need a hidden pool.
    """

    master_rng = np.random.default_rng(seed)

    pools = {
        ("M", "AG"): [],
        ("M", "NAG"): [],
        ("A", "AG"): [],
        ("A", "NAG"): [],
    }

    n = model_order

    for _ in range(num_pool_samples):

        rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))

        y_history = [0.0] * n
        u_history = [0] * n
        cumulative_steps = 0.0
        message_count = 0

        # Morning
        (
            y_history,
            u_history,
            cumulative_steps,
            message_count,
            _,
            _
        ) = simulate_block(
            block_name="M",
            y_history=y_history,
            u_history=u_history,
            cumulative_steps=cumulative_steps,
            message_count=message_count,
            rng=rng,
            record_details=False
        )

        status_M = status_from_cumulative_steps(cumulative_steps, goal)

        pools[("M", status_M)].append(
            HiddenState(
                time_block="M",
                step_status=status_M,
                y_history=list(y_history),
                u_history=list(u_history),
                cumulative_steps=float(cumulative_steps),
                message_count=int(message_count)
            )
        )

        # Afternoon
        (
            y_history,
            u_history,
            cumulative_steps,
            message_count,
            _,
            _
        ) = simulate_block(
            block_name="A",
            y_history=y_history,
            u_history=u_history,
            cumulative_steps=cumulative_steps,
            message_count=message_count,
            rng=rng,
            record_details=False
        )

        status_A = status_from_cumulative_steps(cumulative_steps, goal)

        pools[("A", status_A)].append(
            HiddenState(
                time_block="A",
                step_status=status_A,
                y_history=list(y_history),
                u_history=list(u_history),
                cumulative_steps=float(cumulative_steps),
                message_count=int(message_count)
            )
        )

    return pools


# ============================================================
# 12. Estimate transition probabilities using hidden-state pools
# ============================================================

def estimate_transition_probabilities_from_pools(
    pools: Dict[Tuple[str, str], List[HiddenState]],
    num_transition_samples: int = 15000,
    seed: int = None
) -> Dict[Tuple[str, str, str], Dict[Tuple[str, str], float]]:
    """
    Estimate transitions:

        M -> A
        A -> E

    using uniform sampling from hidden-state pools.

    Then add deterministic terminal transitions:

        (E,AG)  -> (O,AG)  with probability 1
        (E,NAG) -> (O,NAG) with probability 1
    """

    rng = np.random.default_rng(seed)

    transition_counts = {}

    transition_specs = [
        ("M", "A"),
        ("A", "E")
    ]

    for current_block, next_block in transition_specs:

        for current_status in ["AG", "NAG"]:

            pool_key = (current_block, current_status)
            pool = pools.get(pool_key, [])

            if len(pool) == 0:
                continue

            key = (
                current_block,
                current_status,
                "uniform_hidden_pool_with_message_limit"
            )

            transition_counts[key] = {}

            for _ in range(num_transition_samples):

                sampled_hidden = pool[rng.integers(0, len(pool))]

                y_history = list(sampled_hidden.y_history)
                u_history = list(sampled_hidden.u_history)
                cumulative_steps = float(sampled_hidden.cumulative_steps)
                message_count = int(sampled_hidden.message_count)

                (
                    y_history,
                    u_history,
                    cumulative_steps,
                    message_count,
                    _,
                    _
                ) = simulate_block(
                    block_name=next_block,
                    y_history=y_history,
                    u_history=u_history,
                    cumulative_steps=cumulative_steps,
                    message_count=message_count,
                    rng=rng,
                    record_details=False
                )

                next_status = status_from_cumulative_steps(cumulative_steps, goal)
                next_state = (next_block, next_status)

                transition_counts[key][next_state] = transition_counts[key].get(next_state, 0) + 1

    transition_probs = {}

    for key, counts in transition_counts.items():
        total = sum(counts.values())

        transition_probs[key] = {
            next_state: count / total
            for next_state, count in counts.items()
        }

    # Add deterministic terminal transitions E -> O.
    transition_probs[("E", "AG", "terminal_transition")] = {
        ("O", "AG"): 1.0
    }

    transition_probs[("E", "NAG", "terminal_transition")] = {
        ("O", "NAG"): 1.0
    }

    # Make O absorbing if desired.
    transition_probs[("O", "AG", "absorbing_terminal")] = {
        ("O", "AG"): 1.0
    }

    transition_probs[("O", "NAG", "absorbing_terminal")] = {
        ("O", "NAG"): 1.0
    }

    return transition_probs


# ============================================================
# 13. Run one sample DES trajectory
# ============================================================

num_pool_samples = 15000
num_transition_samples = 15000

des_traj, details, block_summary = simulate_des_day(seed=None)

print("\nDES trajectory from one sample run:")
for state in des_traj:
    print(f"({state.time_block}, {state.step_status})")

print("\nBlock-level AG/NAG summary from one sample run:")
print(block_summary)

print("\nFinal cumulative steps from one sample run:")
print(details["cumulative_steps"].iloc[-1])

print("\nTotal messages in one sample run:")
print(block_summary.loc[block_summary["block"].isin(["M", "A", "E"]), "messages_in_block"].sum())


# ============================================================
# 14. Build hidden-state pools and estimate transitions
# ============================================================

pools = build_hidden_state_pools(
    num_pool_samples=num_pool_samples,
    seed=None
)

print("\nHidden-state pool sizes:")
for key, value in pools.items():
    print(f"{key}: {len(value)}")

P_hat = estimate_transition_probabilities_from_pools(
    pools=pools,
    num_transition_samples=num_transition_samples,
    seed=None
)

print("\nEstimated transition probabilities using hidden-state pools:")
for current, next_dict in P_hat.items():
    print(f"\nFrom {current}:")
    for next_state, prob in next_dict.items():
        print(f"  to {next_state}: {prob:.6f}")


# ============================================================
# 15. Prepare Excel outputs
# ============================================================

OUTPUT_FILE = "DES_simulation_results_with_terminal.xlsx"

des_traj_df = pd.DataFrame([
    {
        "DES_state_number": i + 1,
        "DES_state": f"({state.time_block}, {state.step_status})",
        "block": state.time_block,
        "AG_NAG": state.step_status
    }
    for i, state in enumerate(des_traj)
])

state_space_df = pd.DataFrame([
    {"state": "(M, AG)", "block": "M", "AG_NAG": "AG"},
    {"state": "(M, NAG)", "block": "M", "AG_NAG": "NAG"},
    {"state": "(A, AG)", "block": "A", "AG_NAG": "AG"},
    {"state": "(A, NAG)", "block": "A", "AG_NAG": "NAG"},
    {"state": "(E, AG)", "block": "E", "AG_NAG": "AG"},
    {"state": "(E, NAG)", "block": "E", "AG_NAG": "NAG"},
    {"state": "(O, AG)", "block": "O", "AG_NAG": "AG"},
    {"state": "(O, NAG)", "block": "O", "AG_NAG": "NAG"},
])

pool_summary_df = pd.DataFrame([
    {
        "abstract_state": f"({block}, {status})",
        "block": block,
        "AG_NAG": status,
        "pool_size": len(hidden_list)
    }
    for (block, status), hidden_list in pools.items()
])

transition_rows = []

for current, next_dict in P_hat.items():
    current_block, current_status, input_process = current

    for next_state, prob in next_dict.items():
        next_block, next_status = next_state

        transition_rows.append({
            "current_state": f"({current_block}, {current_status})",
            "next_state": f"({next_block}, {next_status})",
            "input_process": input_process,
            "probability": prob
        })

transition_df = pd.DataFrame(transition_rows)

summary_df = pd.DataFrame([
    {
        "goal": goal,
        "message_limit_per_day": MESSAGE_LIMIT,
        "max_messages_per_block": MAX_MESSAGES_PER_BLOCK,
        "num_pool_samples": num_pool_samples,
        "num_transition_samples": num_transition_samples,
        "one_run_final_cumulative_steps": details["cumulative_steps"].iloc[-1],
        "one_run_final_AG_NAG": des_traj[-1].step_status,
        "one_run_total_messages": block_summary.loc[
            block_summary["block"].isin(["M", "A", "E"]),
            "messages_in_block"
        ].sum(),
        "first_time_label": details["time_label"].iloc[0],
        "last_time_label": details["time_label"].iloc[-1],
        "first_excel_row": details["excel_row"].min(),
        "last_excel_row": details["excel_row"].max(),
    }
])


# ============================================================
# 16. Excel formatting helpers
# ============================================================

def format_sheet(ws):
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    header_font = Font(bold=True)
    thin_gray = Side(style="thin", color="C9C9C9")
    border = Border(left=thin_gray, right=thin_gray, top=thin_gray, bottom=thin_gray)

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = border

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(horizontal="center")

    for col in ws.columns:
        max_length = 0
        col_letter = get_column_letter(col[0].column)

        for cell in col:
            if cell.value is not None:
                max_length = max(max_length, len(str(cell.value)))

        ws.column_dimensions[col_letter].width = min(max_length + 3, 40)

    ws.freeze_panes = "A2"


def format_one_simulation_blocks(ws):
    fills = {
        "M": PatternFill("solid", fgColor="EAF4EA"),
        "A": PatternFill("solid", fgColor="FFF2CC"),
        "E": PatternFill("solid", fgColor="FCE4D6"),
    }

    thick_top = Side(style="medium", color="666666")
    thick_bottom = Side(style="medium", color="666666")
    thin_gray = Side(style="thin", color="C9C9C9")

    block_col = None
    for cell in ws[1]:
        if cell.value == "block":
            block_col = cell.column
            break

    if block_col is None:
        return

    max_col = ws.max_column
    max_row = ws.max_row

    for row_idx in range(2, max_row + 1):
        block_value = ws.cell(row=row_idx, column=block_col).value
        fill = fills.get(block_value)

        if fill is not None:
            for col_idx in range(1, max_col + 1):
                ws.cell(row=row_idx, column=col_idx).fill = fill

    for row_idx in range(2, max_row + 1):
        current_block = ws.cell(row=row_idx, column=block_col).value
        previous_block = ws.cell(row=row_idx - 1, column=block_col).value if row_idx > 2 else None
        next_block = ws.cell(row=row_idx + 1, column=block_col).value if row_idx < max_row else None

        if current_block != previous_block:
            for col_idx in range(1, max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = Border(
                    left=thin_gray,
                    right=thin_gray,
                    top=thick_top,
                    bottom=cell.border.bottom
                )

        if current_block != next_block:
            for col_idx in range(1, max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = Border(
                    left=thin_gray,
                    right=thin_gray,
                    top=cell.border.top,
                    bottom=thick_bottom
                )


def format_block_summary(ws):
    fills = {
        "M": PatternFill("solid", fgColor="EAF4EA"),
        "A": PatternFill("solid", fgColor="FFF2CC"),
        "E": PatternFill("solid", fgColor="FCE4D6"),
        "O": PatternFill("solid", fgColor="E4DFEC"),
    }

    thick = Side(style="medium", color="666666")
    thin = Side(style="thin", color="C9C9C9")

    block_col = 1

    for row_idx in range(2, ws.max_row + 1):
        block_value = ws.cell(row=row_idx, column=block_col).value
        fill = fills.get(block_value)

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)

            if fill is not None:
                cell.fill = fill

            cell.border = Border(left=thin, right=thin, top=thick, bottom=thick)
            cell.alignment = Alignment(horizontal="center")

        ag_cell = ws.cell(row=row_idx, column=ws.max_column)
        if ag_cell.value == "AG":
            ag_cell.fill = PatternFill("solid", fgColor="C6EFCE")
        elif ag_cell.value == "NAG":
            ag_cell.fill = PatternFill("solid", fgColor="FFC7CE")


# ============================================================
# 17. Save to Excel
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    details.to_excel(writer, sheet_name="One_Simulation", index=False)
    block_summary.to_excel(writer, sheet_name="Block_AG_NAG_Summary", index=False)
    des_traj_df.to_excel(writer, sheet_name="DES_Trajectory", index=False)
    state_space_df.to_excel(writer, sheet_name="State_Space", index=False)
    pool_summary_df.to_excel(writer, sheet_name="Hidden_Pool_Summary", index=False)
    transition_df.to_excel(writer, sheet_name="Transition_Probabilities", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    workbook = writer.book

    for sheet_name in workbook.sheetnames:
        format_sheet(workbook[sheet_name])

    format_one_simulation_blocks(workbook["One_Simulation"])
    format_block_summary(workbook["Block_AG_NAG_Summary"])

print(f"\nResults saved to: {OUTPUT_FILE}")
