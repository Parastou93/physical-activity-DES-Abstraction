# LPV-Based Physical Activity Simulation and DES Abstraction

This code simulates weekday physical activity using a linear parameter-varying (LPV) model and constructs a probabilistic discrete-event system (DES) describing progress toward a cumulative step goal.

It estimates the probability of achieving the goal across morning, afternoon, and evening under randomly scheduled “move more” messages.

## Overview

The script:

1. Loads model parameters and residual statistics from Excel.
2. Simulates activity at 15-minute intervals.
3. Schedules messages subject to daily and per-block limits.
4. Labels each completed time block according to goal achievement.
5. Builds pools of simulated histories.
6. Estimates transition probabilities between abstract states.
7. Exports simulation details and transition probabilities to Excel.

The code uses an existing model. It does not train the model or optimize message scheduling.

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas openpyxl
```

## Input Data

Place `TR665_model_var.xlsx` in the working directory, or update `FILE_PATH`.

The script reads the first worksheet and uses these columns:

| Column | Purpose |
|---|---|
| `coef_workday` | Weekday LPV model coefficients |
| `mean_resi_workday` | Mean prediction error at each modeled timestep |
| `std_resi_workday` | Standard deviation of prediction errors |
| `goal_param` | Cumulative step goal |
| `model_order` | Number of activity and message lags |
| `LPV_order` | Polynomial degree of the varying coefficients |
| `number_of_inputs` | Number of coefficient groups, including the activity group |
| `Activity info weekday` | Activity timestamps |

The supplied workbook contains a goal of **7,589 steps**, a model order of **5**, and polynomial degree **3**.

The coefficient vector contains **96 values**, arranged as:

```text
4 coefficient groups × 6 terms per group × 4 polynomial coefficients
```

The first group describes baseline activity and activity lags. The remaining groups describe message inputs. This reduced simulation uses only the first message-input group, assumed to represent “move more.”

## Activity Simulation

Each predicted step count combines:

- A time-varying baseline.
- The previous five activity values.
- The current message and previous five message values.
- Gaussian noise sampled from the residual statistics.

Negative predictions are clipped to zero. Simulated step counts remain continuous values.

By default, coefficients are evaluated using normalized time:

```python
tau = k / (T - 1)
```

This time convention must match the convention used when fitting the model.

## Time Blocks

Use the following indices for the supplied workbook:

```python
TIME_BLOCKS = {
    "M": list(range(12, 32)),   # 07:00–11:45
    "A": list(range(32, 46)),   # 12:00–15:15
    "E": list(range(46, 52)),   # 15:30–16:45
}
```

Python indices are zero-based. With one Excel header row:

```text
Excel row = Python index + 2
```

## Messaging Policy

Actions are:

| Action | Meaning |
|---|---|
| `0` | No message |
| `1` | Move-more message |

Default limits:

```python
MESSAGE_LIMIT = 4
MAX_MESSAGES_PER_BLOCK = 2
```

For each block, the code randomly selects an allowed message count and distinct delivery times. The remaining daily budget limits later blocks.

Messages are not selected in response to activity or goal status.

## DES States

Each abstract state is represented by:

```text
(time_block, step_status)
```

Time blocks are `M` (morning), `A` (afternoon), `E` (evening), and `O` (terminal).

Goal status is:

- `AG`: cumulative steps are at least the goal.
- `NAG`: cumulative steps are below the goal.

States describe activity **at the end of each block**. For example:

```text
(M, NAG) → (A, NAG) → (E, AG) → (O, AG)
```

The terminal state inherits the evening status. Terminal self-loops are included in the transition table.

Because step increments are nonnegative, an achieved goal remains achieved.

## Transition Probability Estimation

The abstract state omits details needed to continue the LPV simulation. The code therefore stores hidden states containing:

- Recent activity values.
- Recent message actions.
- Cumulative steps.
- Messages used by block completion.

Repeated simulations generate separate hidden-state pools for:

```text
(M, AG), (M, NAG), (A, AG), (A, NAG)
```

To estimate a transition, the code uniformly samples a hidden state from the relevant pool, simulates the next block, and records its final goal status.

Default settings are:

```python
num_pool_samples = 15000
num_transition_samples = 15000
```

The second setting specifies the number of continuations per nonempty source-state pool.

Evening-to-terminal transitions are deterministic.

## Usage

Save the script as `lpv_des_simulation.py` and run:

```bash
python lpv_des_simulation.py
```

The code can also be executed in a Jupyter notebook.

For reproducible results, replace the `seed=None` arguments in the simulation, pool construction, and transition estimation calls with fixed integer seeds.

## Outputs

The script creates:

```text
DES_simulation_results_with_terminal.xlsx
```

| Worksheet | Contents |
|---|---|
| `One_Simulation` | Timestep-level actions, predicted activity, and cumulative steps |
| `Block_AG_NAG_Summary` | Activity totals, message counts, and goal status by block |
| `DES_Trajectory` | Abstract trajectory from one simulation |
| `State_Space` | Eight candidate abstract states |
| `Hidden_Pool_Summary` | Number of samples in each hidden-state pool |
| `Transition_Probabilities` | Estimated transitions and terminal self-loops |
| `Summary` | Simulation settings and example-run results |

Rerunning the script overwrites the output workbook.


This simulation provides an empirical probabilistic abstraction for subsequent analysis of activity and goal achievement.
