# M6 Real-Robot Deployment

## Goal

Deploy a validated policy to the dual-Piper robot through shadow mode first, then explicit low-speed execution.

## Scope

In scope:

- Policy server and client.
- Shadow mode.
- Low-speed real-robot execution.
- Deployment logs.

Out of scope:

- New model architecture research.
- Dataset loader changes unrelated to deployment.

## Planned Files

- `piper_cosmos/models/policy_server.py`
- `piper_cosmos/models/policy_client.py`
- `piper_cosmos/robot/piper_client.py`
- `tools/launch_shadow_mode.sh`
- `tools/launch_real_robot_slow.sh`

## Required Safety Defaults

- Shadow mode by default.
- Real execution requires explicit opt-in.
- `policy_hz = 2` for first low-speed tests.
- `execute_steps_per_prediction = 1`.
- `speed_scale = 0.1`.
- On any safety filter failure, hold position.

## Exit Criteria

- Shadow mode runs for 30 minutes without executing robot actions.
- Safety violation rate in shadow mode is below 0.5%.
- Real-robot tests proceed from hold position to small single-arm motions before any dual-arm task.
