# LeRobot Validation Guide

This guide explains how to validate trained policies using LeRobot's asynchronous inference system with real hardware.

## Overview

LeRobot provides a client-server architecture for real-time policy inference:
- **Policy Server**: Runs the trained policy model and performs inference
- **Robot Client**: Connects to hardware (robot + cameras), sends observations, and executes actions

This architecture separates compute-heavy inference from real-time robot control, enabling smooth operation even with complex models.

## Setup

### Prerequisites

Ensure you have LeRobot installed with all dependencies:

```bash
uv sync
```

## Running Validation

### 1. Start the Policy Server

The policy server hosts your trained model and waits for observation requests from the robot client.

**Basic usage:**

```bash
uv run -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080
```

**Available parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Server host address | `localhost` |
| `--port` | Server port number | `8080` |
| `--fps` | Target frames per second | `30` |
| `--inference_latency` | Target inference latency in seconds | `0.033` |
| `--obs_queue_timeout` | Observation queue timeout in seconds | `1.0` |

**Example with custom settings:**

```bash
uv run -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
```

### 2. Run the Robot Client

The robot client connects to your hardware and communicates with the policy server.

**Basic usage:**

```bash
uv run examples/validation/airbot_robot_client.py \
    --pretrained /path/to/your/checkpoint \
    --server 127.0.0.1:8080
```

**Available parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pretrained` | Path to trained model checkpoint | *Required* |
| `--server` | Policy server address (host:port) | `127.0.0.1:8080` |
| `--policy-type` | Policy type (xvla, pi0, act, etc.) | `xvla` |
| `--arm-port` | AIRBOT arm port number | `50001` |
| `--top-cam` | Top camera serial number | `CP7JC42000EY` |
| `--wrist-cam` | Wrist camera serial number | `CP7JC42000F4` |
| `--task` | Task instruction for the policy | `pick up the red block and place it in the bowl` |
| `--fps` | Control frequency (Hz) | `30` |
| `--actions-per-chunk` | Number of actions per inference chunk | `30` |
| `--policy-device` | Device for policy inference | `cuda` |
| `--debug-queue` | Visualize action queue size | `False` |

**Example with custom configuration:**

```bash
uv run examples/validation/airbot_robot_client.py \
    --pretrained ./outputs/train/xvla_airbot/checkpoints/last/pretrained_model \
    --server 127.0.0.1:8080 \
    --policy-type xvla \
    --task "pick up the blue cube" \
    --arm-port 50001 \
    --top-cam CP7JC42000EY \
    --wrist-cam CP7JC42000F4 \
    --fps 30 \
    --policy-device cuda
```

## Architecture Details

### Communication Flow

```
Robot Hardware → Robot Client → gRPC → Policy Server
                      ↑                        ↓
                      └─────── Actions ────────┘
```

1. **Observation Collection**: Client captures camera frames and robot state
2. **Transmission**: Observations sent to server via gRPC
3. **Inference**: Server runs policy inference and generates action chunks
4. **Execution**: Client receives actions and executes them on robot
5. **Repeat**: Process continues at target FPS

### Action Chunking

The policy generates multiple future actions per observation (action chunk), enabling:
- Smoother control by overlapping predictions
- Reduced network overhead
- Better handling of inference latency

The `--actions-per-chunk` parameter controls chunk size. Typical values: 10-50 actions.

### Visualization

The client uses [Rerun](https://rerun.io/) for real-time visualization:
- Live camera feeds (top and wrist views)
- Robot state (joint positions, gripper)
- Action commands
- Queue statistics

Rerun viewer launches automatically when the client starts.

## Troubleshooting

### Connection Issues

**Problem**: Client can't connect to server
- Ensure server is running before starting client
- Check firewall settings allow connections on the specified port
- Verify host:port matches between server and client

### Camera Issues

**Problem**: Camera frames not available
- Check camera serial numbers match your hardware
- Verify USB connections
- Ensure no other process is using the cameras
- Try restarting the cameras by unplugging/replugging USB

### Robot Control Issues

**Problem**: Robot not responding to commands
- Verify arm port number is correct
- Check robot is powered on and in correct mode
- Ensure robot communication cable is connected
- Try manually moving robot to home position first

### Performance Issues

**Problem**: Inference too slow
- Use `--policy-device cuda` for GPU acceleration
- Reduce `--fps` if system can't keep up
- Check GPU memory usage (may need smaller batch size)
- Reduce `--actions-per-chunk` to decrease inference load

**Problem**: Actions laggy or jerky
- Increase `--actions-per-chunk` for smoother overlap
- Adjust `chunk_size_threshold` in code if needed
- Check network latency between client and server



### Multiple Policies

Switch between policies by stopping the client and restarting with different `--pretrained` path. The server automatically reconfigures.

### Custom Hardware

To adapt for different robots/cameras, modify `airbot_robot_client.py`:
1. Implement custom camera streamer (replace `DualOrbbecStreamer`)
2. Implement custom robot controller (replace `AirBotArm`)
3. Update `_build_lerobot_features()` to match your observation space
4. Adjust state mapping if needed

## File Reference

- **Policy Server**: `src/lerobot/async_inference/policy_server.py`
- **Robot Client**: `examples/validation/airbot_robot_client.py`
- **Configuration**: `src/lerobot/async_inference/configs.py`
- **Transport Layer**: `src/lerobot/transport/`

## Additional Resources

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [Async Inference Tutorial](examples/tutorial/async-inf/)
- [Policy Training Guide](docs/training.md)

## Safety Notes

⚠️ **Important Safety Guidelines:**
- Always supervise robot during validation
- Keep emergency stop accessible
- Start with slow movements (lower FPS)
- Test in safe environment away from obstacles
- Verify robot workspace is clear before starting
