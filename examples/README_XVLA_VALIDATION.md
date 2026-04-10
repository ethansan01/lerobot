# XVLA Validation Setup for LeRobot

This directory contains scripts for validating XVLA models using LeRobot's async inference system with separate server/client environments.

## Overview

The validation setup uses two separate conda environments:

- **Server Environment** (`lerobot_server`): Runs the policy server with GPU acceleration
- **Client Environment** (`lerobot_client`): Handles hardware APIs (cameras, robot arms)

This separation allows you to run the computationally intensive model inference on a GPU-equipped machine while keeping hardware dependencies isolated in the client environment.

## Quick Start

1. **Setup environments**:
   ```bash
   bash examples/setup_xvla_validation.sh
   ```

2. **Start the policy server** (Terminal 1):
   ```bash
   conda activate lerobot_server
   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
   ```

3. **Run validation client** (Terminal 2):
   ```bash
   conda activate lerobot_client  
   python examples/xvla_validation_client_proper.py \
     --pretrained_name_or_path="your_username/xvla_finetuned_model" \
     --server_address="127.0.0.1:8080" \
     --robot_port="/dev/ttyUSB0" \
     --task="pick up the red block and place it in the bowl"
   ```

## Files

- `xvla_validation_client_proper.py` - Main validation client using LeRobot async inference
- `xvla_policy_server.py` - Server wrapper (optional, can use built-in server)  
- `setup_xvla_validation.sh` - Environment setup script
- `README_XVLA_VALIDATION.md` - This documentation

## Configuration

The validation client accepts these key parameters:

- `--pretrained_name_or_path`: Your finetuned XVLA model path/name
- `--server_address`: Policy server address (default: "127.0.0.1:8080") 
- `--robot_port`: Robot serial port (default: "/dev/ttyUSB0")
- `--top_cam_serial`: Top camera serial number (optional)
- `--wrist_cam_serial`: Wrist camera serial number (optional)
- `--task`: Task description string
- `--control_freq`: Control frequency in Hz (default: 30)

## Hardware Support

The client includes support for:

- **Mock hardware** - For testing without physical devices
- **Orbbec cameras** - Via `pyorbbecsdk` 
- **AirBot arms** - Via `airbot_py`

If hardware packages are not available, mock versions will be used automatically.

## Action Extraction

The client extracts robot-specific actions from the model output:

```python
# Configure action extraction
arm_action_dim: int = 6      # Joint positions  
gripper_action_dim: int = 1  # Gripper position
action_offset: int = 0       # Starting index in action vector
```

This allows the same validation code to work with different robot configurations by adjusting these parameters.

## Troubleshooting

**Server connection issues:**
- Ensure the policy server is running and accessible
- Check firewall settings if using remote server
- Verify the model path is correct

**Hardware issues:** 
- Check device permissions (`sudo chmod 666 /dev/ttyUSB0`)
- Verify camera serial numbers with `lscam` or similar tools
- Install hardware-specific packages in client environment

**Model compatibility:**
- Ensure XVLA is in `SUPPORTED_POLICIES` list
- Check that your model outputs actions in expected format
- Verify action dimensions match your robot configuration

## Based On

This validation setup is inspired by the OpenPI validation_gripper.py approach but adapted to use LeRobot's async inference system properly. The key difference is that LeRobot handles model loading and inference on the server side, while the client focuses purely on hardware control and observation collection.