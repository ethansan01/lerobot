# AIRBOT Robot Clients for Different Models

This directory contains robot clients for different LeRobot model types.

## Files

- **`airbot_robot_client.py`** - Original client for XVLA models
- **`airbot_robot_client_act.py`** - Client for ACT models

## Why Two Clients?

Different LeRobot models expect different camera feature names:

| Model Type | Camera Names | Client to Use |
|------------|--------------|---------------|
| **XVLA** | `image`, `image2` | `airbot_robot_client.py` |
| **ACT** | `top`, `wrist` | `airbot_robot_client_act.py` |

## Usage

### For XVLA Models

```bash
python examples/validation/airbot_robot_client.py \
    --pretrained /path/to/xvla/checkpoint \
    --policy-type xvla
```

### For ACT Models

```bash
python examples/validation/airbot_robot_client_act.py \
    --pretrained /path/to/act/checkpoint \
    --policy-type act
```

## Key Differences

### XVLA Client (`airbot_robot_client.py`)

**Camera mapping:**
```python
obs = {
    "image": frames["top"],      # top camera → "image"
    "image2": frames["wrist"],   # wrist camera → "image2"
}
```

**Features:**
```python
"observation.images.image": {...},
"observation.images.image2": {...},
```

### ACT Client (`airbot_robot_client_act.py`)

**Camera mapping:**
```python
obs = {
    "top": frames["top"],        # top camera → "top"
    "wrist": frames["wrist"],    # wrist camera → "wrist"
}
```

**Features:**
```python
"observation.images.top": {...},
"observation.images.wrist": {...},
```

## Common Arguments

Both clients support the same arguments:

```bash
--server SERVER              # Policy server address (default: 127.0.0.1:8080)
--pretrained PRETRAINED      # Path to model checkpoint (required)
--arm-port ARM_PORT          # AIRBOT arm port (default: 50001)
--top-cam TOP_CAM            # Top camera serial (default: CP7JC42000EY)
--wrist-cam WRIST_CAM        # Wrist camera serial (default: CP7JC42000F4)
--task TASK                  # Task description
--fps FPS                    # Control frequency (default: 30)
--actions-per-chunk          # Actions per chunk (default: 30)
--policy-device DEVICE       # Policy device (default: cuda)
--debug-queue                # Visualize action queue size
```

## Complete Workflow

### 1. Start the Policy Server

```bash
uv run -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
```

### 2. Run the Appropriate Client

**For XVLA:**
```bash
python examples/validation/airbot_robot_client.py \
    --pretrained checkpoints/xvla/my_model \
    --task "pick up the block"
```

**For ACT:**
```bash
python examples/validation/airbot_robot_client_act.py \
    --pretrained checkpoints/act/my_model \
    --task "pick up the block"
```

## Troubleshooting

### Error: "No module named 'examples.validation'"
Make sure you're running from the repository root:
```bash
cd /home/datafactory/lerobot
python examples/validation/airbot_robot_client_act.py ...
```

### Error: Camera feature mismatch
Make sure you're using the correct client for your model type:
- XVLA models → use `airbot_robot_client.py`
- ACT models → use `airbot_robot_client_act.py`

### Server Error: "ACTConfig object has no attribute 'get_florence_config'"
This means you're trying to use XVLA-specific features with an ACT model. Use `airbot_robot_client_act.py` instead.

## Adding Support for Other Models

To add support for a new model type (e.g., Pi0, Diffusion):

1. Copy one of the existing clients
2. Update the camera feature names in `_build_lerobot_features()`
3. Update the observation mapping in `get_observation()`
4. Update the default `policy_type` in the config

Example for Pi0 (which also uses "top" and "wrist"):
```bash
cp airbot_robot_client_act.py airbot_robot_client_pi0.py
# Edit to change default policy_type from "act" to "pi0"
# Camera mappings are the same as ACT
```
