# LeRobot Checkpoint Location and Transfer Guide

## Default Checkpoint Storage

LeRobot stores training checkpoints in the following structure by default:

```
{output_dir}/
└── checkpoints/
    ├── 000010/          # Step-specific checkpoint directories
    │   ├── pretrained_model/
    │   │   ├── model.safetensors
    │   │   ├── config.json
    │   │   └── train_config.json
    │   └── training_state/
    │       ├── optimizer_state.safetensors
    │       ├── scheduler_state.safetensors
    │       └── training_step.json
    ├── 000020/
    ├── ...
    └── last -> 000050   # Symlink to latest checkpoint
```

## Finding Your Checkpoints

1. **Default location**: By default, LeRobot saves checkpoints in `./outputs/` relative to where you ran the training script.

2. **Check your training command**: Look for the `--output-dir` parameter in your training command:
   ```bash
   python lerobot/scripts/lerobot_train.py \
     --config-name=act_xvla \
     --output-dir=/path/to/your/outputs
   ```

3. **Search for existing checkpoints**:
   ```bash
   find /home/datafactory -name "checkpoints" -type d 2>/dev/null
   find /home/datafactory -name "*.safetensors" -path "*/pretrained_model/*" 2>/dev/null
   ```

## Transferring Checkpoints

### Method 1: Full Checkpoint Transfer (Recommended)

Copy the entire checkpoint directory structure:

```bash
# On source machine, compress the checkpoint
tar -czf xvla_checkpoints.tar.gz /path/to/training/outputs/checkpoints/

# Transfer to target machine
scp xvla_checkpoints.tar.gz user@target-machine:/home/datafactory/

# On target machine, extract
cd /home/datafactory/
tar -xzf xvla_checkpoints.tar.gz
```

### Method 2: Model-Only Transfer

If you only need the trained model (not training state):

```bash
# Copy just the pretrained_model directory from your best checkpoint
cp -r /path/to/outputs/checkpoints/last/pretrained_model/ ./xvla_model/
```

## Using Transferred Checkpoints

### Option 1: Use as Local Path

```python
# In your validation client
python examples/xvla_validation_client_proper.py \
  --pretrained_name_or_path="/home/datafactory/xvla_model" \
  --server_address="127.0.0.1:8080"
```

### Option 2: Push to Hugging Face Hub (Recommended)

```python
# Upload your model to Hub for easy access
from lerobot.policies.factory import make_policy

policy = make_policy("xvla", pretrained_path="/path/to/pretrained_model")
policy.push_to_hub("your_username/xvla_finetuned_model")
```

Then use:
```python
python examples/xvla_validation_client_proper.py \
  --pretrained_name_or_path="your_username/xvla_finetuned_model" \
  --server_address="127.0.0.1:8080"
```

## Resuming Training

If you want to continue training from a checkpoint:

```bash
python lerobot/scripts/lerobot_train.py \
  --config-name=act_xvla \
  --output-dir=/path/to/outputs \
  --resume=true
```

This will automatically find the `last` checkpoint and resume from there.

## Checkpoint Files Explained

- **`model.safetensors`**: The actual model weights
- **`config.json`**: Model configuration (architecture, hyperparameters)  
- **`train_config.json`**: Training configuration (dataset, learning rate, etc.)
- **`optimizer_state.safetensors`**: Optimizer state (for resuming training)
- **`scheduler_state.safetensors`**: Learning rate scheduler state
- **`training_step.json`**: Current training step number

## Common Locations to Check

```bash
# Current directory
ls -la ./outputs/checkpoints/

# Home directory
ls -la ~/outputs/checkpoints/

# Common training directories  
ls -la /workspace/outputs/checkpoints/
ls -la /data/outputs/checkpoints/
ls -la /tmp/outputs/checkpoints/
```

## Quick Transfer Script

Here's a script to help locate and transfer your checkpoints:

```bash
#!/bin/bash
# find_and_transfer_checkpoints.sh

echo "Searching for LeRobot checkpoints..."

# Find all checkpoint directories
CHECKPOINT_DIRS=$(find /home /workspace /data /tmp -name "checkpoints" -type d 2>/dev/null | grep -v __pycache__ | head -10)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "No checkpoint directories found!"
    exit 1
fi

echo "Found checkpoint directories:"
for dir in $CHECKPOINT_DIRS; do
    echo "  $dir"
    if [ -d "$dir/last" ]; then
        echo "    -> Contains 'last' checkpoint"
        ls -la "$dir/last/pretrained_model/" 2>/dev/null | grep -E "(model\.safetensors|config\.json)"
    fi
    echo ""
done

echo "To transfer a checkpoint:"
echo "  tar -czf xvla_checkpoint.tar.gz /path/to/checkpoint/directory"
echo "  scp xvla_checkpoint.tar.gz target-machine:/destination/"
```

Save this as `find_checkpoints.sh`, make it executable with `chmod +x find_checkpoints.sh`, and run it to locate your checkpoints.