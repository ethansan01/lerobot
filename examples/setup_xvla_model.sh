#!/bin/bash
# XVLA Model Setup Script for LeRobot Validation
# This script helps set up your trained XVLA model for validation

echo "🤖 XVLA Model Setup for LeRobot Validation"
echo "=========================================="

# Your checkpoint location
CHECKPOINT_DIR="/home/datafactory/Downloads/xvla/checkpoints/020000"
PRETRAINED_MODEL_DIR="$CHECKPOINT_DIR/pretrained_model"
LEROBOT_DIR="/home/datafactory/lerobot"

echo ""
echo "📍 Checking your checkpoint at: $CHECKPOINT_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "✅ Checkpoint directory exists"

# Check pretrained model directory
if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    echo "📁 Creating pretrained_model directory..."
    mkdir -p "$PRETRAINED_MODEL_DIR"
fi

echo ""
echo "📋 Checking model files in: $PRETRAINED_MODEL_DIR"

# Check for required files
REQUIRED_FILES=("model.safetensors" "config.json")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PRETRAINED_MODEL_DIR/$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "🚨 Missing model files detected!"
    echo ""
    echo "To complete your setup, you need to copy the following files"
    echo "from your training machine to: $PRETRAINED_MODEL_DIR/"
    echo ""
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "💡 On your training machine, these files should be located in:"
    echo "   {training_output_dir}/checkpoints/{step_number}/pretrained_model/"
    echo ""
    echo "📥 Transfer commands:"
    echo "   # On training machine:"
    echo "   tar -czf xvla_model.tar.gz /path/to/training/outputs/checkpoints/*/pretrained_model/"
    echo ""
    echo "   # Copy to this machine:"
    echo "   scp xvla_model.tar.gz user@$(hostname):/home/datafactory/"
    echo ""
    echo "   # Extract on this machine:"
    echo "   cd $CHECKPOINT_DIR"
    echo "   tar -xzf /home/datafactory/xvla_model.tar.gz --strip-components=X"
    echo ""
    echo "🔄 Alternatively, you can manually copy the files:"
    echo "   scp user@training-machine:/path/to/pretrained_model/model.safetensors $PRETRAINED_MODEL_DIR/"
    echo "   scp user@training-machine:/path/to/pretrained_model/config.json $PRETRAINED_MODEL_DIR/"
    echo "   scp user@training-machine:/path/to/pretrained_model/train_config.json $PRETRAINED_MODEL_DIR/"
    
    exit 1
fi

echo ""
echo "🎉 Model files are ready!"

# Update the symlink to point to this checkpoint
cd "/home/datafactory/Downloads/xvla/checkpoints"
if [ -L "last" ]; then
    rm last
fi
ln -sf 020000 last
echo "✅ Updated 'last' symlink to point to 020000"

echo ""
echo "🚀 Your XVLA model is ready for validation!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Start the policy server (Terminal 1):"
echo "   conda activate lerobot_server"
echo "   python -m lerobot.async_inference.policy_server \\"
echo "     --host=127.0.0.1 --port=8080"
echo ""
echo "2. Run validation client (Terminal 2):"
echo "   conda activate lerobot_client"
echo "   python $LEROBOT_DIR/examples/xvla_validation_client_proper.py \\"
echo "     --pretrained_name_or_path=\"$PRETRAINED_MODEL_DIR\" \\"
echo "     --server_address=\"127.0.0.1:8080\" \\"
echo "     --task=\"your task description\" \\"
echo "     --robot_port=\"/dev/ttyUSB0\""
echo ""
echo "🛠️ Hardware setup:"
echo "   - Connect your robot arm to /dev/ttyUSB0 (or update --robot_port)"
echo "   - Connect cameras and note their serial numbers"
echo "   - Add camera serials with --top_cam_serial and --wrist_cam_serial"
echo ""
echo "📖 For more details, see:"
echo "   $LEROBOT_DIR/examples/README_XVLA_VALIDATION.md"
echo "   $LEROBOT_DIR/examples/CHECKPOINT_TRANSFER_GUIDE.md"