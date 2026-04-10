#!/bin/bash
"""
XVLA Validation Setup Script
============================

This script helps set up separate conda environments for LeRobot XVLA validation:
- Server environment (GPU, model inference) 
- Client environment (hardware APIs, robot control)

Usage:
    bash examples/setup_xvla_validation.sh
"""

set -e

echo "=============================================="
echo "LeRobot XVLA Validation Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}This script will create two conda environments:${NC}"
echo "  1. lerobot_server - For running the policy server (needs GPU)"
echo "  2. lerobot_client - For hardware control (cameras, robot arms)"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda/Miniconda first.${NC}"
    exit 1
fi

echo ""
echo "=============================================="
echo "Creating Server Environment (lerobot_server)"
echo "=============================================="

# Create server environment
conda create -n lerobot_server python=3.10 -y

echo "Activating lerobot_server environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot_server

echo "Installing LeRobot and dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd /home/datafactory/lerobot
pip install -e ".[all]"

echo -e "${GREEN}Server environment created successfully!${NC}"

echo ""
echo "=============================================="  
echo "Creating Client Environment (lerobot_client)"
echo "=============================================="

# Create client environment
conda create -n lerobot_client python=3.10 -y

echo "Activating lerobot_client environment..."
conda activate lerobot_client

echo "Installing LeRobot core (no GPU dependencies)..."
cd /home/datafactory/lerobot
pip install -e ".[core]"

echo "Installing hardware dependencies..."
# Note: These might need to be installed manually depending on hardware
echo -e "${YELLOW}Installing hardware APIs (may require manual setup):${NC}"

# Try to install common hardware APIs
pip install grpcio grpcio-tools protobuf tyro

echo -e "${YELLOW}Hardware-specific packages (install manually if needed):${NC}"
echo "  - For Orbbec cameras: pip install pyorbbecsdk" 
echo "  - For AirBot arms: pip install airbot_py"
echo "  - For visualization: pip install rerun-sdk"
echo "  - For image processing: pip install opencv-python pillow"

# Try installing common ones
pip install opencv-python pillow numpy || echo "Some packages failed to install"

echo -e "${GREEN}Client environment created successfully!${NC}"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="

echo -e "${GREEN}Environments created:${NC}"
echo "  - lerobot_server: For running policy server"
echo "  - lerobot_client: For hardware control"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Test the server environment:"
echo "   conda activate lerobot_server"
echo "   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080"
echo ""

echo "2. In another terminal, test the client:"
echo "   conda activate lerobot_client"
echo "   python examples/xvla_validation_client_proper.py --help"
echo ""

echo "3. For full validation, specify your model:"
echo "   python examples/xvla_validation_client_proper.py \\"
echo "     --pretrained_name_or_path=\"your_username/xvla_model\" \\"
echo "     --server_address=\"127.0.0.1:8080\" \\"
echo "     --task=\"pick up the red block\""
echo ""

echo -e "${GREEN}Setup complete! Check the documentation in the Python files for more details.${NC}"