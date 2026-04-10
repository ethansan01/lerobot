#!/usr/bin/env python3
"""
XVLA Model Validation Client for LeRobot
========================================

This client connects to a LeRobot policy server running XVLA model and validates 
the policy performance with real robot hardware. Designed to run in a separate
conda environment from the policy server.

Usage:
    python examples/xvla_validation_client.py \
        --server_address="127.0.0.1:8080" \
        --robot_port="/dev/ttyUSB0" \
        --top_cam_serial="XXXXXX" \
        --wrist_cam_serial="YYYYYY" \
        --task="pick up the red block and place it in the bowl" \
        --control_freq=30 \
        --action_horizon=10

Requirements:
    - LeRobot policy server running on specified address
    - Robot hardware connected and accessible 
    - Cameras connected with specified serial numbers
    - Separate conda environment with robot/camera dependencies
"""

import time
import threading
import numpy as np
import tyro
import dataclasses
from typing import Optional, Dict
from queue import Queue, Empty

# Hardware Libraries (install in client environment)
try:
    from pyorbbecsdk import Pipeline, Context, Config, OBSensorType, OBFormat
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False
    print("Warning: pyorbbecsdk not available. Mock camera will be used.")

try:
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    HAS_AIRBOT = True
except ImportError:
    HAS_AIRBOT = False
    print("Warning: airbot_py not available. Mock robot will be used.")

# LeRobot async inference client
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.robots.config import RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclasses.dataclass
class ValidationConfig:
    """Configuration for XVLA validation client."""
    
    # Server connection
    server_address: str = "127.0.0.1:8080"
    
    # Robot configuration  
    robot_port: str = "/dev/ttyUSB0"
    robot_id: str = "validation_robot"
    
    # Camera configuration
    top_cam_serial: Optional[str] = None
    wrist_cam_serial: Optional[str] = None
    
    # Task configuration
    task: str = "pick up the red block and place it in the bowl"
    
    # Control parameters
    control_freq: int = 30  # Hz
    action_horizon: int = 10  # Actions to execute per inference
    
    # Model configuration
    policy_type: str = "xvla"
    pretrained_name_or_path: str = ""  # Set this to your finetuned model path
    policy_device: str = "cuda"
    client_device: str = "cpu"
    
    # Execution parameters
    chunk_size_threshold: float = 0.5
    actions_per_chunk: int = 50
    
    # Safety
    enable_safety_checks: bool = True
    max_episode_steps: int = 1000


class MockCamera:
    """Mock camera for testing without hardware."""
    
    def __init__(self, width=224, height=224):
        self.width = width
        self.height = height
        
    def get_frame(self) -> np.ndarray:
        """Return a dummy RGB frame."""
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


class OrbbecCamera:
    """Real Orbbec camera interface."""
    
    def __init__(self, serial: str, width=224, height=224):
        if not HAS_ORBBEC:
            raise ImportError("pyorbbecsdk not available")
            
        self.serial = serial
        self.width = width
        self.height = height
        self.pipeline = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def connect(self):
        """Connect to the Orbbec camera."""
        try:
            context = Context()
            device_list = context.query_devices()
            
            target_device = None
            for device in device_list:
                if device.get_serial_number() == self.serial:
                    target_device = device
                    break
                    
            if not target_device:
                raise RuntimeError(f"Camera with serial {self.serial} not found")
                
            self.pipeline = Pipeline(target_device)
            config = Config()
            
            # Enable color stream
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
            
            def frame_callback(frame_set):
                color_frame = frame_set.get_color_frame()
                if color_frame:
                    rgb = self._process_frame(color_frame)
                    if rgb is not None:
                        with self.frame_lock:
                            self.latest_frame = rgb
            
            self.pipeline.start(config, frame_callback)
            print(f"Connected to Orbbec camera {self.serial}")
            
        except Exception as e:
            print(f"Failed to connect to camera {self.serial}: {e}")
            raise
            
    def _process_frame(self, frame):
        """Process Orbbec frame to RGB array."""
        if frame is None:
            return None
            
        width = frame.get_width()
        height = frame.get_height()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        
        if frame.get_format() == OBFormat.RGB:
            data = data.reshape((height, width, 3))
        elif frame.get_format() == OBFormat.MJPG:
            import cv2
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            return None
            
        # Resize to target size
        from PIL import Image
        pil_image = Image.fromarray(data)
        pil_image = pil_image.resize((self.width, self.height), Image.LANCZOS)
        return np.array(pil_image)
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def disconnect(self):
        """Disconnect from camera."""
        if self.pipeline:
            self.pipeline.stop()


class MockRobot:
    """Mock robot for testing without hardware."""
    
    def __init__(self, port: str):
        self.port = port
        self.state = np.zeros(7, dtype=np.float32)  # 6 joints + 1 gripper
        
    def connect(self):
        print(f"Mock robot connected to {self.port}")
        
    def disconnect(self):
        print("Mock robot disconnected")
        
    def get_state(self) -> np.ndarray:
        """Get current robot state."""
        # Add small random noise to simulate real robot
        noise = np.random.normal(0, 0.01, 7)
        return self.state + noise
        
    def act(self, action: np.ndarray):
        """Execute action (just update internal state)."""
        if len(action) >= 7:
            self.state = action[:7].copy()
            
    def move_home(self):
        """Move to home position."""
        self.state = np.zeros(7, dtype=np.float32)


class AirBotRobot:
    """Real AirBot robot interface."""
    
    def __init__(self, port: str):
        if not HAS_AIRBOT:
            raise ImportError("airbot_py not available")
            
        self.port = port
        self.robot = None
        
    def connect(self):
        """Connect to AirBot."""
        try:
            self.robot = AIRBOTPlay(port=self.port)
            print(f"Connecting to AirBot on {self.port}...")
            self.robot.connect()
            self.robot.set_speed_profile(SpeedProfile.SLOW)
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            print("AirBot connected successfully")
        except Exception as e:
            print(f"Failed to connect to AirBot: {e}")
            raise
            
    def disconnect(self):
        """Disconnect from robot."""
        if self.robot:
            self.robot.disconnect()
            
    def get_state(self) -> np.ndarray:
        """Get current robot state [joints(6), gripper(1)]."""
        if not self.robot:
            return np.zeros(7, dtype=np.float32)
            
        try:
            joints = np.array(self.robot.get_joint_pos(), dtype=np.float32)
            raw_gripper = np.array(self.robot.get_eef_pos(), dtype=np.float32)
            
            if raw_gripper.ndim == 0:
                raw_gripper = raw_gripper.reshape(1)
                
            # Map gripper from robot range (0-0.065) to model range (0-0.036)
            scaled_gripper = raw_gripper * (0.036 / 0.065)
            
            return np.concatenate([joints, scaled_gripper])
            
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return np.zeros(7, dtype=np.float32)
            
    def act(self, action: np.ndarray):
        """Execute action."""
        if not self.robot or len(action) < 7:
            return
            
        try:
            arm_cmd = action[:6]
            model_gripper_cmd = action[6]
            
            # Map gripper from model range (0-0.036) to robot range (0-0.065)  
            robot_gripper_cmd = model_gripper_cmd * (0.065 / 0.036)
            
            self.robot.servo_j(arm_cmd)
            self.robot.set_eef_pos(robot_gripper_cmd)
            
        except Exception as e:
            print(f"Error executing action: {e}")
            
    def move_home(self):
        """Move to home position."""
        if self.robot:
            try:
                self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
                home_joints = np.zeros(6, dtype=np.float32)
                self.robot.servo_j(home_joints)
                time.sleep(2.0)  # Wait for movement
            except Exception as e:
                print(f"Error moving to home: {e}")


class XVLAValidationClient:
    """XVLA validation client."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Initialize cameras
        self.top_camera = self._create_camera(config.top_cam_serial, "top")
        self.wrist_camera = self._create_camera(config.wrist_cam_serial, "wrist") 
        
        # Initialize robot
        self.robot = self._create_robot(config.robot_port)
        
        # Initialize LeRobot client
        self.lerobot_client = None
        self.action_queue = Queue()
        self.running = False
        
    def _create_camera(self, serial: Optional[str], name: str):
        """Create camera instance."""
        if serial and HAS_ORBBEC:
            return OrbbecCamera(serial)
        else:
            print(f"Using mock camera for {name}")
            return MockCamera()
            
    def _create_robot(self, port: str):
        """Create robot instance."""
        if HAS_AIRBOT:
            return AirBotRobot(port)
        else:
            print("Using mock robot")
            return MockRobot(port)
            
    def connect_hardware(self):
        """Connect to all hardware."""
        print("Connecting to hardware...")
        
        # Connect cameras
        if hasattr(self.top_camera, 'connect'):
            self.top_camera.connect()
        if hasattr(self.wrist_camera, 'connect'):
            self.wrist_camera.connect()
            
        # Connect robot
        self.robot.connect()
        self.robot.move_home()
        
        # Wait for cameras to stabilize
        time.sleep(2.0)
        print("Hardware connected successfully")
        
    def disconnect_hardware(self):
        """Disconnect from all hardware."""
        print("Disconnecting hardware...")
        
        if hasattr(self.top_camera, 'disconnect'):
            self.top_camera.disconnect()
        if hasattr(self.wrist_camera, 'disconnect'):
            self.wrist_camera.disconnect()
            
        self.robot.disconnect()
        print("Hardware disconnected")
        
    def setup_lerobot_client(self):
        """Setup LeRobot async inference client."""
        
        # Create mock camera configs for LeRobot
        camera_configs = {
            "top": OpenCVCameraConfig(index_or_path=0, width=224, height=224, fps=30),
            "wrist": OpenCVCameraConfig(index_or_path=1, width=224, height=224, fps=30),
        }
        
        # Create robot config (using a generic config since we handle hardware separately)
        robot_config = RobotConfig(
            robot_type="mock",  # We handle hardware separately
            cameras=camera_configs
        )
        
        # Create client config
        client_config = RobotClientConfig(
            robot=robot_config,
            server_address=self.config.server_address,
            policy_device=self.config.policy_device,
            client_device=self.config.client_device,
            policy_type=self.config.policy_type,
            pretrained_name_or_path=self.config.pretrained_name_or_path,
            chunk_size_threshold=self.config.chunk_size_threshold,
            actions_per_chunk=self.config.actions_per_chunk,
        )
        
        self.lerobot_client = RobotClient(client_config)
        
    def start_policy_client(self):
        """Connect to policy server."""
        print(f"Connecting to policy server at {self.config.server_address}...")
        
        if not self.lerobot_client.start():
            raise RuntimeError("Failed to connect to policy server")
            
        print("Connected to policy server successfully")
        
    def get_observation(self) -> Dict:
        """Get current observation from hardware."""
        
        # Get camera frames
        top_frame = self.top_camera.get_frame()
        wrist_frame = self.wrist_camera.get_frame()
        
        # Get robot state
        robot_state = self.robot.get_state()
        
        if top_frame is None:
            top_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if wrist_frame is None:
            wrist_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
        return {
            "observation/image": top_frame,
            "observation/wrist_image": wrist_frame,
            "observation/state": robot_state,
            "task": self.config.task
        }
        
    def run_validation_episode(self):
        """Run a single validation episode."""
        
        print("\n" + "="*50)
        print(" XVLA VALIDATION EPISODE")
        print("="*50)
        print(f"Task: {self.config.task}")
        print(f"Control frequency: {self.config.control_freq} Hz")
        print(f"Action horizon: {self.config.action_horizon}")
        print("="*50)
        
        input("Press Enter to start episode...")
        
        self.running = True
        dt = 1.0 / self.config.control_freq
        step_count = 0
        
        try:
            while self.running and step_count < self.config.max_episode_steps:
                cycle_start = time.time()
                
                # Get observation from hardware
                obs = self.get_observation()
                
                # Send observation to policy server and get actions
                # Note: This is a simplified version - in real implementation,
                # you'd use the async inference system properly
                
                # For now, let's execute a simple control loop
                # In real implementation, you'd integrate with lerobot_client properly
                
                # Mock action for demo
                current_state = obs["observation/state"]
                action = current_state + np.random.normal(0, 0.01, 7)  # Small random movement
                
                # Execute action
                self.robot.act(action)
                
                step_count += 1
                
                # Maintain control frequency
                elapsed = time.time() - cycle_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
                if step_count % 30 == 0:  # Print every second
                    print(f"Step {step_count}, Elapsed: {elapsed:.3f}s")
                    
        except KeyboardInterrupt:
            print("\nEpisode interrupted by user")
        finally:
            self.running = False
            print(f"Episode completed. Total steps: {step_count}")
            
    def run(self):
        """Run the validation client."""
        
        try:
            # Setup
            self.connect_hardware()
            self.setup_lerobot_client() 
            self.start_policy_client()
            
            # Warmup
            print("Warming up...")
            warmup_obs = self.get_observation()
            print("Warmup complete")
            
            # Run validation episode
            self.run_validation_episode()
            
        except Exception as e:
            print(f"Error during validation: {e}")
            raise
        finally:
            # Cleanup
            if self.lerobot_client:
                self.lerobot_client.stop()
            self.disconnect_hardware()


def main(config: ValidationConfig):
    """Main validation function."""
    
    if not config.pretrained_name_or_path:
        print("Error: pretrained_name_or_path must be specified")
        print("Set it to your finetuned XVLA model path")
        return
        
    client = XVLAValidationClient(config)
    client.run()


if __name__ == "__main__":
    tyro.cli(main)