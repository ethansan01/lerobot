#!/usr/bin/env python3
"""
XVLA Validation Client for LeRobot
=================================

This client connects to a LeRobot policy server and validates XVLA model performance
with real robot hardware. Uses LeRobot's async inference system properly.

Usage:
    # Terminal 1 (Server environment): 
    python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
    
    # Terminal 2 (Client environment):
    python examples/xvla_validation_client_proper.py \
        --pretrained_name_or_path="your_username/xvla_finetuned_model" \
        --server_address="127.0.0.1:8080" \
        --robot_port="/dev/ttyUSB0" \
        --task="pick up the red block and place it in the bowl"

Environment Setup:
    # Server environment (with GPU):
    conda create -n lerobot_server python=3.10
    conda activate lerobot_server  
    pip install lerobot[all]
    
    # Client environment (with hardware APIs):
    conda create -n lerobot_client python=3.10
    conda activate lerobot_client
    pip install lerobot pyorbbecsdk airbot_py rerun-sdk
"""

import time
import threading
import numpy as np
import tyro
import dataclasses
import pickle
from typing import Optional, Dict, Any
from queue import Queue, Empty

# Hardware Libraries (client environment only)
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

# LeRobot imports
import grpc
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.constants import SUPPORTED_POLICIES


@dataclasses.dataclass  
class ValidationConfig:
    """Configuration for XVLA validation client."""
    
    # Server connection
    server_address: str = "127.0.0.1:8080"
    
    # Model configuration
    policy_type: str = "xvla"
    pretrained_name_or_path: str = "/home/datafactory/Downloads/xvla/checkpoints"  # Set this to your finetuned model path
    policy_device: str = "cuda"
    actions_per_chunk: int = 50
    
    # Robot hardware
    robot_port: str = "/dev/ttyUSB0"
    top_cam_serial: Optional[str] = None 
    wrist_cam_serial: Optional[str] = None
    
    # Task
    task: str = "pick up the red block and place it in the bowl"
    
    # Control 
    control_freq: int = 30  # Hz
    max_episode_steps: int = 1000
    
    # Action extraction (for different robot configurations)
    arm_action_dim: int = 6  # Joint positions
    gripper_action_dim: int = 1  # Gripper position
    action_offset: int = 0  # If actions start at different index


class MockCamera:
    """Mock camera for testing."""
    def get_frame(self) -> np.ndarray:
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


class OrbbecCamera:
    """Orbbec camera interface."""
    def __init__(self, serial: str):
        self.serial = serial
        self.pipeline = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def connect(self):
        """Connect to camera."""
        try:
            context = Context()
            device_list = context.query_devices()
            
            target_device = None
            for device in device_list:
                if device.get_serial_number() == self.serial:
                    target_device = device
                    break
                    
            if not target_device:
                raise RuntimeError(f"Camera {self.serial} not found")
                
            self.pipeline = Pipeline(target_device)
            config = Config()
            
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
            
            def callback(frame_set):
                color_frame = frame_set.get_color_frame()
                if color_frame:
                    rgb = self._process_frame(color_frame)
                    if rgb is not None:
                        with self.frame_lock:
                            self.latest_frame = rgb
                            
            self.pipeline.start(config, callback)
            print(f"Connected to camera {self.serial}")
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            raise
            
    def _process_frame(self, frame):
        """Process frame to RGB."""
        if frame is None:
            return None
            
        width, height = frame.get_width(), frame.get_height()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        
        if frame.get_format() == OBFormat.RGB:
            data = data.reshape((height, width, 3))
        elif frame.get_format() == OBFormat.MJPG:
            import cv2
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            return None
            
        # Resize to 224x224
        from PIL import Image
        pil_image = Image.fromarray(data)
        pil_image = pil_image.resize((224, 224), Image.LANCZOS)
        return np.array(pil_image)
        
    def get_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def disconnect(self):
        if self.pipeline:
            self.pipeline.stop()


class MockRobot:
    """Mock robot for testing."""
    def __init__(self, port: str):
        self.port = port
        self.state = np.zeros(7, dtype=np.float32)
        
    def connect(self):
        print(f"Mock robot connected to {self.port}")
        
    def disconnect(self):
        print("Mock robot disconnected")
        
    def get_state(self) -> np.ndarray:
        noise = np.random.normal(0, 0.01, 7)
        return self.state + noise
        
    def execute_action(self, action: np.ndarray):
        if len(action) >= 7:
            self.state = action[:7].copy()
            
    def move_home(self):
        self.state = np.zeros(7, dtype=np.float32)


class AirBotRobot:
    """AirBot robot interface."""
    def __init__(self, port: str):
        self.port = port
        self.robot = None
        
    def connect(self):
        try:
            self.robot = AIRBOTPlay(port=self.port)
            self.robot.connect()
            self.robot.set_speed_profile(SpeedProfile.SLOW)
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            print(f"AirBot connected on {self.port}")
        except Exception as e:
            print(f"Failed to connect AirBot: {e}")
            raise
            
    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
            
    def get_state(self) -> np.ndarray:
        """Get [joints(6), gripper(1)]."""
        if not self.robot:
            return np.zeros(7, dtype=np.float32)
            
        try:
            joints = np.array(self.robot.get_joint_pos(), dtype=np.float32)
            raw_gripper = np.array(self.robot.get_eef_pos(), dtype=np.float32)
            
            if raw_gripper.ndim == 0:
                raw_gripper = raw_gripper.reshape(1)
                
            # Map gripper 0-0.065 -> 0-0.036 (model range)
            scaled_gripper = raw_gripper * (0.036 / 0.065)
            
            return np.concatenate([joints, scaled_gripper])
            
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return np.zeros(7, dtype=np.float32)
            
    def execute_action(self, action: np.ndarray):
        """Execute action on robot."""
        if not self.robot or len(action) < 7:
            return
            
        try:
            arm_cmd = action[:6]
            model_gripper = action[6] 
            
            # Map gripper 0-0.036 -> 0-0.065 (robot range)
            robot_gripper = model_gripper * (0.065 / 0.036)
            
            self.robot.servo_j(arm_cmd)
            self.robot.set_eef_pos(robot_gripper)
            
        except Exception as e:
            print(f"Error executing action: {e}")
            
    def move_home(self):
        if self.robot:
            try:
                self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
                home_joints = np.zeros(6, dtype=np.float32)
                self.robot.servo_j(home_joints)
                time.sleep(2.0)
            except Exception as e:
                print(f"Error moving home: {e}")


class XVLAValidationClient:
    """XVLA validation client using LeRobot async inference."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Hardware
        self.top_camera = self._create_camera(config.top_cam_serial, "top")
        self.wrist_camera = self._create_camera(config.wrist_cam_serial, "wrist")
        self.robot = self._create_robot(config.robot_port)
        
        # LeRobot client
        self.channel = None
        self.stub = None
        self.action_queue = Queue()
        self.running = False
        
    def _create_camera(self, serial: Optional[str], name: str):
        if serial and HAS_ORBBEC:
            return OrbbecCamera(serial)
        else:
            print(f"Using mock camera for {name}")
            return MockCamera()
            
    def _create_robot(self, port: str):
        if HAS_AIRBOT:
            return AirBotRobot(port)
        else:
            return MockRobot(port)
            
    def connect_hardware(self):
        """Connect all hardware."""
        print("Connecting hardware...")
        
        if hasattr(self.top_camera, 'connect'):
            self.top_camera.connect()
        if hasattr(self.wrist_camera, 'connect'):
            self.wrist_camera.connect()
            
        self.robot.connect()
        self.robot.move_home()
        
        time.sleep(2.0)  # Camera stabilization
        print("Hardware connected")
        
    def disconnect_hardware(self):
        """Disconnect all hardware."""
        print("Disconnecting hardware...")
        
        if hasattr(self.top_camera, 'disconnect'):
            self.top_camera.disconnect()
        if hasattr(self.wrist_camera, 'disconnect'):
            self.wrist_camera.disconnect()
            
        self.robot.disconnect()
        
    def connect_to_server(self):
        """Connect to LeRobot policy server."""
        print(f"Connecting to server at {self.config.server_address}")
        
        try:
            self.channel = grpc.insecure_channel(self.config.server_address)
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            
            # Server handshake
            self.stub.Ready(services_pb2.Empty())
            print("Connected to server")
            
            # Send policy configuration
            policy_config = RemotePolicyConfig(
                policy_type=self.config.policy_type,
                pretrained_name_or_path=self.config.pretrained_name_or_path,
                lerobot_features={},  # Will be filled by server based on model
                actions_per_chunk=self.config.actions_per_chunk,
                device=self.config.policy_device,
            )
            
            policy_bytes = pickle.dumps(policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_bytes)
            
            print(f"Sending policy config:")
            print(f"  Type: {self.config.policy_type}")
            print(f"  Model: {self.config.pretrained_name_or_path}")
            print(f"  Device: {self.config.policy_device}")
            
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy loaded on server")
            
            return True
            
        except grpc.RpcError as e:
            print(f"Failed to connect to server: {e}")
            return False
            
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        top_frame = self.top_camera.get_frame()
        wrist_frame = self.wrist_camera.get_frame()
        robot_state = self.robot.get_state()
        
        if top_frame is None:
            top_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if wrist_frame is None:
            wrist_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
        return {
            "observation": {
                "image": top_frame,
                "wrist_image": wrist_frame, 
                "state": robot_state,
            },
            "task": self.config.task,
        }
        
    def send_observation_get_actions(self, obs_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """Send observation to server and get actions."""
        try:
            # Create timed observation
            timed_obs = TimedObservation(
                timestamp=time.time(),
                observation=obs_dict
            )
            
            # Send to server
            obs_bytes = pickle.dumps(timed_obs)
            obs_msg = services_pb2.Observation(data=obs_bytes)
            
            # Get actions
            response = self.stub.SendObservationAndReceiveAction(obs_msg)
            
            if response.data:
                timed_actions = pickle.loads(response.data)
                actions = timed_actions.actions  # Shape: [chunk_size, action_dim]
                
                # Extract relevant actions for our robot
                # Actions typically include arm + gripper, we extract what we need
                start_idx = self.config.action_offset
                end_idx = start_idx + self.config.arm_action_dim + self.config.gripper_action_dim
                
                robot_actions = actions[:, start_idx:end_idx]  # [chunk_size, 7]
                return robot_actions
                
            return None
            
        except Exception as e:
            print(f"Error in inference: {e}")
            return None
            
    def run_episode(self):
        """Run validation episode."""
        print("\n" + "="*50)
        print("XVLA VALIDATION EPISODE")
        print("="*50)
        print(f"Task: {self.config.task}")
        print(f"Control freq: {self.config.control_freq} Hz")
        print("="*50)
        
        input("Press Enter to start episode...")
        
        self.running = True
        dt = 1.0 / self.config.control_freq
        step_count = 0
        
        try:
            while self.running and step_count < self.config.max_episode_steps:
                cycle_start = time.time()
                
                # Get observation
                obs = self.get_observation()
                
                # Get actions from server
                actions = self.send_observation_get_actions(obs)
                
                if actions is not None and len(actions) > 0:
                    # Execute first action in chunk
                    current_action = actions[0]  # Shape: [7]
                    self.robot.execute_action(current_action)
                    
                    if step_count % 30 == 0:
                        print(f"Step {step_count}, Action: {current_action[:3]:.3f}...")
                else:
                    print(f"No action received at step {step_count}")
                
                step_count += 1
                
                # Maintain frequency
                elapsed = time.time() - cycle_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print(f"\nEpisode stopped by user at step {step_count}")
        finally:
            self.running = False
            
    def run(self):
        """Main run loop."""
        try:
            # Setup
            self.connect_hardware()
            
            if not self.connect_to_server():
                return
                
            # Warmup
            print("Warming up...")
            warmup_obs = self.get_observation()
            self.send_observation_get_actions(warmup_obs)
            print("Warmup complete")
            
            # Run episode
            self.run_episode()
            
        except Exception as e:
            print(f"Error: {e}")
            raise
        finally:
            if self.channel:
                self.channel.close()
            self.disconnect_hardware()


def main(config: ValidationConfig):
    """Main function."""
    if not config.pretrained_name_or_path:
        print("Error: pretrained_name_or_path must be specified")
        print("Example: your_username/xvla_finetuned_model")
        return
        
    if config.policy_type not in SUPPORTED_POLICIES:
        print(f"Error: Policy {config.policy_type} not supported")
        print(f"Supported: {SUPPORTED_POLICIES}")
        return
        
    client = XVLAValidationClient(config)
    client.run()


if __name__ == "__main__":
    tyro.cli(main)