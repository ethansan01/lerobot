"""
AIRBOT Robot Client for ACT Models - LeRobot Async Inference

This client is specifically configured for ACT models with:
- DualOrbbecStreamer for camera input (top and wrist cameras)
- AirBotArm for robot control
- Rerun for visualization
- Camera mappings: top -> top, wrist -> wrist (ACT standard)

Usage:
1. Start the policy server:
   uv run -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080

2. Run this client:
   python examples/validation/airbot_robot_client_act.py --pretrained /path/to/act/checkpoint
"""

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rerun as rr
from PIL import Image

# Hardware Libraries
from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline

# Robot Library
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)


# =============================================================================
# Camera Logic (from validation_gripper.py) with Rerun logging
# =============================================================================

def process_color_frame(frame):
    """Decodes Orbbec color frame to numpy RGB array."""
    import cv2

    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if frame.get_format() == OBFormat.RGB:
        data = data.reshape((height, width, 3))
    elif frame.get_format() == OBFormat.MJPG:
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    else:
        return None

    return data


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image using PIL for high quality."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(size, resample=Image.BICUBIC)
    return np.array(resized)


class DualOrbbecStreamer:
    """
    Manages two Orbbec cameras (Top and Wrist).
    Constantly fetches frames in background threads and logs to Rerun.
    """

    def __init__(self, top_serial: str, wrist_serial: str, image_size: tuple[int, int] = (640, 480)):
        self.target_serials = {"top": top_serial, "wrist": wrist_serial}
        self.image_size = image_size
        self.pipelines = []
        self.latest_frames = {"top": None, "wrist": None}
        self.lock = threading.Lock()
        self.ctx = Context()

    def connect(self):
        device_list = self.ctx.query_devices()
        dev_count = device_list.get_count()
        print(f"[CAM] Found {dev_count} devices.")

        for i in range(dev_count):
            device = device_list.get_device_by_index(i)

            serial = None
            try:
                info_repr = repr(device.get_device_info())
                m = re.search(r"serial_number=([^,\s\)]+)", info_repr)
                if m:
                    serial = m.group(1).strip()
            except Exception:
                pass

            if not serial:
                print(f"[CAM] Could not identify serial for device {i}")
                continue

            role = None
            if serial == self.target_serials["top"]:
                role = "top"
            elif serial == self.target_serials["wrist"]:
                role = "wrist"

            if role:
                print(f"[CAM] Connecting to {role.upper()} camera (Serial: {serial})")
                self._start_pipeline(device, role)
            else:
                print(f"[CAM] Ignoring device {serial} (not in target list)")

    def _start_pipeline(self, device, role):
        pipeline = Pipeline(device)
        config = Config()

        try:
            profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
        except Exception as e:
            print(f"[CAM] Failed to enable color for {role}: {e}")
            return

        def callback(frame_set):
            color_frame = frame_set.get_color_frame()
            if color_frame:
                rgb = process_color_frame(color_frame)
                if rgb is not None:
                    # Resize to target size
                    rgb = resize_image(rgb, self.image_size)
                    with self.lock:
                        self.latest_frames[role] = rgb

                    # Log to Rerun
                    try:
                        rr.log(f"camera/{role}", rr.Image(rgb), static=True)
                    except Exception:
                        pass

        pipeline.start(config, callback)
        self.pipelines.append(pipeline)

    def get_latest(self) -> dict[str, np.ndarray | None]:
        """Returns the most recent frames."""
        with self.lock:
            return {k: v.copy() if v is not None else None for k, v in self.latest_frames.items()}

    def stop(self):
        for p in self.pipelines:
            try:
                p.stop()
            except Exception:
                pass


# =============================================================================
# Robot Logic (from validation_gripper.py)
# =============================================================================


class AirBotArm:
    """AIRBOT arm controller with gripper mapping."""

    def __init__(self, port: str):
        self.port = port
        self.robot = None

    def connect(self):
        self.robot = AIRBOTPlay(port=self.port)
        print(f"[ROBOT] Connecting to AirBot on port {self.port}...")
        self.robot.connect()
        self.robot.set_speed_profile(SpeedProfile.SLOW)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()

    def get_state(self) -> np.ndarray:
        """Returns concatenated [arm_qpos(6), gripper_pos(1)]"""
        if not self.robot:
            return np.zeros(7, dtype=np.float32)

        joints = np.array(self.robot.get_joint_pos(), dtype=np.float32)

        raw_gripper = np.array(self.robot.get_eef_pos(), dtype=np.float32)
        if raw_gripper.ndim == 0:
            raw_gripper = raw_gripper.reshape(1)

        # Map to model range (0.0 - 0.036)
        scaled_gripper = self._map_to_model(raw_gripper)

        return np.concatenate([joints, scaled_gripper])

    def act(self, action: np.ndarray):
        """Expects 7D action: [arm(6), gripper(1)]"""
        if not self.robot:
            print("[ROBOT] Warning: robot not connected!")
            return

        arm_cmd = action[:6]
        model_gripper_cmd = action[6]

        # Map back to robot range (0.0 - 0.065)
        robot_gripper_cmd = self._map_to_robot(model_gripper_cmd)
        robot_gripper_cmd = np.clip(robot_gripper_cmd, 0.0, 0.065)

        # Log action to rerun
        try:
            rr.log("robot/arm_cmd", rr.BarChart(arm_cmd))
            rr.log("robot/gripper_cmd", rr.Scalar(robot_gripper_cmd))
        except Exception:
            pass

        print(f"[ROBOT] Arm: {arm_cmd}, Gripper: {robot_gripper_cmd:.4f}")
        # Convert numpy types to Python native types for airbot_py
        self.robot.servo_joint_pos(arm_cmd.tolist())
        self.robot.servo_eef_pos([float(robot_gripper_cmd)])

    def move_home(self):
        if not self.robot:
            return
        home = [-0.32673379778862, -1.8205920457839966, 0.4194323718547821, -1.8614099025726318, -1.3105592727661133, -0.7951858043670654]

        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.move_to_joint_pos(home, blocking=True)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        self.robot.servo_eef_pos([0.065])  # Open gripper

    def _map_to_model(self, gripper_pos):
        """Maps robot range [0.0, 0.065] -> model range [0.0, 0.036]"""
        return (gripper_pos / 0.065) * 0.036

    def _map_to_robot(self, gripper_cmd):
        """Maps model range [0.0, 0.036] -> robot range [0.0, 0.065]"""
        return (gripper_cmd / 0.036) * 0.065


# =============================================================================
# AIRBOT Client Config
# =============================================================================


@dataclass
class AirbotClientConfig:
    """Configuration for AIRBOT async inference client (ACT models)."""

    # Policy configuration
    policy_type: str = "act"
    pretrained_name_or_path: str = "path/to/your/act/checkpoint"

    # Network configuration
    server_address: str = "127.0.0.1:8080"

    # Device configuration
    policy_device: str = "cuda"
    client_device: str = "cpu"

    # Control behavior
    chunk_size_threshold: float = 0.5
    fps: int = 30
    actions_per_chunk: int = 30

    # Robot configuration
    arm_port: str = "50001"

    # Camera serials
    top_cam_serial: str = "CP7JC42000EY"
    wrist_cam_serial: str = "CP7JC42000F4"

    # Camera image size (width, height)
    image_width: int = 640
    image_height: int = 480

    # Task prompt
    task: str = "pick up the red block and place it in the bowl"

    # Debug
    debug_visualize_queue_size: bool = False

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps


# =============================================================================
# AIRBOT Async Inference Client
# =============================================================================


class AirbotAsyncClient:
    """
    AIRBOT client that interfaces with LeRobot's async inference policy server.

    This client:
    1. Captures images from dual Orbbec cameras
    2. Reads robot state from AIRBOT arm
    3. Sends observations to the policy server via gRPC
    4. Receives action chunks and executes them on the robot
    5. Logs everything to Rerun for visualization
    """

    prefix = "airbot_client"
    logger = get_logger(prefix)

    def __init__(self, config: AirbotClientConfig):
        self.config = config

        # Initialize cameras
        self.cameras = DualOrbbecStreamer(
            top_serial=config.top_cam_serial,
            wrist_serial=config.wrist_cam_serial,
            image_size=(config.image_width, config.image_height),
        )

        # Initialize robot
        self.robot = AirBotArm(port=config.arm_port)

        # Build lerobot_features dict for the policy server
        self.lerobot_features = self._build_lerobot_features()

        # gRPC setup
        import grpc
        from lerobot.transport import services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=self.lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
        )

        # Threading
        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self.action_queue = __import__("queue").Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()

        self.logger.info(f"AIRBOT client initialized, connecting to {config.server_address}")

    def _build_lerobot_features(self) -> dict[str, dict]:
        """Build the lerobot_features dict expected by the policy server.
        
        ACT models use descriptive camera names: top -> top, wrist -> wrist
        Returns dict[str, dict] format matching hw_to_dataset_features output.
        """
        return {
            "observation.images.top": {
                "dtype": "image",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state_0", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6"],
            },
        }

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def connect(self):
        """Connect to cameras and robot."""
        self.logger.info("Connecting to cameras...")
        self.cameras.connect()
        time.sleep(2.0)  # Wait for auto-exposure

        self.logger.info("Connecting to robot...")
        self.robot.connect()
        self.robot.move_home()

        self.logger.info("Hardware connected and ready")

    def start(self) -> bool:
        """Start the client and connect to the policy server."""
        import pickle
        import grpc
        from lerobot.transport import services_pb2

        try:
            # Handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            self.logger.debug(f"Connected to policy server in {time.perf_counter() - start_time:.4f}s")

            # Send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the client."""
        self.shutdown_event.set()
        self.cameras.stop()
        self.robot.disconnect()
        self.channel.close()
        self.logger.info("Client stopped")

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from cameras and robot."""
        frames = self.cameras.get_latest()
        robot_state = self.robot.get_state()

        # Log state to rerun
        try:
            rr.log("robot/state", rr.BarChart(robot_state))
        except Exception:
            pass

        # Build observation with individual state components
        # ACT models use descriptive camera names
        obs = {
            "top": frames["top"],
            "wrist": frames["wrist"],
            "task": self.config.task,
        }
        
        # Add individual state components
        for i in range(7):
            obs[f"state_{i}"] = float(robot_state[i])
        
        return obs

    def send_observation(self, obs: TimedObservation) -> bool:
        """Send observation to the policy server."""
        import pickle
        import grpc
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import send_bytes_in_chunks

        if not self.running:
            raise RuntimeError("Client not running")

        observation_bytes = pickle.dumps(obs)

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            self.logger.debug(f"Sent observation #{obs.get_timestep()}")
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation: {e}")
            return False

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server (runs in background thread)."""
        import pickle
        import grpc
        from lerobot.transport import services_pb2

        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions = pickle.loads(actions_chunk.data)

                # Move to client device if needed
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Add to action queue
                with self.action_queue_lock:
                    for action in timed_actions:
                        with self.latest_action_lock:
                            if action.get_timestep() > self.latest_action:
                                self.action_queue.put(action)

                self.must_go.set()

                if verbose and len(timed_actions) > 0:
                    self.logger.info(
                        f"Received {len(timed_actions)} actions for step #{timed_actions[0].get_timestep()}"
                    )

                # Log queue size to rerun
                try:
                    rr.log("queue/size", rr.Scalar(self.action_queue.qsize()))
                except Exception:
                    pass

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self) -> bool:
        """Check if there are actions available."""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        """Check if we should send a new observation."""
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self.config.chunk_size_threshold

    def control_loop(self, verbose: bool = False):
        """Main control loop: execute actions and stream observations."""
        self.start_barrier.wait()
        self.logger.info("Control loop starting")

        step_count = 0
        while self.running:
            control_loop_start = time.perf_counter()

            # 1. Execute action if available
            if self.actions_available():
                with self.action_queue_lock:
                    self.action_queue_size.append(self.action_queue.qsize())
                    timed_action = self.action_queue.get_nowait()

                # Convert tensor to numpy and execute
                # Convert BFloat16 to float32 first (numpy doesn't support bfloat16)
                action_tensor = timed_action.get_action().detach().cpu()
                if action_tensor.dtype == __import__("torch").bfloat16:
                    action_tensor = action_tensor.float()
                action_np = action_tensor.numpy()
                self.robot.act(action_np)

                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()

                step_count += 1
                if verbose:
                    self.logger.info(f"Executed action #{timed_action.get_timestep()} (step {step_count})")

            # 2. Send observation if ready
            if self._ready_to_send_observation():
                raw_obs = self.get_observation()

                # Check if frames are available
                if raw_obs["top"] is None or raw_obs["wrist"] is None:
                    self.logger.debug("Waiting for camera frames...")
                    time.sleep(0.01)
                    continue

                with self.latest_action_lock:
                    latest_action = self.latest_action

                obs = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_obs,
                    timestep=max(latest_action, 0),
                )

                with self.action_queue_lock:
                    obs.must_go = self.must_go.is_set() and self.action_queue.empty()

                self.send_observation(obs)

                if obs.must_go:
                    self.must_go.clear()

            # Sleep to maintain control frequency
            elapsed = time.perf_counter() - control_loop_start
            time.sleep(max(0, self.config.environment_dt - elapsed))


def main():
    """Main entry point for AIRBOT async inference client (ACT models)."""
    import argparse

    parser = argparse.ArgumentParser(description="AIRBOT Async Inference Client for ACT Models")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Policy server address")
    parser.add_argument("--policy-type", type=str, default="act", help="Policy type (default: act)")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained ACT model path")
    parser.add_argument("--arm-port", type=str, default="50001", help="AIRBOT arm port")
    parser.add_argument("--top-cam", type=str, default="CP7JC42000EY", help="Top camera serial")
    parser.add_argument("--wrist-cam", type=str, default="CP7JC42000F4", help="Wrist camera serial")
    parser.add_argument("--task", type=str, default="pick up the red block and place it in the bowl")
    parser.add_argument("--fps", type=int, default=30, help="Control frequency")
    parser.add_argument("--actions-per-chunk", type=int, default=30, help="Actions per chunk")
    parser.add_argument("--policy-device", type=str, default="cuda", help="Policy device")
    parser.add_argument("--debug-queue", action="store_true", help="Visualize action queue size")
    args = parser.parse_args()

    # Initialize Rerun
    rr.init("airbot_async_client", spawn=True)

    config = AirbotClientConfig(
        server_address=args.server,
        policy_type=args.policy_type,
        pretrained_name_or_path=args.pretrained,
        arm_port=args.arm_port,
        top_cam_serial=args.top_cam,
        wrist_cam_serial=args.wrist_cam,
        task=args.task,
        fps=args.fps,
        actions_per_chunk=args.actions_per_chunk,
        policy_device=args.policy_device,
        debug_visualize_queue_size=args.debug_queue,
    )

    client = AirbotAsyncClient(config)

    print("\n" + "=" * 50)
    print(" AIRBOT Async Inference Client (ACT)")
    print("=" * 50)
    print(f" Server: {config.server_address}")
    print(f" Policy: {config.policy_type} ({config.pretrained_name_or_path})")
    print(f" Task: {config.task}")
    print(f" Camera mapping: top->top, wrist->wrist")
    print("=" * 50)

    # Connect hardware
    client.connect()

    input("\nPress Enter to start inference...")

    if client.start():
        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.receive_actions, kwargs={"verbose": True}, daemon=True
        )
        action_receiver_thread.start()

        try:
            # Run control loop in main thread
            client.control_loop(verbose=True)
        except KeyboardInterrupt:
            print("\n[SYS] Stopping...")
        finally:
            client.stop()
            action_receiver_thread.join(timeout=2.0)

            if config.debug_visualize_queue_size and client.action_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            print("[SYS] Clean shutdown.")


if __name__ == "__main__":
    main()
