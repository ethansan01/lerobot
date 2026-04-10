"""
AIRBOT + Revo2 Bimanual Robot Client for LeRobot Async Inference

This client extends the single-arm airbot_robot_revo2_client.py to support
bimanual operation with two AIRBOT arms and two Revo2 dexterous hands.

Observation/state layout:
    [left_arm_qpos(6), right_arm_qpos(6), left_hand_qpos(6), right_hand_qpos(6)]

Action layout:
    [left_arm_cmd(6), right_arm_cmd(6), left_hand_cmd(6), right_hand_cmd(6)]

Camera views:
    - top: overhead view
    - wrist_left: left arm wrist camera
    - wrist_right: right arm wrist camera

Usage:
1. Start the policy server:
   uv run -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080

2. Run this client:
   python examples/validation/airbot_revo2_bimanual_client.py \
       --pretrained /path/to/checkpoint \
       --left-arm-port 50001 \
       --left-hand-port /dev/ttyUSB0 \
       --right-arm-port 50002 \
       --right-hand-port /dev/ttyUSB1
"""

import asyncio
import concurrent.futures
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr

from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline

from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)

from multistreams import process_color


REVO2_LIBRARY_ROOT = (
    Path(__file__).resolve().parents[2].parent
    / "data_factory"
    / "motion_retargeting"
    / "motion_retargeting"
    / "hand"
    / "revo2_library"
    / "python"
)

if str(REVO2_LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(REVO2_LIBRARY_ROOT))

from revo2.revo2_utils import libstark, open_modbus_revo2  # noqa: E402


HAND_DOF = 6
ARM_DOF = 6
HAND_LIMITS = np.array([1.57, 1.03, 1.41, 1.41, 1.41, 1.41], dtype=np.float32)


class TripleOrbbecStreamer:
    """Manages the top, wrist_left, and wrist_right Orbbec cameras."""

    def __init__(
        self,
        top_serial: str,
        wrist_left_serial: str,
        wrist_right_serial: str,
        image_size: tuple[int, int] = (640, 480),
    ):
        self.target_serials = {
            "top": top_serial,
            "wrist_left": wrist_left_serial,
            "wrist_right": wrist_right_serial,
        }
        self.image_size = image_size
        self.pipelines = []
        self.latest_frames = {"top": None, "wrist_left": None, "wrist_right": None}
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
                match = re.search(r"serial_number=([^,\s\)]+)", info_repr)
                if match:
                    serial = match.group(1).strip()
            except Exception:
                pass

            if not serial:
                print(f"[CAM] Could not identify serial for device {i}")
                continue

            role = None
            for role_name, target_serial in self.target_serials.items():
                if serial == target_serial:
                    role = role_name
                    break

            if role is None:
                print(f"[CAM] Ignoring device {serial} (not in target list)")
                continue

            print(f"[CAM] Connecting to {role.upper()} camera (Serial: {serial})")
            self._start_pipeline(device, role)

    def _start_pipeline(self, device, role):
        pipeline = Pipeline(device)
        config = Config()

        try:
            profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            config.enable_stream(color_profile)
        except Exception as exc:
            print(f"[CAM] Failed to enable color for {role}: {exc}")
            return

        try:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
            config.enable_stream(depth_profile)
        except Exception:
            pass

        def callback(frame_set):
            color_frame = frame_set.get_color_frame()
            if color_frame is None:
                return

            image = process_color(color_frame)
            if image is None:
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[1] != self.image_size[0] or image.shape[0] != self.image_size[1]:
                return

            with self.lock:
                self.latest_frames[role] = image

            try:
                rr.log(f"camera/{role}", rr.Image(image), static=True)
            except Exception:
                pass

        pipeline.start(config, callback)
        self.pipelines.append(pipeline)

    def get_latest(self) -> dict[str, np.ndarray | None]:
        with self.lock:
            return {name: frame.copy() if frame is not None else None for name, frame in self.latest_frames.items()}

    def stop(self):
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass


LEFT_ARM_HOME = [-0.1993209719657898, -0.9321355223655701, 0.33436331152915955, 1.3670176267623901, 1.4566643238067627, 0.08869306743144989]
RIGHT_ARM_HOME = [0.00019073777366429567, -0.8463035225868225, 0.34924086928367615, -1.5402075052261353, -1.2502861022949219, -0.00019073777366429567]


class AirBotArm:
    """AIRBOT arm controller."""

    def __init__(self, port: str, name: str = "arm", home: list[float] | None = None):
        self.port = port
        self.name = name
        self.home = home
        self.robot = None

    def connect(self):
        self.robot = AIRBOTPlay(port=self.port)
        print(f"[ROBOT] Connecting to AirBot {self.name} on port {self.port}...")
        self.robot.connect()
        self.robot.set_speed_profile(SpeedProfile.SLOW)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()

    def get_state(self) -> np.ndarray:
        if not self.robot:
            return np.zeros(ARM_DOF, dtype=np.float32)
        return np.array(self.robot.get_joint_pos(), dtype=np.float32)

    def act(self, action: np.ndarray):
        if not self.robot:
            print(f"[ROBOT] Warning: {self.name} not connected!")
            return
        # print(f"[ROBOT] {self.name} cmd: {np.array2string(action[:ARM_DOF], precision=4, suppress_small=True)}")
        self.robot.servo_joint_pos(action[:ARM_DOF].tolist())

    def move_home(self):
        if not self.robot or self.home is None:
            return

        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.move_to_joint_pos(self.home, blocking=True)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)


class Revo2Hand:
    """Minimal synchronous wrapper around the Revo2 Modbus API."""

    def __init__(self, port: str | None, name: str = "hand"):
        self.port = port
        self.name = name
        self.client = None
        self.slave_id = None
        self.loop = None
        self.loop_thread = None
        self.connected = False

    def _start_loop(self):
        if self.loop is not None:
            return

        self.loop = asyncio.new_event_loop()

        def runner():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=runner, daemon=True, name=f"revo2-loop-{self.name}")
        self.loop_thread.start()

    def _run(self, coro):
        if self.loop is None:
            raise RuntimeError("Revo2 event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=5.0)

    async def _connect_async(self):
        client, slave_id = await open_modbus_revo2(port_name=self.port, quick=True)
        await client.set_finger_unit_mode(slave_id, libstark.FingerUnitMode.Normalized)
        self.client = client
        self.slave_id = slave_id

    async def _disconnect_async(self):
        if self.client is not None:
            await libstark.modbus_close(self.client)
            await asyncio.sleep(0.1)

    async def _get_positions_async(self):
        return await self.client.get_finger_positions(self.slave_id)

    async def _set_positions_async(self, positions: list[int]):
        await self.client.set_finger_positions_and_speeds(
            self.slave_id,
            positions,
            [1000] * HAND_DOF,
        )

    def connect(self):
        print(f"[HAND] Connecting to Revo2 {self.name} on port {self.port}...")
        self._start_loop()
        self._run(self._connect_async())
        self.connected = True

    def disconnect(self):
        if not self.connected:
            return
        try:
            self._run(self._disconnect_async())
        finally:
            self.connected = False
            self.client = None
            self.slave_id = None
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread is not None:
                self.loop_thread.join(timeout=1.0)
            self.loop = None
            self.loop_thread = None

    def _map_to_model(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        return np.clip(positions, 0.0, 1000.0) / 1000.0 * HAND_LIMITS

    def _map_to_robot(self, qpos: np.ndarray) -> list[int]:
        qpos = np.asarray(qpos, dtype=np.float32)
        qpos = np.clip(qpos, 0.0, HAND_LIMITS)
        return np.rint((qpos / HAND_LIMITS) * 1000.0).astype(np.int32).tolist()

    def get_state(self) -> np.ndarray:
        if not self.connected:
            return np.zeros(HAND_DOF, dtype=np.float32)

        try:
            raw_positions = self._run(self._get_positions_async())
            return self._map_to_model(np.asarray(raw_positions, dtype=np.float32))
        except concurrent.futures.TimeoutError:
            print(f"[HAND] Timed out reading Revo2 {self.name} joint positions")
        except Exception as exc:
            print(f"[HAND] Failed to read Revo2 {self.name} joint positions: {exc}")
        return np.zeros(HAND_DOF, dtype=np.float32)

    def act(self, action: np.ndarray):
        if not self.connected:
            print(f"[HAND] Warning: {self.name} not connected!")
            return

        try:
            robot_positions = self._map_to_robot(action[:HAND_DOF])
            self._run(self._set_positions_async(robot_positions))
        except concurrent.futures.TimeoutError:
            print(f"[HAND] Timed out sending Revo2 {self.name} joint positions")
        except Exception as exc:
            print(f"[HAND] Failed to send Revo2 {self.name} joint positions: {exc}")

    def move_home(self):
        """Move hand to home position (all zeros)."""
        if not self.connected:
            return
        self.act(np.zeros(HAND_DOF, dtype=np.float32))


class AirBotRevo2BimanualRobot:
    """Combined bimanual AIRBOT arms + Revo2 hands interface."""

    def __init__(
        self,
        left_arm_port: str,
        left_hand_port: str,
        right_arm_port: str,
        right_hand_port: str,
    ):
        self.left_arm = AirBotArm(port=left_arm_port, name="left_arm", home=LEFT_ARM_HOME)
        self.left_hand = Revo2Hand(port=left_hand_port, name="left_hand")
        self.right_arm = AirBotArm(port=right_arm_port, name="right_arm", home=RIGHT_ARM_HOME)
        self.right_hand = Revo2Hand(port=right_hand_port, name="right_hand")

    def connect(self):
        self.left_arm.connect()
        self.left_arm.move_home()
        self.left_hand.connect()
        self.left_hand.move_home()

        self.right_arm.connect()
        self.right_arm.move_home()
        self.right_hand.connect()
        self.right_hand.move_home()

    def disconnect(self):
        self.left_hand.disconnect()
        self.left_arm.disconnect()
        self.right_hand.disconnect()
        self.right_arm.disconnect()

    def get_state(self) -> np.ndarray:
        """Returns state: [left_arm(6), right_arm(6), left_hand(6), right_hand(6)]"""
        left_arm_state = self.left_arm.get_state()
        right_arm_state = self.right_arm.get_state()
        left_hand_state = self.left_hand.get_state()
        right_hand_state = self.right_hand.get_state()
        return np.concatenate([left_arm_state, right_arm_state, left_hand_state, right_hand_state]).astype(np.float32)

    def act(self, action: np.ndarray):
        """Executes action: [left_arm(6), right_arm(6), left_hand(6), right_hand(6)]"""
        left_arm_action = np.asarray(action[0:ARM_DOF], dtype=np.float32)
        right_arm_action = np.asarray(action[ARM_DOF:2 * ARM_DOF], dtype=np.float32)
        left_hand_action = np.asarray(action[2 * ARM_DOF:2 * ARM_DOF + HAND_DOF], dtype=np.float32)
        right_hand_action = np.asarray(action[2 * ARM_DOF + HAND_DOF:2 * ARM_DOF + 2 * HAND_DOF], dtype=np.float32)

        try:
            rr.log("robot/left_arm_cmd", rr.BarChart(left_arm_action))
            rr.log("robot/right_arm_cmd", rr.BarChart(right_arm_action))
            rr.log("robot/left_hand_cmd", rr.BarChart(left_hand_action))
            rr.log("robot/right_hand_cmd", rr.BarChart(right_hand_action))
        except Exception:
            pass

        self.left_arm.act(left_arm_action)
        self.right_arm.act(right_arm_action)
        self.left_hand.act(left_hand_action)
        self.right_hand.act(right_hand_action)


# Total DOF for bimanual: 2 arms (6 each) + 2 hands (6 each) = 24
BIMANUAL_STATE_DOF = 2 * ARM_DOF + 2 * HAND_DOF  # 24


@dataclass
class AirbotRevo2BimanualClientConfig:
    """Configuration for AIRBOT + Revo2 bimanual async inference client."""

    policy_type: str = "pi05"
    pretrained_name_or_path: str = "path/to/your/checkpoint"
    server_address: str = "127.0.0.1:8080"
    policy_device: str = "cuda"
    client_device: str = "cpu"
    chunk_size_threshold: float = 0.5
    fps: int = 30
    actions_per_chunk: int = 30
    left_arm_port: str = "50001"
    left_hand_port: str = "/dev/ttyUSB0"
    right_arm_port: str = "50002"
    right_hand_port: str = "/dev/ttyUSB1"
    top_cam_serial: str = "CP7JC42000EY"
    wrist_left_cam_serial: str = "CP7JC42000F4"
    wrist_right_cam_serial: str = "CP7JC42000F5"
    image_width: int = 640
    image_height: int = 480
    task: str = "pick up the ball and place it in the bowl"
    debug_visualize_queue_size: bool = False

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps


class AirbotRevo2BimanualAsyncClient:
    """AIRBOT + Revo2 bimanual client for the LeRobot async inference server."""

    prefix = "airbot_revo2_bimanual_client"
    logger = get_logger(prefix)

    def __init__(self, config: AirbotRevo2BimanualClientConfig):
        self.config = config
        self.cameras = TripleOrbbecStreamer(
            top_serial=config.top_cam_serial,
            wrist_left_serial=config.wrist_left_cam_serial,
            wrist_right_serial=config.wrist_right_cam_serial,
            image_size=(config.image_width, config.image_height),
        )
        self.robot = AirBotRevo2BimanualRobot(
            left_arm_port=config.left_arm_port,
            left_hand_port=config.left_hand_port,
            right_arm_port=config.right_arm_port,
            right_hand_port=config.right_hand_port,
        )
        self.lerobot_features = self._build_lerobot_features()

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

    def _build_lerobot_features(self) -> dict[str, dict]:
        return {
            "observation.images.top": {
                "dtype": "image",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_left": {
                "dtype": "image",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_right": {
                "dtype": "image",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (BIMANUAL_STATE_DOF,),
                "names": [f"state_{i}" for i in range(BIMANUAL_STATE_DOF)],
            },
        }

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def connect(self):
        self.logger.info("Connecting to cameras...")
        self.cameras.connect()
        time.sleep(2.0)

        self.logger.info("Connecting to robot...")
        self.robot.connect()
        self.logger.info("Hardware connected and ready")

    def start(self) -> bool:
        import grpc
        import pickle
        from lerobot.transport import services_pb2

        try:
            self.stub.Ready(services_pb2.Empty())
            policy_setup = services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            self.logger.info("Sending policy instructions to policy server")
            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as exc:
            self.logger.error(f"Failed to connect to policy server: {exc}")
            return False

    def stop(self):
        self.shutdown_event.set()
        self.cameras.stop()
        self.robot.disconnect()
        self.channel.close()
        self.logger.info("Client stopped")

    def get_observation(self) -> dict[str, Any]:
        frames = self.cameras.get_latest()
        robot_state = self.robot.get_state()

        try:
            rr.log("robot/state", rr.BarChart(robot_state))
        except Exception:
            pass

        obs = {
            "top": frames["top"],
            "wrist_left": frames["wrist_left"],
            "wrist_right": frames["wrist_right"],
            "task": self.config.task,
        }
        for i, value in enumerate(robot_state):
            obs[f"state_{i}"] = float(value)
        return obs

    def send_observation(self, obs: TimedObservation) -> bool:
        import grpc
        import pickle
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import send_bytes_in_chunks

        if not self.running:
            raise RuntimeError("Client not running")

        try:
            observation_iterator = send_bytes_in_chunks(
                pickle.dumps(obs),
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError as exc:
            self.logger.error(f"Error sending observation: {exc}")
            return False

    def receive_actions(self, verbose: bool = False):
        import grpc
        import pickle
        from lerobot.transport import services_pb2

        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions = pickle.loads(actions_chunk.data)
                if self.config.client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != self.config.client_device:
                            timed_action.action = timed_action.get_action().to(self.config.client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                with self.action_queue_lock:
                    for action in timed_actions:
                        with self.latest_action_lock:
                            if action.get_timestep() > self.latest_action:
                                self.action_queue.put(action)

                self.must_go.set()

                if verbose and timed_actions:
                    self.logger.info(
                        f"Received {len(timed_actions)} actions for step #{timed_actions[0].get_timestep()}"
                    )

                try:
                    rr.log("queue/size", rr.Scalar(self.action_queue.qsize()))
                except Exception:
                    pass

            except grpc.RpcError as exc:
                self.logger.error(f"Error receiving actions: {exc}")

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self.config.chunk_size_threshold

    def control_loop(self, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop starting")

        while self.running:
            control_loop_start = time.perf_counter()

            if self.actions_available():
                with self.action_queue_lock:
                    self.action_queue_size.append(self.action_queue.qsize())
                    timed_action = self.action_queue.get_nowait()

                action_tensor = timed_action.get_action().detach().cpu()
                if action_tensor.dtype == __import__("torch").bfloat16:
                    action_tensor = action_tensor.float()
                action_np = action_tensor.numpy()
                print(f"[CLIENT] Executing action #{timed_action.get_timestep()} shape={action_np.shape}")
                self.robot.act(action_np)

                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()

                if verbose:
                    self.logger.info(f"Executed action #{timed_action.get_timestep()}")

            if self._ready_to_send_observation():
                raw_obs = self.get_observation()
                if raw_obs["top"] is None or raw_obs["wrist_left"] is None or raw_obs["wrist_right"] is None:
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

            elapsed = time.perf_counter() - control_loop_start
            time.sleep(max(0, self.config.environment_dt - elapsed))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AIRBOT + Revo2 Bimanual Async Inference Client")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Policy server address")
    parser.add_argument("--policy-type", type=str, default="pi05", help="Policy type")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--left-arm-port", type=str, default="50001", help="Left AIRBOT arm port")
    parser.add_argument("--left-hand-port", type=str, default="/dev/ttyUSB0", help="Left Revo2 hand port")
    parser.add_argument("--right-arm-port", type=str, default="50000", help="Right AIRBOT arm port")
    parser.add_argument("--right-hand-port", type=str, default="/dev/ttyUSB1", help="Right Revo2 hand port")
    parser.add_argument("--top-cam", type=str, default="CP7JC42000EY", help="Top camera serial")
    parser.add_argument("--wrist-left-cam", type=str, default="CP7JC42000F4", help="Left wrist camera serial")
    parser.add_argument("--wrist-right-cam", type=str, default="CP7X54P0009B", help="Right wrist camera serial")
    parser.add_argument("--task", type=str, default="pick up the red block and place it in the bowl")
    parser.add_argument("--fps", type=int, default=30, help="Control frequency")
    parser.add_argument("--actions-per-chunk", type=int, default=30, help="Actions per chunk")
    parser.add_argument("--policy-device", type=str, default="cuda", help="Policy device")
    parser.add_argument("--debug-queue", action="store_true", help="Visualize action queue size")
    args = parser.parse_args()

    rr.init("airbot_revo2_bimanual_async_client", spawn=True)

    config = AirbotRevo2BimanualClientConfig(
        server_address=args.server,
        policy_type=args.policy_type,
        pretrained_name_or_path=args.pretrained,
        left_arm_port=args.left_arm_port,
        left_hand_port=args.left_hand_port,
        right_arm_port=args.right_arm_port,
        right_hand_port=args.right_hand_port,
        top_cam_serial=args.top_cam,
        wrist_left_cam_serial=args.wrist_left_cam,
        wrist_right_cam_serial=args.wrist_right_cam,
        task=args.task,
        fps=args.fps,
        actions_per_chunk=args.actions_per_chunk,
        policy_device=args.policy_device,
        debug_visualize_queue_size=args.debug_queue,
    )

    client = AirbotRevo2BimanualAsyncClient(config)

    print("\n" + "=" * 60)
    print(" AIRBOT + REVO2 Bimanual Async Inference Client")
    print("=" * 60)
    print(f" Server: {config.server_address}")
    print(f" Policy: {config.policy_type} ({config.pretrained_name_or_path})")
    print(f" Left arm port: {config.left_arm_port}")
    print(f" Left hand port: {config.left_hand_port}")
    print(f" Right arm port: {config.right_arm_port}")
    print(f" Right hand port: {config.right_hand_port}")
    print(f" State/action layout: left_arm(6) + right_arm(6) + left_hand(6) + right_hand(6) = 24")
    print(f" Cameras: top, wrist_left, wrist_right")
    print("=" * 60)

    client.connect()
    input("\nPress Enter to start inference...")

    if client.start():
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            kwargs={"verbose": True},
            daemon=True,
        )
        action_receiver_thread.start()

        try:
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
