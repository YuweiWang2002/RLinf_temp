from __future__ import annotations

import argparse
import threading
from collections import deque
from typing import Any

import rospy
from flask import Flask, request
from flask_socketio import SocketIO
from pynput.keyboard import Listener
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Header


app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=100 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

SUCCESS_TRIGGERED = False
SUCCESS_LOCK = threading.Lock()


class AgilexRobotSideBridge:
    """Robot-side ROS bridge process exposed over Socket.IO."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._state_lock = threading.Lock()
        self._operator_confirmed = False
        self._episode_started = False
        self._session_generation = 0
        self._confirmation_error: str | None = None
        self._startup_confirmation_thread: threading.Thread | None = None
        self.image_deques = {
            "cam_high": deque(maxlen=10),
            "cam_left_wrist": deque(maxlen=10),
            "cam_right_wrist": deque(maxlen=10),
        }
        self.joint_state_deques = {
            "left": deque(maxlen=200),
            "right": deque(maxlen=200),
        }
        rospy.init_node("rlinf_agilex_socket_server", anonymous=True)
        rospy.Subscriber(
            args.img_front_topic,
            CompressedImage,
            lambda msg: self.image_deques["cam_high"].append(msg),
        )
        rospy.Subscriber(
            args.img_left_topic,
            CompressedImage,
            lambda msg: self.image_deques["cam_left_wrist"].append(msg),
        )
        rospy.Subscriber(
            args.img_right_topic,
            CompressedImage,
            lambda msg: self.image_deques["cam_right_wrist"].append(msg),
        )
        rospy.Subscriber(
            args.puppet_arm_left_topic,
            JointState,
            lambda msg: self.joint_state_deques["left"].append(msg),
        )
        rospy.Subscriber(
            args.puppet_arm_right_topic,
            JointState,
            lambda msg: self.joint_state_deques["right"].append(msg),
        )
        self.left_publisher = rospy.Publisher(
            args.puppet_arm_left_cmd_topic,
            JointState,
            queue_size=10,
        )
        self.right_publisher = rospy.Publisher(
            args.puppet_arm_right_cmd_topic,
            JointState,
            queue_size=10,
        )
        self.intervention_publisher = rospy.Publisher(
            args.intervention_topic,
            Bool,
            queue_size=1,
        )
        self.alive_publisher = rospy.Publisher(
            args.cli_alive_topic,
            Bool,
            queue_size=1,
        )
        self.publish_rate = rospy.Rate(args.publish_rate)
        self.alive_timer = rospy.Timer(
            rospy.Duration(1.0 / float(args.cli_alive_rate)),
            self._publish_alive,
        )

    def _is_robot_ready(self) -> tuple[bool, str | None]:
        if not all(self.image_deques.values()):
            return False, "camera_frames_not_ready"
        if not all(self.joint_state_deques.values()):
            return False, "joint_states_not_ready"
        return True, None

    def _reset_session(self) -> None:
        with self._state_lock:
            self._session_generation += 1
            self._operator_confirmed = False
            self._episode_started = False
            self._confirmation_error = None
            self._startup_confirmation_thread = None

    def _get_runtime_state(self) -> tuple[str, str | None]:
        is_ready, reason = self._is_robot_ready()
        if not is_ready:
            return "BLOCKED", reason

        with self._state_lock:
            if self._confirmation_error is not None:
                return "BLOCKED", self._confirmation_error
            if not self._operator_confirmed:
                return "BLOCKED", "waiting_operator_confirm"
            if not self._episode_started:
                return "IDLE", None
        return "RUNNING", None

    def _build_state_response(self, status: str, **kwargs: Any) -> dict[str, Any]:
        state, reason = self._get_runtime_state()
        response: dict[str, Any] = {
            "status": status,
            "state": state,
        }
        if reason is not None:
            response["reason"] = reason
        response.update(kwargs)
        return response

    def _confirm_startup(self, session_generation: int) -> None:
        confirmation_error = None
        try:
            input("\n[ROBOT] Link is ready. Press Enter to allow policy to start.\n")
        except EOFError:
            confirmation_error = "confirm_input_unavailable"

        with self._state_lock:
            if session_generation != self._session_generation:
                return
            self._operator_confirmed = confirmation_error is None
            self._confirmation_error = confirmation_error
            self._startup_confirmation_thread = None

    def _ensure_startup_confirmation_requested(self) -> None:
        with self._state_lock:
            if self._operator_confirmed or self._confirmation_error is not None:
                return
            if (
                self._startup_confirmation_thread is not None
                and self._startup_confirmation_thread.is_alive()
            ):
                return

            session_generation = self._session_generation
            self._startup_confirmation_thread = threading.Thread(
                target=self._confirm_startup,
                args=(session_generation,),
                daemon=True,
            )
            self._startup_confirmation_thread.start()

    def handle_client_connected(self, sid: str) -> dict[str, Any]:
        self._reset_session()
        return self._build_state_response("connected", sid=sid)

    def handle_client_disconnected(self) -> None:
        self._reset_session()

    def get_observation(self, prompt: str) -> dict | None:
        is_ready, reason = self._is_robot_ready()
        if not is_ready and reason == "camera_frames_not_ready":
            rospy.logwarn_throttle(
                5.0, "Agilex bridge is waiting for compressed camera frames."
            )
            return None
        if not is_ready and reason == "joint_states_not_ready":
            rospy.logwarn_throttle(
                5.0, "Agilex bridge is waiting for front arm joint states."
            )
            return None

        sync_time = self.image_deques["cam_high"][-1].header.stamp

        def get_closest(queue, target_time):
            return min(
                queue,
                key=lambda msg: abs(msg.header.stamp.to_sec() - target_time.to_sec()),
            )

        images = {
            name: get_closest(queue, sync_time).data
            for name, queue in self.image_deques.items()
        }
        left_state = list(get_closest(self.joint_state_deques["left"], sync_time).position)
        right_state = list(
            get_closest(self.joint_state_deques["right"], sync_time).position
        )

        global SUCCESS_TRIGGERED
        with SUCCESS_LOCK:
            success = SUCCESS_TRIGGERED
            SUCCESS_TRIGGERED = False

        return {
            "images": images,
            "state": left_state + right_state,
            "prompt": prompt,
            "success": success,
        }

    def _publish_alive(self, _event=None) -> None:
        self.alive_publisher.publish(Bool(data=True))

    def _get_latest_joint_states(self) -> tuple[list[float] | None, list[float] | None]:
        left = None
        right = None
        if self.joint_state_deques["left"]:
            left = list(self.joint_state_deques["left"][-1].position)
        if self.joint_state_deques["right"]:
            right = list(self.joint_state_deques["right"][-1].position)
        return left, right

    def publish_joint_commands(
        self,
        left_joints: list[float],
        right_joints: list[float],
        *,
        allow_before_episode_start: bool = False,
        sleep: bool = True,
    ) -> None:
        state, reason = self._get_runtime_state()
        if state == "BLOCKED":
            raise RuntimeError(f"Robot bridge is blocked: reason={reason!r}.")
        if not allow_before_episode_start and state != "RUNNING":
            raise RuntimeError(
                "Episode is not started. Call start_episode before sending actions."
            )

        header = Header(stamp=rospy.Time.now())
        joint_names = [f"joint{i}" for i in range(7)]
        self.left_publisher.publish(
            JointState(header=header, name=joint_names, position=left_joints)
        )
        self.right_publisher.publish(
            JointState(header=header, name=joint_names, position=right_joints)
        )
        if sleep:
            self.publish_rate.sleep()

    def move_to_position_smoothly(
        self,
        target_left: list[float],
        target_right: list[float],
    ) -> None:
        current_left, current_right = self._get_latest_joint_states()
        start_left = target_left if current_left is None else current_left
        start_right = target_right if current_right is None else current_right

        reset_rate = rospy.Rate(self.args.reset_publish_rate)
        for i in range(self.args.reset_num_steps):
            alpha = float(i + 1) / float(self.args.reset_num_steps)
            interp_left = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_left, target_left)
            ]
            interp_right = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_right, target_right)
            ]
            self.publish_joint_commands(
                interp_left,
                interp_right,
                allow_before_episode_start=True,
                sleep=False,
            )
            reset_rate.sleep()

    def move_to_default_pose(self) -> None:
        left_pose = [float(x) for x in self.args.default_left_pose.split(",")]
        right_pose = [float(x) for x in self.args.default_right_pose.split(",")]
        self.move_to_position_smoothly(left_pose, right_pose)

    def toggle_intervention(self, enable: bool) -> None:
        self.intervention_publisher.publish(Bool(data=enable))

    def handle_reset(self, request_data: dict[str, Any] | None = None) -> dict[str, Any]:
        request_data = request_data or {}
        manual_reset = request_data.get("manual_reset")
        if manual_reset is None:
            manual_reset = self.args.require_manual_reset

        state, reason = self._get_runtime_state()
        if state == "BLOCKED":
            status = "error" if reason == "confirm_input_unavailable" else "blocked"
            return self._build_state_response(status, manual_reset=bool(manual_reset))

        with self._state_lock:
            self._episode_started = False
        self.toggle_intervention(False)
        self.move_to_default_pose()

        if manual_reset:
            try:
                input("\n[ROBOT] Reset the scene manually, then press Enter to continue.\n")
            except EOFError as exc:
                raise RuntimeError(
                    "Manual reset was requested, but stdin is not available for "
                    "confirmation."
                ) from exc

        return self._build_state_response("success", manual_reset=bool(manual_reset))

    def handle_start(self, _request_data: dict[str, Any] | None = None) -> dict[str, Any]:
        state, reason = self._get_runtime_state()
        if state == "BLOCKED":
            status = "error" if reason == "confirm_input_unavailable" else "blocked"
            return self._build_state_response(status, started=False)
        if state == "RUNNING":
            return self._build_state_response("success", started=True)

        with self._state_lock:
            self._episode_started = True
        return self._build_state_response("success", started=True)

    def handle_startup_check(
        self, _request_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        state, reason = self._get_runtime_state()
        if state == "BLOCKED" and reason == "waiting_operator_confirm":
            self._ensure_startup_confirmation_requested()
            state, reason = self._get_runtime_state()

        if state == "BLOCKED":
            status = "error" if reason == "confirm_input_unavailable" else "blocked"
            return self._build_state_response(status, checked=False)
        return self._build_state_response("success", checked=True)


BRIDGE: AgilexRobotSideBridge | None = None


def on_press(key):
    global SUCCESS_TRIGGERED
    try:
        if getattr(key, "char", None) == "s":
            with SUCCESS_LOCK:
                SUCCESS_TRIGGERED = True
    except AttributeError:
        return


@socketio.on("connect")
def handle_connect():
    assert BRIDGE is not None
    return BRIDGE.handle_client_connected(request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    assert BRIDGE is not None
    BRIDGE.handle_client_disconnected()


@socketio.on("reset_state")
def handle_reset_state(data=None):
    assert BRIDGE is not None
    return BRIDGE.handle_reset(data)


@socketio.on("get_observation")
def handle_get_observation(data):
    assert BRIDGE is not None
    prompt = data.get("prompt", "")
    return BRIDGE.get_observation(prompt)


@socketio.on("start_episode")
def handle_start_episode(data=None):
    assert BRIDGE is not None
    return BRIDGE.handle_start(data)


@socketio.on("startup_check")
def handle_startup_check(data=None):
    assert BRIDGE is not None
    return BRIDGE.handle_startup_check(data)


@socketio.on("publish_joint_commands")
def handle_publish_joint_commands(data):
    assert BRIDGE is not None
    BRIDGE.publish_joint_commands(data["left"], data["right"])
    return {"status": "ok"}


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLinf Agilex robot-side bridge")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--publish-rate", type=int, default=20)
    parser.add_argument("--reset-publish-rate", type=int, default=100)
    parser.add_argument("--reset-num-steps", type=int, default=150)
    parser.add_argument("--cli-alive-rate", type=float, default=10.0)
    parser.add_argument("--require-manual-reset", action="store_true", default=False)
    parser.add_argument("--img-front-topic", type=str, default="/camera_f/color/compressed")
    parser.add_argument("--img-left-topic", type=str, default="/camera_l/color/compressed")
    parser.add_argument("--img-right-topic", type=str, default="/camera_r/color/compressed")
    parser.add_argument("--puppet-arm-left-cmd-topic", type=str, default="/cmd/joint_left")
    parser.add_argument("--puppet-arm-right-cmd-topic", type=str, default="/cmd/joint_right")
    parser.add_argument("--puppet-arm-left-topic", type=str, default="/front/left/joint_states")
    parser.add_argument("--puppet-arm-right-topic", type=str, default="/front/right/joint_states")
    parser.add_argument("--intervention-topic", type=str, default="/intervention/takeover")
    parser.add_argument("--cli-alive-topic", type=str, default="/cli/alive")
    parser.add_argument("--default-left-pose", type=str, default="0,0,0,0,0,0,0")
    parser.add_argument("--default-right-pose", type=str, default="0,0,0,0,0,0,0")
    return parser


def main():
    global BRIDGE
    args = setup_argparser().parse_args()
    BRIDGE = AgilexRobotSideBridge(args)
    listener = Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    socketio.run(app, host=args.host, port=args.port, use_reloader=False)


if __name__ == "__main__":
    main()
