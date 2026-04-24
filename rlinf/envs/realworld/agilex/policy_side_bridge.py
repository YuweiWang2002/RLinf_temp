from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


def _decode_compressed_image(image_bytes: bytes) -> np.ndarray:
    import cv2

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image from robot server.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class AgilexPolicySideBridge:
    """Policy-side Socket.IO bridge used to talk to the robot-side process."""

    _CONNECT_TIMEOUT_S = 120.0
    _CONNECT_RETRY_INTERVAL_S = 1.0
    _SINGLE_CONNECT_WAIT_TIMEOUT_S = 1.0

    def __init__(self, host: str, port: int, wait_for_connection: bool = True):
        import socketio

        self._host = host
        self._port = port
        self.server_url = f"http://{host}:{port}"
        self._connected_event = threading.Event()
        self._connection_generation = 0
        self._client = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
        )
        self._register_handlers()
        self._connect()
        if wait_for_connection:
            self.wait_for_connection()

    def _register_handlers(self) -> None:
        @self._client.event
        def connect():
            self._connection_generation += 1
            self._connected_event.set()

        @self._client.event
        def connect_error(_data):
            self._connected_event.clear()

        @self._client.event
        def disconnect():
            self._connected_event.clear()

    def _connect(self) -> None:
        start_time = time.monotonic()
        last_error: Exception | None = None

        while True:
            elapsed_seconds = int(time.monotonic() - start_time)
            LOGGER.info(
                "Connecting to %s:%s, waited %d seconds",
                self._host,
                self._port,
                elapsed_seconds,
            )
            try:
                self._client.connect(
                    self.server_url,
                    transports=["websocket"],
                    wait_timeout=self._SINGLE_CONNECT_WAIT_TIMEOUT_S,
                )
                if self._client.connected:
                    return
            except Exception as exc:  # socketio raises ConnectionError on failure
                last_error = exc

            elapsed = time.monotonic() - start_time
            if elapsed >= self._CONNECT_TIMEOUT_S:
                raise TimeoutError(
                    f"Failed to connect to robot bridge at {self._host}:{self._port} "
                    f"within {int(self._CONNECT_TIMEOUT_S)} seconds."
                ) from last_error

            time.sleep(self._CONNECT_RETRY_INTERVAL_S)

    def wait_for_connection(self, timeout: float | None = None) -> bool:
        if self.ready:
            return True
        return self._connected_event.wait(timeout=timeout)

    @property
    def ready(self) -> bool:
        return self._client.connected

    @property
    def connection_generation(self) -> int:
        return self._connection_generation

    def close(self) -> None:
        if self._client.connected:
            self._client.disconnect()

    def reset(
        self,
        *,
        manual_reset: bool | None = None,
        timeout: float | None = None,
    ) -> dict | None:
        if not self.ready:
            return None
        payload: dict[str, Any] = {}
        if manual_reset is not None:
            payload["manual_reset"] = bool(manual_reset)
        return self._client.call("reset_state", payload, timeout=timeout)

    def start_episode(
        self,
        *,
        timeout: float | None = None,
    ) -> dict | None:
        if not self.ready:
            return None
        return self._client.call("start_episode", {}, timeout=timeout)

    def startup_check(
        self,
        *,
        timeout: float | None = None,
    ) -> dict | None:
        if not self.ready:
            return None
        return self._client.call("startup_check", {}, timeout=timeout)

    def wait_startup_check(
        self,
        *,
        timeout: float | None = 120.0,
        poll_interval: float = 1.0,
        request_timeout: float = 10.0,
    ) -> dict:
        start_time = time.monotonic()
        while True:
            response = self.startup_check(timeout=request_timeout)
            if response is None:
                raise RuntimeError("Robot bridge startup check returned no response.")

            status = str(response.get("status", ""))
            if status == "success":
                return response

            reason = str(response.get("reason", "unknown"))
            if status != "blocked":
                raise RuntimeError(
                    "Robot bridge startup check failed: "
                    f"status={status!r}, response={response!r}."
                )

            elapsed = time.monotonic() - start_time
            LOGGER.info(
                "Robot bridge is not ready yet: reason=%s, waited=%.1fs",
                reason,
                elapsed,
            )
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    "Robot bridge startup check timed out "
                    f"after {int(timeout)} seconds (last reason={reason!r})."
                )
            time.sleep(poll_interval)

    def get_observation(self, prompt: str) -> collections.OrderedDict | None:
        if not self.ready:
            return None

        response = self._client.call(
            "get_observation",
            {"prompt": prompt},
            timeout=10,
        )
        if response is None:
            return None

        images = collections.OrderedDict(
            (
                camera_name,
                _decode_compressed_image(image_bytes),
            )
            for camera_name, image_bytes in response["images"].items()
        )
        observation = collections.OrderedDict()
        observation["state"] = np.asarray(response["state"], dtype=np.float32)
        observation["images"] = images
        observation["prompt"] = response["prompt"]
        observation["success"] = bool(response.get("success", False))
        return observation

    def publish_joint_commands(
        self, left_joints: list[float], right_joints: list[float]
    ) -> None:
        if not self.ready:
            raise RuntimeError("Robot bridge is not connected.")
        self._client.call(
            "publish_joint_commands",
            {"left": left_joints, "right": right_joints},
            timeout=10,
        )
