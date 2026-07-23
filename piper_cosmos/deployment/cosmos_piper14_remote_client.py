"""Remote RTC-compatible client for the Cosmos Piper14 policy server."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Client as ConnectionClient
from typing import Any, Mapping

import numpy as np


class CosmosPiper14RemotePolicyClient:
    """Small trusted-network RPC client for Piper14 policy inference."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        authkey: str | bytes = "cosmos-piper14",
        timing: bool | None = None,
    ):
        self.address = (host, int(port))
        self.authkey = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
        self.timing = _env_enabled("COSMOS_PIPER14_CLIENT_TIMING") if timing is None else bool(timing)
        self.conn = ConnectionClient(self.address, authkey=self.authkey)
        self.last_inference_metadata: dict[str, Any] = {}

    def close(self) -> None:
        self.conn.close()

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self._request({"op": "update_observation", "obs": dict(obs)})

    def get_action(self):
        response = self._request({"op": "get_action"})
        self._record_inference_metadata(response)
        return self._action_from_response(response)

    def infer(self, obs: Mapping[str, Any]):
        response = self._request({"op": "infer", "obs": dict(obs)})
        self._record_inference_metadata(response)
        return self._action_from_response(response)

    def metadata(self) -> dict[str, Any]:
        return self._request({"op": "metadata"})["metadata"]

    def reset(self) -> None:
        self._request({"op": "reset"})
        self.last_inference_metadata = {}

    def shutdown_server(self) -> None:
        self._request({"op": "shutdown"})

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        send_started = time.perf_counter()
        self.conn.send(dict(payload))
        send_ms = (time.perf_counter() - send_started) * 1000.0
        recv_started = time.perf_counter()
        response = self.conn.recv()
        recv_ms = (time.perf_counter() - recv_started) * 1000.0
        if getattr(self, "timing", False):
            total_ms = (time.perf_counter() - started) * 1000.0
            print(
                f"[cosmos-piper14-client-timing] op={payload.get('op')} "
                f"send={send_ms:.3f}ms recv_wait={recv_ms:.3f}ms total={total_ms:.3f}ms",
                flush=True,
            )
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid Cosmos Piper14 policy server response: {response!r}")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Cosmos Piper14 policy server request failed."))
        return response

    @staticmethod
    def _action_from_response(response: Mapping[str, Any]) -> np.ndarray:
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2:
            raise RuntimeError(f"Invalid action shape from Cosmos Piper14 policy server: {action.shape}")
        return np.ascontiguousarray(action)

    def _record_inference_metadata(self, response: Mapping[str, Any]) -> None:
        metadata = response.get("inference_metadata", {})
        self.last_inference_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

    def __enter__(self) -> "CosmosPiper14RemotePolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
