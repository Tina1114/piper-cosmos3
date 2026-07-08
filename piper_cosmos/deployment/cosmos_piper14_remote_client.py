"""Remote RTC-compatible client for the Cosmos Piper14 policy server."""

from __future__ import annotations

from multiprocessing.connection import Client as ConnectionClient
from typing import Any, Mapping

import numpy as np


class CosmosPiper14RemotePolicyClient:
    """Small trusted-network RPC client for Piper14 policy inference."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, authkey: str | bytes = "cosmos-piper14"):
        self.address = (host, int(port))
        self.authkey = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
        self.conn = ConnectionClient(self.address, authkey=self.authkey)

    def close(self) -> None:
        self.conn.close()

    def update_observation(self, obs: Mapping[str, Any]) -> None:
        self._request({"op": "update_observation", "obs": dict(obs)})

    def get_action(self):
        return self._action_from_response(self._request({"op": "get_action"}))

    def infer(self, obs: Mapping[str, Any]):
        return self._action_from_response(self._request({"op": "infer", "obs": dict(obs)}))

    def metadata(self) -> dict[str, Any]:
        return self._request({"op": "metadata"})["metadata"]

    def reset(self) -> None:
        self._request({"op": "reset"})

    def shutdown_server(self) -> None:
        self._request({"op": "shutdown"})

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.conn.send(dict(payload))
        response = self.conn.recv()
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

    def __enter__(self) -> "CosmosPiper14RemotePolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
