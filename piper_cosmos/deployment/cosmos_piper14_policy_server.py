"""Persistent RTC-style RPC server for Cosmos Piper14 policy inference."""

from __future__ import annotations

import traceback
from multiprocessing.connection import Listener
from typing import Any, Mapping

from piper_cosmos.deployment.cosmos_piper14_policy import CosmosPiper14PolicyClient, CosmosPiper14PolicyConfig


def serve_cosmos_piper14_policy(
    config: CosmosPiper14PolicyConfig | Mapping[str, Any],
    host: str = "127.0.0.1",
    port: int = 8766,
    authkey: str | bytes = "cosmos-piper14",
) -> None:
    key = authkey.encode("utf-8") if isinstance(authkey, str) else authkey
    policy = CosmosPiper14PolicyClient(config)
    listener = Listener((host, int(port)), authkey=key)
    print(f"[cosmos-piper14-policy-server] Listening on {host}:{port}", flush=True)
    try:
        while True:
            conn = listener.accept()
            print(f"[cosmos-piper14-policy-server] Client connected from {listener.last_accepted}", flush=True)
            should_shutdown = _serve_connection(policy, conn)
            conn.close()
            if should_shutdown:
                break
    finally:
        listener.close()
        print("[cosmos-piper14-policy-server] Stopped.", flush=True)


def _serve_connection(policy: CosmosPiper14PolicyClient, conn: Any) -> bool:
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return False

        if not isinstance(request, Mapping):
            conn.send({"ok": False, "error": f"Expected request mapping, got {type(request)}"})
            continue

        op = request.get("op")
        try:
            if op == "update_observation":
                policy.update_observation(request["obs"])
                conn.send({"ok": True})
            elif op == "get_action":
                conn.send({"ok": True, "action": policy.get_action().tolist()})
            elif op == "infer":
                conn.send({"ok": True, "action": policy.infer(request["obs"]).tolist()})
            elif op == "metadata":
                conn.send({"ok": True, "metadata": policy.metadata()})
            elif op == "reset":
                policy.reset()
                conn.send({"ok": True})
            elif op == "shutdown":
                conn.send({"ok": True})
                return True
            else:
                conn.send({"ok": False, "error": f"Unknown operation: {op!r}"})
        except Exception as exc:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
