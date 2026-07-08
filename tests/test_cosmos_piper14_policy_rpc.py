import socket
import time
import unittest
from multiprocessing import Process
from unittest.mock import Mock

import numpy as np

from piper_cosmos.deployment.cosmos_piper14_policy import CosmosPiper14PolicyConfig
from piper_cosmos.deployment.cosmos_piper14_policy_server import serve_cosmos_piper14_policy
from piper_cosmos.deployment.cosmos_piper14_remote_client import CosmosPiper14RemotePolicyClient


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _obs() -> dict:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return {
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "state": np.arange(14, dtype=np.float32),
        "prompt": "Assemble the mouse's battery.",
    }


class CosmosPiper14PolicyRpcTest(unittest.TestCase):
    def test_mock_server_client_infer_metadata_reset_shutdown(self) -> None:
        port = _free_port()
        config = CosmosPiper14PolicyConfig(
            mock_backend=True,
            checkpoint="mock://cosmos-piper14",
            action_horizon=4,
            camera_height=8,
            camera_width=10,
        )
        proc = Process(
            target=serve_cosmos_piper14_policy,
            kwargs={
                "config": config,
                "host": "127.0.0.1",
                "port": port,
                "authkey": "cosmos-piper14-test",
            },
        )
        proc.start()
        try:
            client = None
            deadline = time.time() + 5.0
            last_error: Exception | None = None
            while time.time() < deadline:
                try:
                    client = CosmosPiper14RemotePolicyClient(
                        host="127.0.0.1",
                        port=port,
                        authkey="cosmos-piper14-test",
                    )
                    break
                except ConnectionRefusedError as exc:
                    last_error = exc
                    time.sleep(0.05)
            if client is None:
                raise AssertionError(f"server did not start: {last_error}")

            with client:
                metadata = client.metadata()
                self.assertEqual(metadata["domain_name"], "piper14")
                self.assertEqual(metadata["raw_action_dim"], 14)
                self.assertEqual(metadata["action_horizon"], 4)

                action = client.infer(_obs())
                self.assertEqual(action.shape, (4, 14))
                self.assertEqual(action.dtype, np.float32)

                client.update_observation(_obs())
                action2 = client.get_action()
                self.assertEqual(action2.shape, (4, 14))

                client.reset()
                with self.assertRaisesRegex(RuntimeError, "observation is empty"):
                    client.get_action()
                client.shutdown_server()
        finally:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        self.assertEqual(proc.exitcode, 0)

    def test_remote_client_converts_wire_safe_action_lists_to_numpy(self) -> None:
        client = CosmosPiper14RemotePolicyClient.__new__(CosmosPiper14RemotePolicyClient)
        client.conn = Mock()
        client.conn.recv.return_value = {"ok": True, "action": [[1.0] * 14, [2.0] * 14]}

        action = client.get_action()

        self.assertEqual(action.shape, (2, 14))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.allclose(action[1], 2.0))


if __name__ == "__main__":
    unittest.main()
