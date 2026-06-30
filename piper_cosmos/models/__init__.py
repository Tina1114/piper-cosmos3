"""Model definitions for Piper/Cosmos experiments."""

__all__ = [
    "Cosmos3Piper14Adapter",
    "SimpleMultiViewCNNPolicy",
    "baseline_action_loss",
    "build_m5_adapter_from_config",
]


def __getattr__(name: str):
    if name in {"SimpleMultiViewCNNPolicy", "baseline_action_loss"}:
        from .baseline_policy import SimpleMultiViewCNNPolicy, baseline_action_loss

        values = {
            "SimpleMultiViewCNNPolicy": SimpleMultiViewCNNPolicy,
            "baseline_action_loss": baseline_action_loss,
        }
        return values[name]
    if name in {"Cosmos3Piper14Adapter", "build_m5_adapter_from_config"}:
        from .cosmos3_piper14_adapter import Cosmos3Piper14Adapter, build_m5_adapter_from_config

        values = {
            "Cosmos3Piper14Adapter": Cosmos3Piper14Adapter,
            "build_m5_adapter_from_config": build_m5_adapter_from_config,
        }
        return values[name]
    raise AttributeError(name)
