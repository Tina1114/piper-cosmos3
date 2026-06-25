"""Model definitions for Piper/Cosmos experiments."""

__all__ = ["SimpleMultiViewCNNPolicy", "baseline_action_loss"]


def __getattr__(name: str):
    if name in __all__:
        from .baseline_policy import SimpleMultiViewCNNPolicy, baseline_action_loss

        values = {
            "SimpleMultiViewCNNPolicy": SimpleMultiViewCNNPolicy,
            "baseline_action_loss": baseline_action_loss,
        }
        return values[name]
    raise AttributeError(name)
