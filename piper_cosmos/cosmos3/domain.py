"""Piper14 action-domain registration for Cosmos3."""

from __future__ import annotations


PIPER14_DOMAIN_NAME = "piper14"
PIPER14_DOMAIN_ID = 21
PIPER14_RAW_ACTION_DIM = 14


def register_piper14_domain() -> None:
    """Register Piper14 in Cosmos3's action-domain lookup tables."""

    from cosmos_framework.data.vfm.action import domain_utils

    domain_utils.EMBODIMENT_TO_DOMAIN_ID[PIPER14_DOMAIN_NAME] = PIPER14_DOMAIN_ID
    domain_utils.EMBODIMENT_TO_RAW_ACTION_DIM[PIPER14_DOMAIN_NAME] = PIPER14_RAW_ACTION_DIM

