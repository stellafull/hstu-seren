"""SASRec + Seren expert retrieval module."""

from __future__ import annotations

from generative_recommenders_pl.models.hstu_seren import HSTUSeren

__all__ = ["SASRecSeren"]


class SASRecSeren(HSTUSeren):
    """Serendipity-aware retrieval model that uses the SASRec sequential encoder.

    The implementation inherits from :class:`HSTUSeren` to keep the hybrid training
    workflow while exposing a new Hydra target that pairs the serendipity expert
    with the SASRec backbone without modifying the existing HSTU-specific module.
    """

    pass
