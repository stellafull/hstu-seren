"""SASRec-based retrieval module augmented with serendipity expert scoring."""

from generative_recommenders_pl.models.hstu_seren import HSTUSeren


class SASRecSeren(HSTUSeren):
    """Thin wrapper that reuses the HSTU+Seren plumbing with a SASRec backbone."""

    pass

