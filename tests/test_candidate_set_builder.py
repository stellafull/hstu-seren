import torch

from generative_recommenders_pl.models.postprocessors.ser_postprocessors import (
    CandidateSetBuilder,
)


class _DummyIndex(torch.nn.Module):
    def __init__(self, ids: torch.Tensor):
        super().__init__()
        self.register_buffer("ids", ids.unsqueeze(0))

    @property
    def num_objects(self) -> int:
        return self.ids.size(1)

    def get_top_k_outputs(self, query_embeddings, k=None, invalid_ids=None):
        if k is None or k > self.ids.size(1):
            k = self.ids.size(1)
        top_ids = self.ids[:, :k].expand(query_embeddings.size(0), -1)
        scores = torch.zeros(
            query_embeddings.size(0),
            k,
            dtype=query_embeddings.dtype,
            device=query_embeddings.device,
        )
        return top_ids, scores


def test_candidate_set_builder_appends_target_when_missing():
    builder = CandidateSetBuilder(candidate_size=3, ensure_target=True, pad_id=0)
    index = _DummyIndex(torch.tensor([1, 2, 3, 4]))

    query = torch.randn(2, 4)
    target_ids = torch.tensor([5, 2])

    candidate_ids, candidate_mask = builder(
        query_embeddings=query,
        candidate_index=index,
        invalid_ids=None,
        target_ids=target_ids,
    )

    assert candidate_ids.shape == (2, 4)
    assert candidate_mask.shape == (2, 4)

    # Row 0: target missing in retrieved ids -> appended as last column.
    assert candidate_ids[0, -1].item() == 5
    assert candidate_mask[0, -1].item() == 1

    # Row 1: target already present -> extra column stays masked out.
    assert target_ids[1].item() in candidate_ids[1].tolist()
    assert candidate_mask[1, -1].item() == 0


def test_candidate_mask_keeps_target_even_if_pad_id_matches():
    builder = CandidateSetBuilder(candidate_size=2, ensure_target=True, pad_id=0)
    index = _DummyIndex(torch.tensor([0, 5, 6]))

    query = torch.randn(1, 3)
    target_ids = torch.tensor([0])

    candidate_ids, candidate_mask = builder(
        query_embeddings=query,
        candidate_index=index,
        invalid_ids=None,
        target_ids=target_ids,
    )

    assert candidate_ids.shape == (1, 3)
    assert candidate_mask.shape == (1, 3)
    # Target should be treated as valid despite matching pad_id.
    assert candidate_mask[0].any()
