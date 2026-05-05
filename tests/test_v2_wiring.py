import subprocess
import sys
from pathlib import Path

import pytest

V2_EXPERIMENTS = [
    "serenfree_v2_r_pretrain_movielens",
    "serenfree_v2_future_ai_movielens",
    "serenfree_v2_lf_rank_movielens",
    "serenfree_v2_eval_loo_movielens",
    "serenfree_v2_eval_ser_event_movielens",
    "serenfree_v2_r_pretrain_books",
    "serenfree_v2_future_ai_books",
    "serenfree_v2_lf_rank_books",
    "serenfree_v2_eval_loo_books",
    "serenfree_v2_eval_ser_event_books",
    "serenfree_v2_r_pretrain_movies",
    "serenfree_v2_future_ai_movies",
    "serenfree_v2_lf_rank_movies",
    "serenfree_v2_eval_loo_movies",
    "serenfree_v2_eval_ser_event_movies",
]

@pytest.mark.parametrize("experiment", V2_EXPERIMENTS)
def test_v2_hydra_configs_compose(experiment):
    subprocess.check_call([
        sys.executable,
        "src/generative_recommenders_pl/scripts/train.py",
        f"experiment={experiment}",
        "--cfg", "job",
    ], stdout=subprocess.DEVNULL)


def test_v2_model_wiring_consumes_future_losses_and_guards_labels():
    model_src = Path("src/generative_recommenders_pl/models/hstu_serenfree.py").read_text()
    train_src = Path("src/generative_recommenders_pl/scripts/train.py").read_text()
    assert "_batch_future_sid_targets" in model_src
    assert "trie_marginal_nll" in model_src
    assert "lf_rank_loss" in model_src
    assert "legacy history-window proxy is disabled for V2" in model_src
    assert "enforce_label_free_config" in train_src


def test_no_fixed_aig_level12_regression():
    haystack = "\n".join(
        p.read_text()
        for p in [*Path("configs/experiment").glob("serenfree_stage[34]_*.yaml"), *Path("configs/experiment").glob("serenfree_v2_*.yaml")]
    )
    evaluator = Path("src/generative_recommenders_pl/scripts/evaluate_serenfree_retrieval.py").read_text()
    assert "aig_levels: [1, 2]" not in haystack
    assert 'configured("aig_levels", [1, 2])' not in evaluator


def test_v2_future_training_configs_use_future_window_dataset():
    future_cfgs = list(Path("configs/experiment").glob("serenfree_v2_future_ai_*.yaml"))
    rank_cfgs = list(Path("configs/experiment").glob("serenfree_v2_lf_rank_*.yaml"))
    assert future_cfgs and rank_cfgs
    for cfg in future_cfgs:
        text = cfg.read_text()
        assert "DynamicFutureWindowTargetDataset" in text
        assert "loo_manifest/" in text
        assert "future_targets/" not in text
        assert "emit_rank_from_future_targets: false" in text
        assert "acceptable_min_gap: 4" in text
        assert "val_dataset:" not in text
        assert "monitor: val/ndcg@100" in text
        assert "save_top_k: 1" in text
        assert "patience: 10" in text
    runner = Path("tools/run_serenfree_v2_joint.sh").read_text()
    assert "build_future_window_targets.py" not in runner
    assert "dynamic sparse future targets" in runner
    for cfg in rank_cfgs:
        text = cfg.read_text()
        assert "emit_rank_from_future_targets: true" in text
        assert "rank_geometry_history_len: ${geometry.history_len}" in text
        assert "val_dataset:" not in text
        assert 'features: ["prefix_surprise"]' in text
        assert "qwen_distance" not in text


def test_v2_r_pretrain_configs_route_to_large_logs_or_named_movielens():
    books = Path("configs/experiment/serenfree_v2_r_pretrain_books.yaml").read_text()
    movies = Path("configs/experiment/serenfree_v2_r_pretrain_movies.yaml").read_text()
    movielens = Path("configs/experiment/serenfree_v2_r_pretrain_movielens.yaml").read_text()
    data_ser2018 = Path("configs/data/serendipity_2018_training.yaml").read_text()

    assert "- serenfree_stage1_amazon_books" in books
    assert "serenlens_books" not in books
    assert "- serenfree_stage1_amazon_movies" in movies
    assert "serenlens_movies" not in movies
    assert "override /data: serendipity_2018_training" in movielens
    assert "serenfree_stage1_serendipity_2018" not in movielens
    assert "dataset_name: serendipity-2018" in data_ser2018
    assert "semantic_id_prefix: serendipity_2018" in data_ser2018
    assert "serendipity-2018" in data_ser2018


def test_v2_joint_training_monitors_retrieval_metrics():
    model_cfg = Path("configs/model/hstu_serenfree_stage1.yaml").read_text()
    train_model = Path("src/generative_recommenders_pl/models/hstu_serenfree.py").read_text()

    assert "monitor: val/ndcg@100" in model_cfg
    assert "validation_ks: [10, 20, 50, 100, 200]" in model_cfg
    assert "val/ndcg@100" in "\n".join(
        Path("configs/experiment/serenfree_v2_future_ai_books.yaml").read_text().splitlines()
    )
    assert "_retrieval_metrics" in train_model
    assert "batched_constrained_beam_search" in train_model


def test_v2_eval_configs_enable_late_fusion_and_alpha_mapping():
    eval_cfgs = list(Path("configs/experiment").glob("serenfree_v2_eval_loo_*.yaml"))
    assert eval_cfgs
    for cfg in eval_cfgs:
        text = cfg.read_text()
        assert "late_fusion: true" in text
        assert "aig: false" in text
        assert "aig_alpha: ${inference.alpha}" in text
        assert "candidate_M: ${inference.candidate_M}" in text
        assert "geometry_levels: ${inference.geometry_levels}" in text
        assert "manifest_dir:" in text
        assert "LOOManifestEvalDataset" in text
        assert "ratings_file: ${serenfree_eval.manifest_dir}/loo_eval.parquet" in text
        assert "geometry_history_len: ${inference.geometry_history_len}" in text
    evaluator = Path("src/generative_recommenders_pl/scripts/evaluate_serenfree_retrieval.py").read_text()
    assert "LateFusionConfig.from_args" in evaluator
    assert 'configured("aig_alpha", configured("alpha", 0.0))' in evaluator
    assert "loo_manifest_hash" in evaluator


def test_v2_ser_event_eval_configs_use_ser_event_manifests():
    eval_cfgs = list(Path("configs/experiment").glob("serenfree_v2_eval_ser_event_*.yaml"))
    assert eval_cfgs
    for cfg in eval_cfgs:
        text = cfg.read_text()
        assert "ser_event_manifest/" in text
        assert "test_eval.parquet" in text
        assert 'buckets: ["ser_targets_only"]' in text
    evaluator = Path("src/generative_recommenders_pl/scripts/evaluate_serenfree_retrieval.py").read_text()
    assert "eval_protocol" in evaluator
    assert "eval_denominator" in evaluator
