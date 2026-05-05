
help:  ## Show help
	grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format the code
	pre-commit run -a

test: ## Run all tests
	coverage erase
	coverage run --source=src/ -m pytest tests --durations=10 -vv
	coverage report --format=markdown

train: ## Train the model
	python src/generative_recommenders_pl/scripts/train.py $(MAKEOVERRIDES)

eval: ## Evaluate the model
	python src/generative_recommenders_pl/scripts/eval.py $(MAKEOVERRIDES)

predict: ## Predict the model
	python src/generative_recommenders_pl/scripts/predict.py $(MAKEOVERRIDES)

download_data: ## Download raw datasets
	python src/generative_recommenders_pl/scripts/download.py

prepare_data: ## Prepare data
	python src/generative_recommenders_pl/scripts/prepare_data.py $(MAKEOVERRIDES)

# Semantic ID two-stage workflow
semantic_id_embed: ## Generate item embeddings (Hydra config in configs/semantic_id/embedding)
	python src/generative_recommenders_pl/scripts/semantic_id_embed.py $(MAKEOVERRIDES)

semantic_id_train: ## Train residual quantizer (Hydra config in configs/semantic_id/training)
	python src/generative_recommenders_pl/scripts/semantic_id_train.py $(MAKEOVERRIDES)

semantic_id_infer: ## Run semantic ID inference (Hydra config in configs/semantic_id/inference)
	python src/generative_recommenders_pl/scripts/semantic_id_infer.py $(MAKEOVERRIDES)

semantic_id_pipeline: ## Run full embedding→training→inference pipeline
	python src/generative_recommenders_pl/scripts/semantic_id_pipeline.py $(MAKEOVERRIDES)

v2-test: ## Run V2 protocol/unit gates
	PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_loo_manifest.py tests/test_ser_event_manifest.py tests/test_future_window_targets.py tests/test_no_ser_label_leakage.py tests/test_lf_rank_miner.py
	PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_trie_multipositive_loss.py tests/test_geometry_features.py tests/test_collision_resolver.py tests/test_aig_geo_inference.py
	PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_v2_wiring.py tests/test_future_window_dataset.py
v2-build-loo: ## Build V2 LOO manifest: make v2-build-loo DATASET=x INPUT=path.csv SID_LOOKUP=path.pt
	.venv/bin/python tools/build_loo_manifest.py --dataset $(DATASET) --input $(INPUT) --sid-lookup $(SID_LOOKUP)

v2-build-ser-event: ## Build ser-positive event manifest: make v2-build-ser-event DATASET=x INPUT=path.csv SID_LOOKUP=path.pt
	.venv/bin/python tools/build_ser_event_manifest.py --dataset $(DATASET) --input $(INPUT) --sid-lookup $(SID_LOOKUP)

v2-audit-loo: ## Audit V2 LOO manifest: make v2-audit-loo MANIFEST_DIR=tmp/loo_manifest/x
	.venv/bin/python tools/audit_loo_manifest.py $(MANIFEST_DIR)

v2-build-future: ## Build V2 future targets: make v2-build-future LOO_TRAIN=... OUTPUT=...
	.venv/bin/python tools/build_future_window_targets.py --loo-train $(LOO_TRAIN) --output $(OUTPUT)

v2-mine-lf: ## Mine V2 context candidates: make v2-mine-lf FUTURE_TARGETS=... OUTPUT=...
	.venv/bin/python tools/mine_label_free_ser_candidates.py --future-targets $(FUTURE_TARGETS) --output $(OUTPUT)

v2-audit-ser-reachability: ## Audit ser target reachability: make v2-audit-ser-reachability LOO_EVAL=...
	.venv/bin/python tools/audit_ser_target_reachability.py $(LOO_EVAL)
