
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
