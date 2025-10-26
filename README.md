<div align="center">

# HSTU-Seren

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)


</div>

<br>

## Description

This repository aims to explore [Generative Recommenders](https://github.com/facebookresearch/generative-recommenders) in serendipity recommendations. build up on [Generative Recommenders PL](https://github.com/foreverYoungGitHub/generative-recommenders-pl)

- **Efficient Training & Inference**: enhances training and inference speed by optimizing GPU utilization. As a result, the training of the MovieLens-1M dataset over 100 epochs can now be completed in under 10 minutes on a single 4090 or L4 machine.
- **Experimentation Made Easy**: Easily manage and create hierarchical configurations with overrides via config files and command-line options to support various experiments.
- **Modular Configuration**: Dynamically instantiate objects through configuration files, allowing seamless switching between different datasets or modules without extensive rewriting.
- **Hardware Agnostic**: The dependency on NVIDIA GPUs has been removed, enabling you to run the scripts on any device, including local machines for training, evaluation, and debugging.
- **Improved Readability**: The code has been significantly refactored for clarity. The Generative Recommenders module is now divided into four major components: embeddings, preprocessor, sequence encoder, and postprocessor, making the training and evaluation processes more transparent.

## Env

A100 40G, CUDA 12.4, Ubuntu 22.04

## Installation

It is recommended to use `uv` to install the library:

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install --extra dev --extra test -r pyproject.toml
uv pip install -e . --no-deps
```

For Linux systems with GPU support, you can also install fbgemm-gpu to enhance performance:
```bash
uv pip install fbgemm-gpu
```

## How to Run

Prepare dataset based on configs/data, first build pre-train dataset, then build tuning dataset

Download data

```bash
make download_data 
```

Prepare data
example:

```bash
make prepare_data data=amazon_books
make prepare_data data=serenlens_books
```

Train and inference semantic id

```bash
make semantic_id_pipeline pipeline_cfg=sid_all
```

Train the Model with Default Configuration

```bash
# Train on CPU
make train trainer=cpu

# Train on GPU
make train trainer=gpu
```

Train the Model with a Specific Experiment Configuration. Choose an experiment configuration from [configs/experiment/](configs/experiment/):

```bash
make train experiment=finetune_ser2018
```

### Serendipity Two-Stage Pipeline

We provide dedicated configs for the serendipity-focused HSTU multi-head model:

1. **Stage A – relevance pre-training**
   - Amazon Books (2015): `make train experiment=pretrain_amazon_books`
   - Amazon Movies (2015): `make train experiment=pretrain_amazon_movies`
   - Serendipity-2018 training split: `make train experiment=pretrain_ser2018`
   - Checkpoints are written to `tmp/checkpoints/<experiment>/last.ckpt` for use in Stage B.
2. **Stage B – serendipity fine-tuning**
   - Serendipity-2018 answers: `make train experiment=finetune_ser2018`
   - SerenLens Books: `make train experiment=finetune_serenlens_books`
   - SerenLens Movies: `make train experiment=finetune_serenlens_movies`
   - To fine-tune from a custom checkpoint, override `model.init_from_ckpt` on the command line.

The fine-tuning stage logs HR@K / NDCG@K for relevance together with HR_ser@K / NDCG_ser@K computed on ser-positive targets.

Override Parameters from the Command Line

```bash
make train trainer.max_epochs=20 data.batch_size=64
```

Feel free to explore and modify the configurations to suit your needs. Your contributions and suggestions are always welcome!
