from typing import TYPE_CHECKING, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    import lightning as L
    import torch
    from lightning.pytorch.loggers import Logger


_RUNTIME = None
log = None


from generative_recommenders_pl.utils.omegaconf_resolvers import register_safe_resolvers


register_safe_resolvers()


def _runtime_imports():
    global _RUNTIME, log
    if _RUNTIME is None:
        import lightning as L
        import torch
        import torch.multiprocessing
        from lightning.pytorch.loggers import Logger

        from generative_recommenders_pl.utils.instantiators import (
            get_metric_value,
            instantiate_callbacks,
            instantiate_loggers,
        )
        from generative_recommenders_pl.utils.logger import RankedLogger

        torch.backends.cuda.matmul.allow_tf32 = True if torch.cuda.is_available() else False
        torch.backends.cudnn.allow_tf32 = True
        torch.multiprocessing.set_sharing_strategy("file_system")

        log = RankedLogger(__name__)
        _RUNTIME = {
            "L": L,
            "torch": torch,
            "Logger": Logger,
            "get_metric_value": get_metric_value,
            "instantiate_callbacks": instantiate_callbacks,
            "instantiate_loggers": instantiate_loggers,
        }
    return _RUNTIME


def enforce_label_free_config(cfg: DictConfig) -> None:
    label_free = cfg.get("label_free")
    main_method = cfg.get("main_method")
    if main_method:
        forbidden = {
            "use_serendipity_labels_for_loss": "loss",
            "use_serendipity_labels_for_early_stopping": "early stopping",
            "use_serendipity_labels_for_hpo": "hyperparameter selection",
        }
        for key, name in forbidden.items():
            if bool(main_method.get(key, False)):
                raise ValueError(
                    f"SerenFree main_method must not use serendipity labels for {name}"
                )
    if not label_free:
        return
    if bool(label_free.get("forbid_ser_labels_in_train", False)):
        pseudo_path = cfg.get("model", {}).get("pseudo_ser_path") if cfg.get("model") else None
        if pseudo_path not in {None, "", "null"}:
            raise ValueError("label_free forbids pseudo_ser_path / ser-label-derived training inputs")
    if bool(label_free.get("forbid_ser_labels_in_early_stop", False)):
        early_stopping = cfg.get("callbacks", {}).get("early_stopping") if cfg.get("callbacks") else None
        monitor = early_stopping.get("monitor") if early_stopping else None
        if monitor and "ser" in str(monitor).lower():
            raise ValueError("label_free forbids ser-label early stopping monitors")
    if bool(label_free.get("forbid_ser_labels_in_hparam_selection", False)):
        metric = cfg.get("optimized_metric")
        if metric and "ser" in str(metric).lower():
            raise ValueError("label_free forbids ser-label hyperparameter selection metrics")


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _runtime_imports()
    L = runtime["L"]
    torch = runtime["torch"]
    instantiate_callbacks = runtime["instantiate_callbacks"]
    instantiate_loggers = runtime["instantiate_loggers"]

    enforce_label_free_config(cfg)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)

    max_item_id = getattr(datamodule, "max_item_id", None)
    if max_item_id is not None and cfg.get("model") and cfg.model.get("embeddings"):
        with open_dict(cfg.model.embeddings):
            cfg.model.embeddings.num_items = int(max_item_id)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule, _recursive_=False)
    init_from_checkpoint = cfg.get("init_from_checkpoint")
    if init_from_checkpoint:
        log.info(f"Initializing model weights from <{init_from_checkpoint}>")
        checkpoint = torch.load(
            init_from_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        current_state = model.state_dict()
        skipped_shape_mismatch = []
        state_dict = {}
        for name, value in checkpoint["state_dict"].items():
            if name == "pseudo_ser_items":
                continue
            current_value = current_state.get(name)
            if current_value is not None and current_value.shape != value.shape:
                skipped_shape_mismatch.append(
                    (name, tuple(value.shape), tuple(current_value.shape))
                )
                continue
            state_dict[name] = value
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            log.info(
                "Non-strict checkpoint init completed with missing=%s unexpected=%s",
                missing,
                unexpected,
            )
        if skipped_shape_mismatch:
            log.info(
                "Skipped checkpoint tensors with shape mismatch: %s",
                skipped_shape_mismatch,
            )

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../../../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    metric_dict, _ = train(cfg)
    get_metric_value = _runtime_imports()["get_metric_value"]
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
