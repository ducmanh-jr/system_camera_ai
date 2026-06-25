"""Production-ready trainer for VN-plate OCR (CRNN, from scratch).

Features:
    - torch.cuda.amp (mixed precision)
    - DDP support (single-node, multi-GPU)
    - EMA of model weights
    - OneCycleLR / CosineAnnealingWarmRestarts
    - Gradient clipping
    - Checkpointing (best CER, last, every-N epochs)
    - Resume training
    - Early stopping on val CER
    - TensorBoard logging
"""
from __future__ import annotations
import os
import math
import time
import json
import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import torch.distributed as dist

from ..data.vocab import Vocab
from ..data.transforms import Preprocessor, PreprocessConfig
from ..data.dataset import PlateDataset, SyntheticListDataset, MixedDataset
from ..data.augmentation import build_train_augment
from ..data.collate import make_collate_fn
from ..models.crnn import build_model
from ..losses.ctc import CTCWithEntropy
from ..inference.decode import greedy_decode
from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.ema import ModelEMA
from ..utils.logger import setup_logger, TBWriter

log = logging.getLogger(__name__)


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.out_dir = cfg["output"]["dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        self.logger = setup_logger("vn_alpr", log_file=os.path.join(self.out_dir, "train.log"))
        # determinism
        torch.manual_seed(cfg.get("seed", 42))

        # device / DDP
        self.ddp = cfg["train"].get("ddp", False)
        if self.ddp:
            assert dist.is_initialized(), "Launch with torchrun for DDP."
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = Vocab(cfg["vocab"]["charset"])
        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", False))
        self.ema = ModelEMA(self.model, decay=cfg["train"]["ema_decay"]) if cfg["train"].get("ema", False) else None
        self.tb = TBWriter(os.path.join(self.out_dir, "tb"),
                           enabled=cfg["output"].get("tensorboard", True) and _is_main_process())
        self.best_cer = float("inf")
        self.start_epoch = 0
        self.epochs_no_improve = 0

        if cfg["train"].get("resume"):
            self._resume(cfg["train"]["resume"])

    # ----- builders -----
    def _build_data(self):
        cfg = self.cfg
        pre_cfg = PreprocessConfig(
            img_height=cfg["data"]["img_height"],
            img_width=cfg["data"]["img_width"],
            grayscale=cfg["data"]["grayscale"],
            apply_clahe=cfg["data"]["apply_clahe"],
        )
        self.preprocessor = Preprocessor(pre_cfg)
        aug = build_train_augment()

        real_train = PlateDataset(
            root=cfg["data"]["root"], split=cfg["data"]["train_split"],
            vocab=self.vocab, preprocessor=self.preprocessor, augment=aug,
        )
        if cfg["data"].get("synthetic_dir"):
            synth = SyntheticListDataset(
                cfg["data"]["synthetic_dir"], vocab=self.vocab,
                preprocessor=self.preprocessor, augment=aug,
            )
            train_ds = MixedDataset(real_train, synth, cfg["data"].get("synthetic_ratio", 0.8))
            self.logger.info("Mixed train: real=%d, synth=%d", len(real_train), len(synth))
        else:
            train_ds = real_train

        val_ds = PlateDataset(
            root=cfg["data"]["root"], split=cfg["data"]["val_split"],
            vocab=self.vocab, preprocessor=self.preprocessor, augment=None,
        )

        collate = make_collate_fn(self.vocab)
        bs = cfg["train"]["batch_size"]
        nw = cfg["data"]["num_workers"]
        pm = cfg["data"]["pin_memory"]

        train_sampler = DistributedSampler(train_ds) if self.ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.ddp else None

        self.train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=nw, pin_memory=pm,
            collate_fn=collate, drop_last=True, persistent_workers=(nw > 0),
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, sampler=val_sampler,
            num_workers=nw, pin_memory=pm, collate_fn=collate,
            drop_last=False, persistent_workers=(nw > 0),
        )

    def _build_model(self):
        cfg = self.cfg
        in_ch = 1 if cfg["data"]["grayscale"] else 3
        model = build_model(cfg["model"], num_classes=self.vocab.num_classes, in_channels=in_ch)
        self.logger.info("Model params: %.2fM", model.num_parameters() / 1e6)
        model = model.to(self.device)
        if self.ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
        self.model = model

    def _model_module(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_loss(self):
        lc = self.cfg["loss"]["ctc"]
        self.criterion = CTCWithEntropy(
            blank=lc["blank_index"],
            zero_infinity=lc["zero_infinity"],
            entropy_weight=lc.get("entropy_weight", 0.0),
        ).to(self.device)

    def _build_optim(self):
        opt_cfg = self.cfg["optim"]
        params = self.model.parameters()
        if opt_cfg["optimizer"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"],
                                                weight_decay=opt_cfg["weight_decay"],
                                                betas=tuple(opt_cfg["betas"]))
        elif opt_cfg["optimizer"].lower() == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=opt_cfg["lr"], momentum=0.9,
                                              weight_decay=opt_cfg["weight_decay"], nesterov=True)
        else:
            raise ValueError(f"unknown optimizer: {opt_cfg['optimizer']}")

        sched = opt_cfg["scheduler"]
        epochs = self.cfg["train"]["epochs"]
        steps_per_epoch = max(1, len(self.train_loader))
        if sched == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=opt_cfg["lr"],
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=min(0.3, opt_cfg.get("warmup_epochs", 3) / max(1, epochs)),
                anneal_strategy="cos",
            )
            self.step_scheduler_per = "batch"
        elif sched == "cosine_warm_restart":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            self.step_scheduler_per = "epoch"
        elif sched == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
            self.step_scheduler_per = "epoch"
        else:
            raise ValueError(f"unknown scheduler: {sched}")

    # ----- training -----
    def train(self) -> None:
        epochs = self.cfg["train"]["epochs"]
        patience = self.cfg["train"]["early_stop_patience"]
        clip = self.cfg["optim"].get("grad_clip", 0.0)
        log_every = self.cfg["train"]["log_every"]
        ckpt_every = self.cfg["train"]["ckpt_every_epoch"]
        for epoch in range(self.start_epoch, epochs):
            self._train_one_epoch(epoch, clip, log_every)
            if self.step_scheduler_per == "epoch":
                self.scheduler.step()
            val_metrics = self.validate()
            if _is_main_process():
                self.logger.info(
                    "epoch %d: val seq_acc=%.4f cer=%.4f wer=%.4f",
                    epoch, val_metrics.seq_acc, val_metrics.cer, val_metrics.wer,
                )
                self.tb.add_scalar("val/seq_acc", val_metrics.seq_acc, epoch)
                self.tb.add_scalar("val/cer", val_metrics.cer, epoch)
                self.tb.add_scalar("val/wer", val_metrics.wer, epoch)
                improved = val_metrics.cer < self.best_cer - 1e-6
                if improved:
                    self.best_cer = val_metrics.cer
                    self.epochs_no_improve = 0
                    self._save("best.pt", epoch, val_metrics)
                else:
                    self.epochs_no_improve += 1
                self._save("last.pt", epoch, val_metrics)
                if (epoch + 1) % ckpt_every == 0:
                    self._save(f"epoch_{epoch + 1:03d}.pt", epoch, val_metrics)
                if self.epochs_no_improve >= patience:
                    self.logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                                     epoch, patience)
                    break
        if _is_main_process():
            self.tb.close()

    def _train_one_epoch(self, epoch: int, clip: float, log_every: int):
        self.model.train()
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        for i, (imgs, targets, target_lengths, texts) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.cfg["train"]["amp"]):
                logits = self.model(imgs)                          # (T, B, C)
                T = logits.size(0)
                B = logits.size(1)
                input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
                out = self.criterion(logits, targets, input_lengths, target_lengths)
                loss = out["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.step_scheduler_per == "batch":
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self._model_module())

            if _is_main_process() and (i % log_every == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info("epoch %d step %d/%d loss=%.4f lr=%.2e",
                                 epoch, i, len(self.train_loader), loss.item(), lr)
                global_step = epoch * len(self.train_loader) + i
                self.tb.add_scalar("train/loss", loss.item(), global_step)
                self.tb.add_scalar("train/lr", lr, global_step)
        if _is_main_process():
            self.logger.info("epoch %d wallclock=%.1fs", epoch, time.time() - t0)

    @torch.no_grad()
    def validate(self):
        model = self.ema.ema if self.ema is not None else self._model_module()
        model.eval()
        all_preds: list = []
        all_gts: list = []
        for imgs, targets, target_lengths, texts in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            preds = greedy_decode(log_probs, self.vocab)
            all_preds.extend(preds)
            all_gts.extend(texts)
        return compute_metrics(all_preds, all_gts)

    # ----- io -----
    def _save(self, fname: str, epoch: int, val_metrics) -> None:
        path = os.path.join(self.out_dir, fname)
        state = {
            "model_state_dict": self._model_module().state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "best_cer": self.best_cer,
            "val_metrics": val_metrics.__dict__,
            "config": self.cfg,
        }
        save_checkpoint(path, state)
        self.logger.info("saved %s", path)

    def _resume(self, path: str) -> None:
        self.logger.info("resuming from %s", path)
        ckpt = load_checkpoint(path, map_location=str(self.device))
        self._model_module().load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("ema_state_dict") and self.ema is not None:
            self.ema.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_cer = ckpt.get("best_cer", float("inf"))
