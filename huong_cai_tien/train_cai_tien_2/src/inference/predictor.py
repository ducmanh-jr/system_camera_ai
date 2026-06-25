"""End-to-end inference predictor.

Usage:
    pred = Predictor.from_checkpoint('runs/exp1/best.pt')
    text = pred.predict_image('plate.jpg')
"""
from __future__ import annotations
import os
from typing import List, Optional, Union
import numpy as np
import cv2
import torch

from ..data.vocab import Vocab
from ..data.transforms import Preprocessor, PreprocessConfig
from ..models.crnn import build_model
from .decode import greedy_decode, beam_search_decode


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        vocab: Vocab,
        preprocessor: Preprocessor,
        device: str = "cpu",
        decoder: str = "greedy",
        beam_size: int = 10,
    ):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.device = device
        self.decoder = decoder
        self.beam_size = beam_size

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: Optional[str] = None,
                        prefer_ema: bool = True, decoder: str = "greedy") -> "Predictor":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["config"]
        vocab = Vocab(cfg["vocab"]["charset"])
        pre_cfg = PreprocessConfig(
            img_height=cfg["data"]["img_height"],
            img_width=cfg["data"]["img_width"],
            grayscale=cfg["data"]["grayscale"],
            apply_clahe=cfg["data"]["apply_clahe"],
        )
        pre = Preprocessor(pre_cfg)
        model = build_model(cfg["model"], num_classes=vocab.num_classes,
                            in_channels=1 if pre_cfg.grayscale else 3)
        state = ckpt.get("ema_state_dict") if prefer_ema else None
        if state is None:
            state = ckpt["model_state_dict"]
        model.load_state_dict(state, strict=True)
        return cls(model, vocab, pre, device=device, decoder=decoder)

    @torch.inference_mode()
    def predict_batch(self, images: List[np.ndarray]) -> List[str]:
        tensors = [self.preprocessor(im).unsqueeze(0) for im in images]
        x = torch.cat(tensors, dim=0).to(self.device, non_blocking=True)
        log_probs = self.model.predict_logp(x)
        if self.decoder == "beam":
            return beam_search_decode(log_probs, self.vocab, self.beam_size)
        return greedy_decode(log_probs, self.vocab)

    def predict_image(self, source: Union[str, np.ndarray]) -> str:
        if isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"cannot read {source}")
        else:
            img = source
        return self.predict_batch([img])[0]
