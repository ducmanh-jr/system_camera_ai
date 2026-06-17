import argparse
from pathlib import Path

import torch
from PIL import Image

from train_best import BestOCR, Codec as BestCodec, normalize_plate_prediction, pil_to_tensor, resize_pad
from train_scratch import Codec as ScratchCodec, ScratchCTCRecognizer


def load_best_model(ckpt, raw=False):
    cfg = ckpt["config"]
    codec = BestCodec(ckpt["charset"])
    model = BestOCR(
        codec.num_classes,
        len(codec.chars),
        cfg["model_dim"],
        cfg["layers"],
        cfg["heads"],
        cfg["dropout"],
    )
    state = ckpt["model"] if raw or "ema_model" not in ckpt else ckpt["ema_model"]
    model.load_state_dict(state)
    return model, codec, cfg["img_h"], cfg["img_w"], True


def load_scratch_model(ckpt, raw=False):
    cfg = ckpt["args"]
    codec = ScratchCodec(ckpt["charset"])
    model = ScratchCTCRecognizer(
        codec.num_classes,
        cfg["model_dim"],
        cfg["layers"],
        cfg["heads"],
        cfg["dropout"],
    )
    state = ckpt["model"] if raw or "ema_model" not in ckpt else ckpt["ema_model"]
    model.load_state_dict(state)
    return model, codec, cfg["img_h"], cfg["img_w"], False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--raw", action="store_true", help="Use raw model weights instead of EMA weights when available.")
    parser.add_argument("--no-rule", action="store_true", help="Print raw CTC text without plate-rule decoding.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "config" in ckpt:
        model, codec, img_h, img_w, is_best = load_best_model(ckpt, args.raw)
    else:
        model, codec, img_h, img_w, is_best = load_scratch_model(ckpt, args.raw)
    model.eval()

    img = Image.open(Path(args.image))
    x = pil_to_tensor(resize_pad(img, img_h, img_w)).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        pred = codec.decode(logits)[0]
    if is_best and not args.no_rule:
        print(normalize_plate_prediction(pred))
    else:
        print(pred)


if __name__ == "__main__":
    main()
