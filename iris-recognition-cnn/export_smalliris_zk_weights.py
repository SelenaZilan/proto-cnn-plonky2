#!/usr/bin/env python3
"""Fuse BatchNorm into Conv and export a high-fidelity integer SmallIris JSON."""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
import torch.nn as nn

from models import SmallIrisCNN


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused weight [cout,cin,kh,kw] and bias [cout] (float32)."""
    assert conv.bias is None
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    denom = torch.sqrt(var + eps)
    scale = (gamma / denom).view(-1, 1, 1, 1)
    fused_w = conv.weight * scale
    fused_b = beta - gamma * mu / denom
    return fused_w.detach(), fused_b.detach()


def flatten_w(w: torch.Tensor) -> list[int]:
    """[cout, cin, kh, kw] -> list row-major same as Rust wco()."""
    w = w.cpu().numpy().astype("int64")
    return [int(x) for x in w.flatten()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="./models/smalliris_e_40_lr_0_001_in_48_c1_24_c2_48_emb_64_best.pth",
        help="state_dict from train.py",
    )
    p.add_argument(
        "--out",
        default="../fixtures/smalliris_real_i32.json",
        help="JSON path (relative to iris-recognition-cnn ok)",
    )
    p.add_argument("--q", type=float, default=4096.0, help="float = int / q after quantize")
    p.add_argument(
        "--activation-q",
        type=float,
        default=None,
        help=(
            "scale for the normalized input activations; defaults to --q. "
            "Biases are exported at activation_q * q so the integer forward can "
            "preserve the normalized-input scale across layers."
        ),
    )
    p.add_argument(
        "--h",
        type=int,
        default=48,
        help="spatial H in JSON (must equal --w, multiple of 8, 8..128); smaller ⇒ smaller ZK circuit",
    )
    p.add_argument("--w", type=int, default=48, help="spatial W in JSON (must equal --h)")
    args = p.parse_args()

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt.resolve()}")

    h, w = args.h, args.w
    if h != w:
        raise SystemExit("h and w must be equal (square input)")
    if h < 8 or h % 8 != 0 or h > 128:
        raise SystemExit("h=w must be in [8,128] and divisible by 8")

    payload = torch.load(ckpt, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        model_config = payload.get("model_config", {})
    else:
        state_dict = payload
        model_config = {}

    m = SmallIrisCNN(
        num_classes=model_config.get("num_classes", 1500),
        embedding_dim=model_config.get("embedding_dim", 128),
        c1=model_config.get("c1", 32),
        c2=model_config.get("c2", 64),
    )
    m.load_state_dict(state_dict)
    m.eval()

    seq = m.features
    conv1, bn1 = seq[0], seq[1]
    conv2, bn2 = seq[4], seq[5]
    conv3, bn3 = seq[8], seq[9]

    w1f, b1f = fuse_conv_bn(conv1, bn1)
    w2f, b2f = fuse_conv_bn(conv2, bn2)
    w3f, b3f = fuse_conv_bn(conv3, bn3)

    q = args.q
    activation_q = args.activation_q if args.activation_q is not None else q
    bias_q = q * activation_q
    w1 = torch.clamp((w1f * q).round(), -(2**31), 2**31 - 1).to(torch.int32)
    b1 = torch.clamp((b1f * bias_q).round(), -(2**31), 2**31 - 1).to(torch.int32)
    w2 = torch.clamp((w2f * q).round(), -(2**31), 2**31 - 1).to(torch.int32)
    b2 = torch.clamp((b2f * bias_q).round(), -(2**31), 2**31 - 1).to(torch.int32)
    w3 = torch.clamp((w3f * q).round(), -(2**31), 2**31 - 1).to(torch.int32)
    b3 = torch.clamp((b3f * bias_q).round(), -(2**31), 2**31 - 1).to(torch.int32)

    gap_cells = (h // 8) ** 2
    doc = {
        "h": h,
        "w": w,
        "quantize_q": q,
        "activation_q": activation_q,
        "c1": int(conv1.out_channels),
        "c2": int(conv2.out_channels),
        "c3": int(conv3.out_channels),
        "note": (
            "Weights are quantized by quantize_q, normalized input activations are "
            "quantized by activation_q, and fused biases are quantized by "
            "activation_q * quantize_q. Final public ZK GAP uses per-channel sums; "
            f"divide by activation_q * {gap_cells} to compare against PyTorch global "
            "average pool outputs."
        ),
        "w1": flatten_w(w1),
        "b1": b1.cpu().numpy().astype("int64").tolist(),
        "w2": flatten_w(w2),
        "b2": b2.cpu().numpy().astype("int64").tolist(),
        "w3": flatten_w(w3),
        "b3": b3.cpu().numpy().astype("int64").tolist(),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    print("Wrote", out.resolve(), "sizes", {k: len(v) if isinstance(v, list) else v for k, v in doc.items() if k != "note"})


if __name__ == "__main__":
    main()
