#!/usr/bin/env python3
"""
Compare SmallIris feature vectors across float, fused-float, and current ZK-style semantics.

This script is meant to be the first diagnostic step toward making the ZK circuit a
"high-fidelity approximation" of the original PyTorch model. It computes:

1. The original float GAP vector from `model.feature_extract_avg_pool(...)`.
2. A fused-float GAP vector using Conv+BN fused weights on the same normalized input.
3. A quantized integer GAP vector using the exported high-fidelity semantics:
   normalized input quantized by `activation_q`, fused weights quantized by
   `quantize_q`, biases quantized by `activation_q * quantize_q`, ReLU, MaxPool,
   divide-by-q after each pool, final GAP sum.
4. Two legacy raw-RGB integer paths to isolate the impact of the old pre-processing mismatch.

The output is JSON so it can be diffed or saved for later comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_open_set import get_model
from export_smalliris_zk_weights import fuse_conv_bn


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original float GAP, fused-float GAP, and current ZK-style GAP "
            "for the same SmallIris input image."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="./models/smalliris_e_40_lr_0_001_in_48_c1_24_c2_48_emb_64_best.pth",
        help="PyTorch checkpoint used for the original float model.",
    )
    parser.add_argument(
        "--weights-json",
        default="../fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64.json",
        help="Exported integer SmallIris JSON used by the Rust/ZK path.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a single RGB image.",
    )
    parser.add_argument(
        "--public-gap-json",
        default=None,
        help="Optional JSON produced by the recursive prover to compare against public_gap_sums.",
    )
    parser.add_argument(
        "--model-name",
        default="smalliris",
        help="Model name passed to get_model(...).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1500,
        help="Classifier output size used when loading the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for float/fused-float inference.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full JSON report.",
    )
    parser.add_argument(
        "--activation-q",
        type=int,
        default=4096,
        help=(
            "Scale used to quantize the normalized PyTorch input when running the "
            "experimental high-fidelity integer path."
        ),
    )
    return parser.parse_args()


def load_export(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resize_and_center_crop_exact(img: Image.Image, size: int) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("image must be non-empty")

    target_w = target_h = int(size)
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))

    resized = img.resize((resized_w, resized_h), resample=Image.BILINEAR)
    crop_x = (resized_w - target_w) // 2
    crop_y = (resized_h - target_h) // 2
    return resized.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))


def load_image_views(image_path: str, input_size: int) -> tuple[np.ndarray, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    img = resize_and_center_crop_exact(img, input_size)
    arr_hwc_u8 = np.asarray(img, dtype=np.uint8).copy()

    tensor = torch.from_numpy(arr_hwc_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    normalized = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return arr_hwc_u8, normalized


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def clamp_i32(value: int) -> int:
    return max(-(2**31), min(2**31 - 1, int(value)))


def relu_i32(x: int) -> int:
    return x if x > 0 else 0


def rescale_nonnegative_i32(x: int, divisor: int) -> int:
    if x <= 0:
        return 0
    return x // divisor


def idx3(c: int, i: int, j: int, h: int, w: int) -> int:
    return c * h * w + i * w + j


def wco(co: int, ci: int, ki: int, kj: int, cin: int, kh: int, kw: int) -> int:
    return co * (cin * kh * kw) + ci * (kh * kw) + ki * kw + kj


def conv2d_i32(
    inp: np.ndarray,
    cin: int,
    h: int,
    w: int,
    weights: np.ndarray,
    bias: np.ndarray,
    cout: int,
    k: int,
    pad: int,
) -> np.ndarray:
    out = np.zeros(cout * h * w, dtype=np.int64)
    for co in range(cout):
        for i in range(h):
            for j in range(w):
                acc = int(bias[co])
                for ci in range(cin):
                    for ki in range(k):
                        for kj in range(k):
                            ii = i + ki - pad
                            jj = j + kj - pad
                            if 0 <= ii < h and 0 <= jj < w:
                                wi = wco(co, ci, ki, kj, cin, k, k)
                                acc += int(inp[idx3(ci, ii, jj, h, w)]) * int(weights[wi])
                out[idx3(co, i, j, h, w)] = clamp_i32(acc)
    return out.astype(np.int32)


def maxpool2x2_i32(inp: np.ndarray, cin: int, h: int, w: int) -> np.ndarray:
    oh = h // 2
    ow = w // 2
    out = np.zeros(cin * oh * ow, dtype=np.int32)
    for c in range(cin):
        for i in range(oh):
            for j in range(ow):
                a = int(inp[idx3(c, 2 * i, 2 * j, h, w)])
                b = int(inp[idx3(c, 2 * i, 2 * j + 1, h, w)])
                c0 = int(inp[idx3(c, 2 * i + 1, 2 * j, h, w)])
                d = int(inp[idx3(c, 2 * i + 1, 2 * j + 1, h, w)])
                out[idx3(c, i, j, oh, ow)] = max(a, b, c0, d)
    return out


def zk_style_forward_int_gap_sum(exp: dict[str, Any], input_flat: np.ndarray) -> np.ndarray:
    h = int(exp["h"])
    w = int(exp["w"])
    c1 = int(exp["c1"])
    c2 = int(exp["c2"])
    c3 = int(exp["c3"])
    q = int(round(float(exp["quantize_q"])))

    w1 = np.asarray(exp["w1"], dtype=np.int32)
    b1 = np.asarray(exp["b1"], dtype=np.int32)
    w2 = np.asarray(exp["w2"], dtype=np.int32)
    b2 = np.asarray(exp["b2"], dtype=np.int32)
    w3 = np.asarray(exp["w3"], dtype=np.int32)
    b3 = np.asarray(exp["b3"], dtype=np.int32)

    pre1 = conv2d_i32(input_flat, 3, h, w, w1, b1, c1, 3, 1)
    a1 = np.maximum(pre1, 0).astype(np.int32)
    p1 = maxpool2x2_i32(a1, c1, h, w)
    p1 = np.array([rescale_nonnegative_i32(int(x), q) for x in p1], dtype=np.int32)

    h1 = h // 2
    w1_spatial = w // 2
    pre2 = conv2d_i32(p1, c1, h1, w1_spatial, w2, b2, c2, 3, 1)
    a2 = np.maximum(pre2, 0).astype(np.int32)
    p2 = maxpool2x2_i32(a2, c2, h1, w1_spatial)
    p2 = np.array([rescale_nonnegative_i32(int(x), q) for x in p2], dtype=np.int32)

    h2 = h // 4
    w2_spatial = w // 4
    pre3 = conv2d_i32(p2, c2, h2, w2_spatial, w3, b3, c3, 3, 1)
    a3 = np.maximum(pre3, 0).astype(np.int32)
    p3 = maxpool2x2_i32(a3, c3, h2, w2_spatial)
    p3 = np.array([rescale_nonnegative_i32(int(x), q) for x in p3], dtype=np.int32)

    hh = h // 8
    ww = w // 8
    sums = np.zeros(c3, dtype=np.int64)
    for c in range(c3):
        channel_sum = 0
        for i in range(hh):
            for j in range(ww):
                channel_sum += int(p3[idx3(c, i, j, hh, ww)])
        sums[c] = clamp_i32(channel_sum)
    return sums.astype(np.int32)


def original_float_gap(model: torch.nn.Module, normalized_input: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        vec = model.feature_extract_avg_pool(normalized_input.to(device)).cpu().numpy().reshape(-1)
    return vec.astype(np.float64)


def get_fused_params(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = model.features
    conv1, bn1 = seq[0], seq[1]
    conv2, bn2 = seq[4], seq[5]
    conv3, bn3 = seq[8], seq[9]

    w1f, b1f = fuse_conv_bn(conv1, bn1)
    w2f, b2f = fuse_conv_bn(conv2, bn2)
    w3f, b3f = fuse_conv_bn(conv3, bn3)
    return w1f, b1f, w2f, b2f, w3f, b3f


def fused_float_gap_from_fused_params(
    normalized_input: torch.Tensor,
    device: torch.device,
    fused_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> np.ndarray:
    w1f, b1f, w2f, b2f, w3f, b3f = fused_params
    x = normalized_input.to(device)
    with torch.no_grad():
        x = F.conv2d(x, w1f.to(device), b1f.to(device), stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.conv2d(x, w2f.to(device), b2f.to(device), stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.conv2d(x, w3f.to(device), b3f.to(device), stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        vec = torch.flatten(x, 1).cpu().numpy().reshape(-1)
    return vec.astype(np.float64)


def fused_float_gap(model: torch.nn.Module, normalized_input: torch.Tensor, device: torch.device) -> np.ndarray:
    return fused_float_gap_from_fused_params(
        normalized_input,
        device,
        get_fused_params(model),
    )


def quantize_float_tensor_to_i32(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    return torch.clamp(
        torch.round(tensor * float(scale)),
        min=-(2**31),
        max=2**31 - 1,
    ).to(torch.int32)


def experimental_aligned_int_gap_sum(
    model: torch.nn.Module,
    normalized_input: torch.Tensor,
    weight_q: int,
    activation_q: int,
) -> np.ndarray:
    fused_params = get_fused_params(model)
    w1f, b1f, w2f, b2f, w3f, b3f = fused_params

    input_int = quantize_float_tensor_to_i32(normalized_input.cpu(), activation_q)
    input_flat = input_int.numpy().reshape(-1)

    w1 = quantize_float_tensor_to_i32(w1f.cpu(), weight_q).numpy().reshape(-1)
    w2 = quantize_float_tensor_to_i32(w2f.cpu(), weight_q).numpy().reshape(-1)
    w3 = quantize_float_tensor_to_i32(w3f.cpu(), weight_q).numpy().reshape(-1)

    bias_scale = activation_q * weight_q
    b1 = quantize_float_tensor_to_i32(b1f.cpu(), bias_scale).numpy().reshape(-1)
    b2 = quantize_float_tensor_to_i32(b2f.cpu(), bias_scale).numpy().reshape(-1)
    b3 = quantize_float_tensor_to_i32(b3f.cpu(), bias_scale).numpy().reshape(-1)

    c1 = int(w1f.shape[0])
    c2 = int(w2f.shape[0])
    c3 = int(w3f.shape[0])
    h = int(normalized_input.shape[-2])
    w = int(normalized_input.shape[-1])

    pre1 = conv2d_i32(input_flat, 3, h, w, w1, b1, c1, 3, 1)
    a1 = np.maximum(pre1, 0).astype(np.int32)
    p1 = maxpool2x2_i32(a1, c1, h, w)
    p1 = np.array([rescale_nonnegative_i32(int(x), weight_q) for x in p1], dtype=np.int32)

    h1 = h // 2
    w1_spatial = w // 2
    pre2 = conv2d_i32(p1, c1, h1, w1_spatial, w2, b2, c2, 3, 1)
    a2 = np.maximum(pre2, 0).astype(np.int32)
    p2 = maxpool2x2_i32(a2, c2, h1, w1_spatial)
    p2 = np.array([rescale_nonnegative_i32(int(x), weight_q) for x in p2], dtype=np.int32)

    h2 = h // 4
    w2_spatial = w // 4
    pre3 = conv2d_i32(p2, c2, h2, w2_spatial, w3, b3, c3, 3, 1)
    a3 = np.maximum(pre3, 0).astype(np.int32)
    p3 = maxpool2x2_i32(a3, c3, h2, w2_spatial)
    p3 = np.array([rescale_nonnegative_i32(int(x), weight_q) for x in p3], dtype=np.int32)

    hh = h // 8
    ww = w // 8
    sums = np.zeros(c3, dtype=np.int64)
    for c in range(c3):
        channel_sum = 0
        for i in range(hh):
            for j in range(ww):
                channel_sum += int(p3[idx3(c, i, j, hh, ww)])
        sums[c] = clamp_i32(channel_sum)
    return sums.astype(np.int32)


def preview(vec: np.ndarray, limit: int = 8) -> list[float | int]:
    out: list[float | int] = []
    for x in np.asarray(vec).reshape(-1)[:limit]:
        if np.issubdtype(np.asarray([x]).dtype, np.integer):
            out.append(int(x))
        else:
            out.append(float(x))
    return out


def maybe_load_public_gap(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return np.asarray(payload["public_gap_sums"], dtype=np.int32)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    export = load_export(args.weights_json)

    model, checkpoint_input_size = get_model(
        args.model_name,
        args.checkpoint,
        num_classes=args.num_classes,
    )
    model.to(device)
    model.eval()

    export_input_size = int(export["h"])
    if checkpoint_input_size != export_input_size:
        raise ValueError(
            f"checkpoint input_size={checkpoint_input_size} but export h={export_input_size}"
        )

    arr_hwc_u8, normalized_input = load_image_views(args.image, export_input_size)
    gap_cells = (export_input_size // 8) ** 2
    weight_q = int(round(float(export["quantize_q"])))

    float_gap = original_float_gap(model, normalized_input, device)
    fused_params = get_fused_params(model)
    fused_gap = fused_float_gap_from_fused_params(normalized_input, device, fused_params)

    quantized_normalized_input = (
        quantize_float_tensor_to_i32(normalized_input.cpu(), args.activation_q).float()
        / float(args.activation_q)
    )
    fused_gap_quantized_input = fused_float_gap_from_fused_params(
        quantized_normalized_input,
        device,
        fused_params,
    )

    rust_layout_flat = arr_hwc_u8.reshape(-1).astype(np.int32)
    chw_layout_flat = np.transpose(arr_hwc_u8, (2, 0, 1)).reshape(-1).astype(np.int32)

    legacy_raw_rgb_sum_hwc = zk_style_forward_int_gap_sum(export, rust_layout_flat)
    legacy_raw_rgb_sum_chw = zk_style_forward_int_gap_sum(export, chw_layout_flat)
    legacy_raw_rgb_avg_hwc = legacy_raw_rgb_sum_hwc.astype(np.float64) / gap_cells
    legacy_raw_rgb_avg_chw = legacy_raw_rgb_sum_chw.astype(np.float64) / gap_cells

    export_activation_q = int(round(float(export.get("activation_q", export["quantize_q"]))))
    aligned_int_gap_sum = experimental_aligned_int_gap_sum(
        model,
        normalized_input,
        weight_q=weight_q,
        activation_q=export_activation_q,
    )
    aligned_int_gap_avg = aligned_int_gap_sum.astype(np.float64) / float(export_activation_q * gap_cells)

    public_gap = maybe_load_public_gap(args.public_gap_json)

    result: dict[str, Any] = {
        "image": str(pathlib.Path(args.image).resolve()),
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "weights_json": str(pathlib.Path(args.weights_json).resolve()),
        "input_size": int(export_input_size),
        "gap_cells": int(gap_cells),
        "variants": {
            "float_gap_eval_preproc": {
                "dimension": int(float_gap.shape[0]),
                "first8": preview(float_gap),
            },
            "fused_float_gap_eval_preproc": {
                "dimension": int(fused_gap.shape[0]),
                "first8": preview(fused_gap),
            },
            "fused_float_gap_eval_preproc_quantized_input_only": {
                "dimension": int(fused_gap_quantized_input.shape[0]),
                "first8": preview(fused_gap_quantized_input),
            },
            "export_aligned_int_gap_sum_eval_preproc": {
                "dimension": int(aligned_int_gap_sum.shape[0]),
                "first8": preview(aligned_int_gap_sum),
            },
            "export_aligned_int_gap_avg_eval_preproc": {
                "dimension": int(aligned_int_gap_avg.shape[0]),
                "first8": preview(aligned_int_gap_avg),
            },
            "legacy_raw_rgb_gap_sum_hwc_layout": {
                "dimension": int(legacy_raw_rgb_sum_hwc.shape[0]),
                "first8": preview(legacy_raw_rgb_sum_hwc),
            },
            "legacy_raw_rgb_gap_avg_hwc_layout": {
                "dimension": int(legacy_raw_rgb_avg_hwc.shape[0]),
                "first8": preview(legacy_raw_rgb_avg_hwc),
            },
            "legacy_raw_rgb_gap_sum_chw_layout": {
                "dimension": int(legacy_raw_rgb_sum_chw.shape[0]),
                "first8": preview(legacy_raw_rgb_sum_chw),
            },
            "legacy_raw_rgb_gap_avg_chw_layout": {
                "dimension": int(legacy_raw_rgb_avg_chw.shape[0]),
                "first8": preview(legacy_raw_rgb_avg_chw),
            },
        },
        "cosine_similarity": {
            "float_vs_fused_float_eval_preproc": cosine_similarity(float_gap, fused_gap),
            "float_vs_fused_float_eval_preproc_quantized_input_only": cosine_similarity(
                float_gap, fused_gap_quantized_input
            ),
            "float_vs_export_aligned_int_gap_avg_eval_preproc": cosine_similarity(
                float_gap, aligned_int_gap_avg
            ),
            "float_vs_legacy_raw_rgb_gap_avg_hwc_layout": cosine_similarity(float_gap, legacy_raw_rgb_avg_hwc),
            "float_vs_legacy_raw_rgb_gap_avg_chw_layout": cosine_similarity(float_gap, legacy_raw_rgb_avg_chw),
            "fused_float_vs_export_aligned_int_gap_avg_eval_preproc": cosine_similarity(
                fused_gap, aligned_int_gap_avg
            ),
            "fused_float_vs_legacy_raw_rgb_gap_avg_hwc_layout": cosine_similarity(
                fused_gap, legacy_raw_rgb_avg_hwc
            ),
            "fused_float_vs_legacy_raw_rgb_gap_avg_chw_layout": cosine_similarity(
                fused_gap, legacy_raw_rgb_avg_chw
            ),
            "fused_float_quantized_input_only_vs_export_aligned_int_gap_avg_eval_preproc": cosine_similarity(
                fused_gap_quantized_input, aligned_int_gap_avg
            ),
            "legacy_raw_rgb_gap_avg_hwc_vs_chw": cosine_similarity(
                legacy_raw_rgb_avg_hwc, legacy_raw_rgb_avg_chw
            ),
        },
        "experimental_scales": {
            "weight_q": int(weight_q),
            "activation_q": int(export_activation_q),
        },
    }

    if public_gap is not None:
        result["public_gap_json"] = str(pathlib.Path(args.public_gap_json).resolve())
        result["variants"]["public_gap_sums_from_json"] = {
            "dimension": int(public_gap.shape[0]),
            "first8": preview(public_gap),
        }
        result["cosine_similarity"]["public_gap_vs_export_aligned_int_gap_sum_eval_preproc"] = cosine_similarity(
            public_gap, aligned_int_gap_sum
        )
        result["cosine_similarity"]["public_gap_vs_legacy_raw_rgb_gap_sum_hwc_layout"] = cosine_similarity(
            public_gap, legacy_raw_rgb_sum_hwc
        )
        result["cosine_similarity"]["public_gap_vs_legacy_raw_rgb_gap_sum_chw_layout"] = cosine_similarity(
            public_gap, legacy_raw_rgb_sum_chw
        )
        result["exact_match_to_public_gap_json"] = {
            "export_aligned_int_gap_sum_eval_preproc": bool(np.array_equal(public_gap, aligned_int_gap_sum)),
            "legacy_raw_rgb_gap_sum_hwc_layout": bool(np.array_equal(public_gap, legacy_raw_rgb_sum_hwc)),
            "legacy_raw_rgb_gap_sum_chw_layout": bool(np.array_equal(public_gap, legacy_raw_rgb_sum_chw)),
        }

    text = json.dumps(result, indent=2)
    print(text)

    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
