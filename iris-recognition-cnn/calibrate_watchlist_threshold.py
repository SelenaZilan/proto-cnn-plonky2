#!/usr/bin/env python3
"""
Calibrate a threshold for 1:N watchlist identification.

Workflow:
1. Build a watchlist database from a random subset of identities in enrollment.
2. Score every probe in the probe split against the full watchlist database.
3. Use the maximum cosine similarity over all watchlist identities as the probe score.
4. Choose thresholds at target FPR values using non-watchlist probes only.

This script does not require a special "criminal" dataset. It simulates an airport
watchlist by designating a subset of ordinary identities as watchlist subjects.
"""

# python calibrate_watchlist_threshold.py \
#   --checkpoint "./models/smalliris_e_40_lr_0_001_in_48_c1_24_c2_48_emb_64_best.pth" \
#   --enrollment-dir "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment" \
#   --probe-dir "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test" \
#   --watchlist-ratio 0.2 \
#   --seed 42 \
#   --output-json "./results/watchlist_threshold_calibration_48_c1_24_c2_48_emb_64.json"


from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import Counter

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from eval_open_set import get_model


def make_dataset(data_path: str, input_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return datasets.ImageFolder(data_path, transform)


def extract_embeddings(feature_extract_func, dataloader, device):
    embeddings = []
    labels = []
    paths = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_embeddings = feature_extract_func(images).cpu().detach().numpy()
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.cpu().detach().numpy())

    base_dataset = dataloader.dataset
    paths = [sample_path for sample_path, _ in base_dataset.samples]
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    embeddings = normalize(embeddings, axis=1, norm="l2")
    return embeddings, labels, paths


def select_watchlist_identities(common_ids, ratio, count, seed):
    rng = random.Random(seed)
    if count is None:
        count = max(1, int(round(len(common_ids) * ratio)))
    if count <= 0 or count > len(common_ids):
        raise ValueError(
            f"watchlist_count must be in [1, {len(common_ids)}], got {count}"
        )
    return sorted(rng.sample(list(common_ids), count))


def build_watchlist_templates(
    enroll_embeddings, enroll_labels, watchlist_ids, template_strategy="max"
):
    watchlist_templates = {}
    for identity in watchlist_ids:
        feats = enroll_embeddings[enroll_labels == identity]
        if len(feats) == 0:
            raise ValueError(f"Identity {identity} has no enrollment templates")
        if template_strategy == "mean":
            feats = np.mean(feats, axis=0, keepdims=True)
        elif template_strategy != "max":
            raise ValueError(f"Unsupported template_strategy: {template_strategy}")
        watchlist_templates[identity] = feats
    return watchlist_templates


def score_probe_against_watchlist(probe_vec, watchlist_templates):
    scores = {}
    probe_col = probe_vec.reshape(-1, 1)
    for identity, templates in watchlist_templates.items():
        cos = np.matmul(templates, probe_col)
        scores[identity] = float(np.max(cos))
    pred_identity = max(scores, key=scores.get)
    max_score = scores[pred_identity]
    return max_score, pred_identity, scores


def collect_watchlist_scores(
    probe_embeddings,
    probe_labels,
    probe_paths,
    watchlist_templates,
    watchlist_ids,
):
    rows = []
    watchlist_id_set = set(watchlist_ids)

    for embedding, label, path in zip(probe_embeddings, probe_labels, probe_paths):
        max_score, pred_identity, _ = score_probe_against_watchlist(
            embedding, watchlist_templates
        )
        is_watchlist = int(label in watchlist_id_set)
        correct_identity = int(is_watchlist and pred_identity == label)
        rows.append(
            {
                "path": path,
                "label": int(label),
                "is_watchlist": is_watchlist,
                "pred_identity": int(pred_identity),
                "max_score": float(max_score),
                "correct_identity": correct_identity,
            }
        )

    return rows


def empirical_rate_ge(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores >= threshold)) if len(scores) else 0.0


def threshold_for_target_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    if len(negative_scores) == 0:
        raise ValueError("No negative probe scores available for calibration")
    try:
        return float(np.quantile(negative_scores, 1.0 - target_fpr, method="higher"))
    except TypeError:
        return float(
            np.quantile(negative_scores, 1.0 - target_fpr, interpolation="higher")
        )


def percentile_summary(scores: np.ndarray, percentiles):
    if len(scores) == 0:
        return {f"p{int(p):02d}": None for p in percentiles}
    return {
        f"p{int(p):02d}": float(np.percentile(scores, p)) for p in percentiles
    }


def evaluate_thresholds(positive_scores, negative_scores, positive_pred_correct, targets):
    rows = []
    for alpha in targets:
        threshold = threshold_for_target_fpr(negative_scores, alpha)
        tpr = empirical_rate_ge(positive_scores, threshold)
        fpr = empirical_rate_ge(negative_scores, threshold)
        hit_and_correct = (
            float(
                np.mean((positive_scores >= threshold) & positive_pred_correct.astype(bool))
            )
            if len(positive_scores)
            else 0.0
        )
        rows.append(
            {
                "target_fpr": float(alpha),
                "threshold_cosine_similarity": float(threshold),
                "empirical_fpr": float(fpr),
                "tpr_watchlist_detection": float(tpr),
                "frr_watchlist_detection": float(1.0 - tpr),
                "top1_correct_and_above_threshold": float(hit_and_correct),
            }
        )
    return rows


def parse_target_fprs(text: str):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("target FPR list must not be empty")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a 1:N watchlist threshold at fixed FPR."
    )
    parser.add_argument(
        "--checkpoint",
        default="./models/smalliris_e_40_lr_0_001_in_48_c1_24_c2_48_emb_64_best.pth",
    )
    parser.add_argument("--model-name", default="smalliris")
    parser.add_argument(
        "--enrollment-dir",
        default="./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment",
    )
    parser.add_argument(
        "--probe-dir",
        default="./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test",
    )
    parser.add_argument("--num-classes", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=196)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--watchlist-ratio",
        type=float,
        default=0.2,
        help="fraction of common identities used as watchlist if --watchlist-count is not set",
    )
    parser.add_argument(
        "--watchlist-count",
        type=int,
        default=None,
        help="override watchlist identity count directly",
    )
    parser.add_argument(
        "--target-fprs",
        default="0.1,0.05,0.01,0.001",
        help="comma-separated target FPR values",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-json",
        default="./results/watchlist_threshold_calibration.json",
    )
    parser.add_argument(
        "--template-strategy",
        choices=["max", "mean"],
        default="max",
        help=(
            "How to build per-identity watchlist templates from enrollment images. "
            "'max' keeps all templates and scores with per-identity max cosine; "
            "'mean' averages each identity's enrollment embeddings into one template."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, input_size = get_model(
        args.model_name, args.checkpoint, num_classes=args.num_classes
    )
    model.to(device)
    model.eval()

    enrollment_dataset = make_dataset(args.enrollment_dir, input_size)
    probe_dataset = make_dataset(args.probe_dir, input_size)
    if enrollment_dataset.classes != probe_dataset.classes:
        raise ValueError("Enrollment and probe datasets must share the same class ordering")

    enrollment_loader = DataLoader(
        enrollment_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print("Extracting enrollment embeddings...")
    enroll_embeddings, enroll_labels, _ = extract_embeddings(
        model.feature_extract_avg_pool, enrollment_loader, device
    )
    print("Extracting probe embeddings...")
    probe_embeddings, probe_labels, probe_paths = extract_embeddings(
        model.feature_extract_avg_pool, probe_loader, device
    )

    enrollment_ids = set(int(x) for x in np.unique(enroll_labels))
    probe_ids = set(int(x) for x in np.unique(probe_labels))
    common_ids = sorted(enrollment_ids & probe_ids)
    if len(common_ids) < 2:
        raise ValueError("Need at least two common identities between enrollment and probe sets")

    watchlist_ids = select_watchlist_identities(
        common_ids,
        args.watchlist_ratio,
        args.watchlist_count,
        args.seed,
    )
    watchlist_templates = build_watchlist_templates(
        enroll_embeddings, enroll_labels, watchlist_ids, args.template_strategy
    )

    print(
        f"Scoring probes against watchlist of {len(watchlist_ids)} identities "
        f"(out of {len(common_ids)} common identities)..."
    )
    rows = collect_watchlist_scores(
        probe_embeddings,
        probe_labels,
        probe_paths,
        watchlist_templates,
        watchlist_ids,
    )

    positive_rows = [row for row in rows if row["is_watchlist"] == 1]
    negative_rows = [row for row in rows if row["is_watchlist"] == 0]
    positive_scores = np.array([row["max_score"] for row in positive_rows], dtype=np.float64)
    negative_scores = np.array([row["max_score"] for row in negative_rows], dtype=np.float64)
    positive_pred_correct = np.array(
        [row["correct_identity"] for row in positive_rows], dtype=np.int32
    )

    target_fprs = parse_target_fprs(args.target_fprs)
    threshold_rows = evaluate_thresholds(
        positive_scores, negative_scores, positive_pred_correct, target_fprs
    )

    class_names = enrollment_dataset.classes
    watchlist_identity_names = [class_names[idx] for idx in watchlist_ids]
    summary = {
        "scenario": "1:N watchlist threshold calibration",
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "input_size": int(input_size),
        "watchlist_ratio": float(args.watchlist_ratio),
        "watchlist_count": int(len(watchlist_ids)),
        "template_strategy": args.template_strategy,
        "common_identity_count": int(len(common_ids)),
        "watchlist_identity_indices": [int(x) for x in watchlist_ids],
        "watchlist_identity_names": watchlist_identity_names,
        "num_enrollment_images": int(len(enrollment_dataset)),
        "num_probe_images": int(len(probe_dataset)),
        "num_positive_probes": int(len(positive_rows)),
        "num_negative_probes": int(len(negative_rows)),
        "positive_score_percentiles": percentile_summary(positive_scores, [1, 5, 50, 95, 99]),
        "negative_score_percentiles": percentile_summary(negative_scores, [50, 90, 95, 99]),
        "thresholds_for_target_fpr": threshold_rows,
        "decision_rule": (
            "For each probe, compute cosine similarity against every watchlist identity. "
            f"Templates are built with template_strategy='{args.template_strategy}'. "
            "Then take the maximum identity score and alarm if max_score >= threshold."
        ),
        "note": (
            "This script simulates a watchlist by selecting a subset of ordinary identities. "
            "Thresholds are calibrated from watchlist-negative probe max scores. "
            "Use a separate final evaluation split if you need a strict held-out report."
        ),
    }

    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
