#!/usr/bin/env python3
"""
Evaluate final 1:N airport-style watchlist matching with a fixed threshold.

This script consumes the output of `calibrate_watchlist_threshold.py`, reuses the
same simulated watchlist identities, applies the chosen threshold, and reports
final alarm-rule metrics on a probe split.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate_watchlist_threshold import (
    build_watchlist_templates,
    collect_watchlist_scores,
    extract_embeddings,
    make_dataset,
)
from eval_open_set import get_model


def load_calibration(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_threshold(payload, mode: str, target_fpr: float, threshold_override):
    if threshold_override is not None:
        return float(threshold_override)

    if mode != "target_fpr":
        raise ValueError(f"Unsupported threshold mode: {mode}")

    for row in payload["thresholds_for_target_fpr"]:
        if abs(float(row["target_fpr"]) - target_fpr) < 1e-12:
            return float(row["threshold_cosine_similarity"])

    raise ValueError(f"target_fpr={target_fpr} not found in calibration file")


def evaluate_alarm_rule(rows, threshold: float):
    tp = tn = fp = fn = 0
    positive_total = 0
    positive_top1_correct = 0
    positive_alarm_total = 0
    positive_alarm_correct = 0
    alarm_total = 0
    alarm_correct_identity_total = 0

    for row in rows:
        is_watchlist = bool(row["is_watchlist"])
        pred_correct = bool(row["correct_identity"])
        alarm = float(row["max_score"]) >= threshold

        row["alarm"] = bool(alarm)
        row["threshold"] = float(threshold)

        if is_watchlist:
            positive_total += 1
            positive_top1_correct += int(pred_correct)
            if alarm:
                tp += 1
                positive_alarm_total += 1
                positive_alarm_correct += int(pred_correct)
            else:
                fn += 1
        else:
            if alarm:
                fp += 1
            else:
                tn += 1

        if alarm:
            alarm_total += 1
            alarm_correct_identity_total += int(is_watchlist and pred_correct)

    total = len(rows)
    detection_tpr = tp / positive_total if positive_total else 0.0
    detection_frr = fn / positive_total if positive_total else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    top1_on_watchlist = positive_top1_correct / positive_total if positive_total else 0.0
    top1_on_alarm_watchlist = (
        positive_alarm_correct / positive_alarm_total if positive_alarm_total else 0.0
    )
    correct_alarm_fraction = (
        alarm_correct_identity_total / alarm_total if alarm_total else 0.0
    )

    return {
        "threshold": float(threshold),
        "num_probes": int(total),
        "num_watchlist_probes": int(positive_total),
        "num_non_watchlist_probes": int(fp + tn),
        "num_alarms": int(alarm_total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision_alarm_is_watchlist": float(precision),
        "false_alarm_rate": float(false_alarm_rate),
        "tpr_watchlist_detection": float(detection_tpr),
        "frr_watchlist_detection": float(detection_frr),
        "top1_identity_accuracy_on_watchlist_probes": float(top1_on_watchlist),
        "top1_identity_accuracy_on_alarm_watchlist_probes": float(top1_on_alarm_watchlist),
        "fraction_of_alarms_with_correct_identity": float(correct_alarm_fraction),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate airport-style watchlist alarm rule with a fixed threshold."
    )
    parser.add_argument(
        "--calibration-json",
        default="./results/watchlist_threshold_calibration_48_c1_24_c2_48_emb_64.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path. Defaults to checkpoint stored in calibration JSON.",
    )
    parser.add_argument(
        "--enrollment-dir",
        default="./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment",
    )
    parser.add_argument(
        "--probe-dir",
        default="./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-classes", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=196)
    parser.add_argument("--threshold-mode", choices=["target_fpr"], default="target_fpr")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-json",
        default="./results/watchlist_match_eval_48_c1_24_c2_48_emb_64.json",
    )
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_json)
    checkpoint = args.checkpoint or calibration["checkpoint"]
    model_name = args.model_name or calibration["model"]
    template_strategy = calibration.get("template_strategy", "max")
    threshold = select_threshold(
        calibration, args.threshold_mode, args.target_fpr, args.threshold
    )
    watchlist_ids = [int(x) for x in calibration["watchlist_identity_indices"]]

    device = torch.device(args.device)
    model, input_size = get_model(model_name, checkpoint, num_classes=args.num_classes)
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

    watchlist_templates = build_watchlist_templates(
        enroll_embeddings, enroll_labels, watchlist_ids, template_strategy
    )
    print("Scoring probes and applying final alarm rule...")
    rows = collect_watchlist_scores(
        probe_embeddings,
        probe_labels,
        probe_paths,
        watchlist_templates,
        watchlist_ids,
    )
    metrics = evaluate_alarm_rule(rows, threshold)

    class_names = enrollment_dataset.classes
    alarms = [row for row in rows if row["alarm"]]
    alarm_preview = [
        {
            "path": row["path"],
            "true_identity": class_names[row["label"]],
            "pred_identity": class_names[row["pred_identity"]],
            "is_watchlist": bool(row["is_watchlist"]),
            "score": float(row["max_score"]),
            "correct_identity": bool(row["correct_identity"]),
        }
        for row in alarms[:25]
    ]

    output = {
        "scenario": "1:N watchlist alarm-rule evaluation",
        "calibration_json": args.calibration_json,
        "model": model_name,
        "checkpoint": checkpoint,
        "input_size": int(input_size),
        "threshold_mode": args.threshold_mode,
        "target_fpr": float(args.target_fpr),
        "threshold": float(threshold),
        "template_strategy": template_strategy,
        "watchlist_count": int(len(watchlist_ids)),
        "watchlist_identity_names": [class_names[idx] for idx in watchlist_ids],
        "decision_rule": (
            f"For each probe, build templates with template_strategy='{template_strategy}', "
            "compute max cosine similarity over watchlist identities, "
            "Alarm iff max_score >= threshold; predicted identity is the argmax identity."
        ),
        "metrics": metrics,
        "alarm_preview_first_25": alarm_preview,
        "note": (
            "If probe-dir is the same split used for threshold calibration, this is an exploratory "
            "evaluation rather than a strict held-out final test."
        ),
    }

    pathlib.Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Threshold: {threshold:.6f}")
    print(f"Alarms: {metrics['num_alarms']} / {metrics['num_probes']}")
    print(
        f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
    )
    print(f"False alarm rate: {metrics['false_alarm_rate']:.4f}")
    print(f"Watchlist detection TPR: {metrics['tpr_watchlist_detection']:.4f}")
    print(
        "Top-1 identity accuracy on alarmed watchlist probes: "
        f"{metrics['top1_identity_accuracy_on_alarm_watchlist_probes']:.4f}"
    )
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
