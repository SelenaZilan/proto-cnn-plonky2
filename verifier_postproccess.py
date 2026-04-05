#!/usr/bin/env python3
"""Verify a saved recursive proof manifest and postprocess its public GAP sums.

Example with the default max-template watchlist configuration used in this repository:

    python ./verifier_postproccess.py \
      --manifest-json ./fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64_S5750L01_proof_manifest.json \
      --templates-json ./fixtures/watchlist_templates_750_754_lr.json \
      --threshold 0.9529750943183899 \
      --output-json ./fixtures/watchlist_result_750_754.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a saved SmallIris recursive proof, recover the GAP-average "
            "vector, and optionally compute cosine similarity / threshold decisions."
        )
    )
    parser.add_argument(
        "--manifest-json",
        required=True,
        help="Proof manifest JSON written by zk-smalliris-real-recursive.",
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help="Override weights JSON path. Defaults to the path stored in the manifest.",
    )
    parser.add_argument(
        "--templates-json",
        default='fixtures/watchlist_templates_750_754_lr.json',
        help=(
            "Optional template JSON. Supported formats: "
            "a single vector list, {'vector': [...]}, "
            "{'id1': [...], 'id2': [[...], [...]]}, or "
            "[{'identity': 'id1', 'vector': [...]}, ...]. "
            "For the current default max-template watchlist demo, use "
            "./fixtures/watchlist_templates_750_754_lr.json."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default = 0.9529750943183899,
        help="Optional cosine threshold used for the final alarm decision.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top cosine scores to include in the output preview.",
    )
    parser.add_argument(
        "--output-json",
        default='fixtures/watchlist_result_750_754.json',
        help="Optional path to save the full verifier result JSON.",
    )
    return parser.parse_args()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def resolve_path(path_str: str, manifest_path: pathlib.Path) -> pathlib.Path:
    raw = pathlib.Path(path_str)
    if raw.is_absolute():
        return raw

    candidates = [
        repo_root() / raw,
        manifest_path.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root() / raw).resolve()


def load_json(path: pathlib.Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("cannot L2-normalize a zero vector")
    return vec / norm


def run_rust_verifier(
    proof_path: pathlib.Path, verifier_data_path: pathlib.Path, gap_dimension: int | None = None
) -> dict[str, Any]:
    binary = repo_root() / "target" / "release" / "zk-smalliris-real-recursive-verify"
    if binary.exists():
        cmd = [
            str(binary),
            "--proof-path",
            str(proof_path),
            "--verifier-data-path",
            str(verifier_data_path),
        ]
    else:
        cmd = [
            "cargo",
            "+nightly",
            "run",
            "--release",
            "--quiet",
            "--bin",
            "zk-smalliris-real-recursive-verify",
            "--",
            "--proof-path",
            str(proof_path),
            "--verifier-data-path",
            str(verifier_data_path),
        ]
    if gap_dimension is not None:
        cmd.extend(["--gap-dimension", str(int(gap_dimension))])

    completed = subprocess.run(
        cmd,
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def recover_gap_average(public_gap_sums: np.ndarray, activation_q: float, h: int, w: int) -> np.ndarray:
    gap_cells = (h // 8) * (w // 8)
    return public_gap_sums.astype(np.float64) / (float(activation_q) * gap_cells)


def load_templates(path: pathlib.Path) -> dict[str, np.ndarray]:
    payload = load_json(path)

    def to_matrix(obj: Any) -> np.ndarray:
        arr = np.asarray(obj, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("template vectors must be 1-D or 2-D numeric arrays")
        return arr

    templates: dict[str, np.ndarray] = {}

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            for row in payload:
                identity = str(row.get("identity", "template"))
                vecs = row.get("vectors", row.get("vector"))
                if vecs is None:
                    raise ValueError("template row must contain 'vector' or 'vectors'")
                templates[identity] = to_matrix(vecs)
        else:
            templates["template"] = to_matrix(payload)
        return templates

    if isinstance(payload, dict):
        if "vector" in payload or "vectors" in payload:
            identity = str(payload.get("identity", "template"))
            templates[identity] = to_matrix(payload.get("vectors", payload.get("vector")))
            return templates

        for identity, vecs in payload.items():
            templates[str(identity)] = to_matrix(vecs)
        return templates

    raise ValueError("unsupported template JSON format")


def score_templates(query_vec: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, Any]:
    query_norm = l2_normalize(query_vec)
    scores = []

    for identity, matrix in templates.items():
        matrix_norm = np.vstack([l2_normalize(row) for row in matrix])
        per_template = matrix_norm @ query_norm
        scores.append(
            {
                "identity": identity,
                "max_cosine_similarity": float(np.max(per_template)),
                "num_templates": int(matrix_norm.shape[0]),
                "all_template_scores": [float(x) for x in per_template.tolist()],
            }
        )

    scores.sort(key=lambda row: row["max_cosine_similarity"], reverse=True)
    return {
        "top_identity": scores[0]["identity"] if scores else None,
        "top_score": scores[0]["max_cosine_similarity"] if scores else None,
        "scores": scores,
    }


def build_terminal_summary(output: dict[str, Any], output_json: pathlib.Path | None) -> str:
    lines = [
        "VERIFIER POSTPROCESS SUMMARY",
        f"proof_verified: {output['proof_verified']}",
        f"manifest_matches_verified_proof: {output['manifest_public_gap_sums_match_verified_proof']}",
        f"dimension: {output['dimension']}",
        f"input_image: {output['input_image']}",
    ]

    preview = output.get("preview", {})
    first8 = preview.get("public_gap_sums_first8")
    if first8 is not None:
        lines.append(f"public_gap_sums_first8: {first8}")

    template_match = output.get("template_match")
    if template_match:
        top_score = template_match.get("top_score")
        score_text = "None" if top_score is None else f"{top_score:.4f}"
        lines.append(f"top_identity: {template_match.get('top_identity')}")
        lines.append(f"top_score: {score_text}")

    decision = output.get("decision")
    if decision:
        threshold = decision.get("threshold")
        threshold_text = "None" if threshold is None else f"{threshold:.4f}"
        lines.append(f"threshold: {threshold_text}")
        lines.append(f"alarm: {decision.get('alarm')}")
        lines.append(f"pred_identity: {decision.get('pred_identity')}")

    if output_json is not None:
        lines.append(f"full_output_json: {output_json}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = pathlib.Path(args.manifest_json).resolve()
    manifest = load_json(manifest_path)

    proof_path = resolve_path(manifest["proof_path"], manifest_path)
    verifier_data_path = resolve_path(manifest["verifier_data_path"], manifest_path)
    weights_path = resolve_path(args.weights_json or manifest["weights"], manifest_path)
    weights = load_json(weights_path)

    verification = run_rust_verifier(proof_path, verifier_data_path, manifest.get("dimension"))
    verified_public_gap_sums = np.asarray(verification["public_gap_sums"], dtype=np.int64)

    manifest_public_gap_sums = np.asarray(manifest["public_gap_sums"], dtype=np.int64)
    manifest_match = np.array_equal(verified_public_gap_sums, manifest_public_gap_sums)

    activation_q = float(manifest.get("activation_q", weights.get("activation_q", weights["quantize_q"])))
    h = int(manifest.get("h", weights["h"]))
    w = int(manifest.get("w", weights["w"]))
    gap_cells = (h // 8) * (w // 8)

    gap_average = recover_gap_average(verified_public_gap_sums, activation_q, h, w)
    gap_average_l2 = l2_normalize(gap_average)

    output: dict[str, Any] = {
        "proof_verified": bool(verification["proof_verified"]),
        "manifest_public_gap_sums_match_verified_proof": bool(manifest_match),
        "manifest_json": str(manifest_path),
        "proof_path": str(proof_path),
        "verifier_data_path": str(verifier_data_path),
        "weights_json": str(weights_path),
        "input_image": manifest.get("input_image"),
        "dimension": int(len(verified_public_gap_sums)),
        "activation_q": activation_q,
        "h": h,
        "w": w,
        "gap_cells": gap_cells,
        "public_gap_sums": [int(x) for x in verified_public_gap_sums.tolist()],
        "public_gap_average": [float(x) for x in gap_average.tolist()],
        "public_gap_average_l2_normalized": [float(x) for x in gap_average_l2.tolist()],
        "preview": {
            "public_gap_sums_first8": [int(x) for x in verified_public_gap_sums[:8].tolist()],
            "public_gap_average_first8": [float(x) for x in gap_average[:8].tolist()],
            "public_gap_average_l2_normalized_first8": [float(x) for x in gap_average_l2[:8].tolist()],
        },
    }

    if args.templates_json:
        templates_path = pathlib.Path(args.templates_json).resolve()
        template_scores = score_templates(gap_average, load_templates(templates_path))
        output["templates_json"] = str(templates_path)
        output["template_match"] = {
            "top_identity": template_scores["top_identity"],
            "top_score": template_scores["top_score"],
            "top_k_scores": template_scores["scores"][: args.top_k],
        }
        if args.threshold is not None:
            output["decision"] = {
                "threshold": float(args.threshold),
                "alarm": bool(template_scores["top_score"] is not None and template_scores["top_score"] >= args.threshold),
                "pred_identity": (
                    template_scores["top_identity"]
                    if template_scores["top_score"] is not None and template_scores["top_score"] >= args.threshold
                    else None
                ),
            }
    elif args.threshold is not None:
        output["decision"] = {
            "threshold": float(args.threshold),
            "alarm": None,
            "pred_identity": None,
            "note": "threshold was provided, but no templates_json was supplied",
        }

    if args.output_json:
        out_path = pathlib.Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(output, indent=2)
        out_path.write_text(text, encoding="utf-8")
    else:
        out_path = None

    print(build_terminal_summary(output, out_path))


if __name__ == "__main__":
    main()