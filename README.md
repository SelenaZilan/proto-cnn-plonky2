# SmallIris: Privacy-Preserving Iris Recognition via Zero-Knowledge Proofs

A complete zkML system that proves correct execution of a convolutional neural network on a private iris image using the Plonky2 recursive proving system.

## Overview

SmallIris enables an edge device (Prover) to extract a 64-dimensional iris feature vector from a private image and produce a zero-knowledge proof that the extraction was performed correctly. The Verifier can check the proof and use the public feature vector for watchlist matching — without ever seeing the raw iris image.

**Privacy guarantee:** the raw image and all intermediate activations remain hidden; only the 64-D GAP feature sums are public.

**Integrity guarantee:** a successful proof verification confirms the features were derived from the agreed-upon CNN model with correct execution.

### System Components


| Component       | Description                                             | Key files                                             |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------- |
| CNN Model       | SmallIrisCNN (3-layer, 48×48 input, 64-D output)        | `iris-recognition-cnn/models/*.pth`                   |
| Weight Export   | Fused BN + integer quantization (q=4096)                | `iris-recognition-cnn/export_smalliris_zk_weights.py` |
| ZK Prover       | 3-stage recursive Plonky2 proof (L1→R2→R3)              | `src/bin/zk_smalliris_real_recursive.rs`              |
| ZK Verifier     | Standalone proof verification                           | `src/bin/zk_smalliris_real_recursive_verify.rs`       |
| Post-processing | Proof verification + feature recovery + watchlist match | `verifier_postproccess.py`                            |


### Performance (demo FRI ~64-bit security, 20 threads)


| Metric          | Prover        | Verifier                 |
| --------------- | ------------- | ------------------------ |
| Wall-clock time | ~145 s        | < 10 ms                  |
| Peak memory     | 15.5 GB       | 3.2 MB                   |
| Artifact size   | Proof: 117 KB | Verification key: 2.0 KB |


Feature fidelity (cosine similarity between ZK output and PyTorch float): **0.999987**

---

## Quick Start

Requires: Rust nightly toolchain, Python 3 with PyTorch/NumPy.

```bash
# 1. Build prover and verifier binaries
cargo +nightly build --release \
  --bin zk-smalliris-real-recursive \
  --bin zk-smalliris-real-recursive-verify

# 2. Run prover on a test image (generates proof + verification key + manifest)
RAYON_NUM_THREADS=20 ./target/release/zk-smalliris-real-recursive \
  --weights fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64.json \
  --input-image "iris-recognition-cnn/data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test/750_L/S5750L01.png" \
  --demo-fri

# 3. Verify proof + recover features + watchlist match
python verifier_postproccess.py \
  --manifest-json fixtures/smalliris_real_i32_48_c1_24_c2_48_c3_64_S5750L01_proof_manifest.json \
  --templates-json fixtures/watchlist_templates_750_754_lr.json \
  --threshold 0.9529750943183899 \
  --output-json fixtures/watchlist_result_750_754.json
```

The post-processing script internally calls the Rust verifier, so step 2 produces all necessary artifacts and step 3 handles the full verification pipeline.

---

## Project Structure

| File | Description |
| ---- | ----------- |
| `src/bin/zk_smalliris_real_recursive.rs` | 3-stage recursive prover binary |
| `src/bin/zk_smalliris_real_recursive_verify.rs` | Standalone verifier binary |
| `src/smalliris_real_zk.rs` | Circuit building blocks (conv, relu, pool, GAP) |
| `src/smalliris_real_zk_recursive.rs` | Recursive proving logic (L1 → R2 → R3) |
| `iris-recognition-cnn/models.py` | SmallIrisCNN model definition |
| `iris-recognition-cnn/train.py` | Training script |
| `iris-recognition-cnn/export_smalliris_zk_weights.py` | BN fusion + quantized weight export |
| `iris-recognition-cnn/eval_open_set.py` | Open-set recognition evaluation |
| `iris-recognition-cnn/compare_smalliris_zk_fidelity.py` | Float vs ZK fidelity comparison |
| `fixtures/*.json` / `fixtures/*.bin` | Quantized weights, proofs, verification keys, templates |
| `verifier_postproccess.py` | End-to-end verification + feature recovery + watchlist match |

---

## Environment

### Rust

```bash
rustup toolchain install nightly
cargo +nightly build --release \
  --bin zk-smalliris-real-recursive \
  --bin zk-smalliris-real-recursive-verify
```

### Python

```bash
cd iris-recognition-cnn
conda env create -f environment.yml
conda activate pytorch
```

Key dependencies: `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn`.

### Dataset

The preprocessed CASIA-Iris-Thousand images are not included in the repository. Download from:

**[Google Drive: casia-iris-preprocessed](https://drive.google.com/file/d/172GZgrAzNrA146BjFpF1vXtAH-zWymlA/view?usp=sharing)**

The preprocessed dataset is provided by [AndrejHafner/iris-recognition-cnn](https://github.com/AndrejHafner/iris-recognition-cnn).

Place the downloaded folders under `iris-recognition-cnn/data/casia-iris-preprocessed/`:

```
iris-recognition-cnn/data/casia-iris-preprocessed/
  CASIA_thousand_norm_256_64_e_nn_stacked/           # for training
  CASIA_thousand_norm_256_64_e_nn_open_set_stacked/  # for evaluation & ZK demo
```

## References

The open-set evaluation methodology in `iris-recognition-cnn/eval_open_set.py` references the enrollment/matching pipeline from [AndrejHafner/iris-recognition-cnn](https://github.com/AndrejHafner/iris-recognition-cnn).
