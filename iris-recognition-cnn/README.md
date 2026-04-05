# SmallIris CNN – Training & Evaluation

CNN training and evaluation code for the SmallIris ZKP project.

## Files

| File | Purpose |
|------|---------|
| `models.py` | `SmallIrisCNN` model definition |
| `train.py` | Training script |
| `export_smalliris_zk_weights.py` | Export fused BN + quantized weights as JSON for Plonky2 |
| `eval_open_set.py` | Open-set identification evaluation (rank-k accuracy) |
| `eval_watchlist_match.py` | 1:N watchlist detection evaluation |
| `calibrate_watchlist_threshold.py` | Threshold calibration for watchlist matching |
| `compare_smalliris_zk_fidelity.py` | Feature fidelity: cosine similarity between float and ZK outputs |

## Setup

```bash
conda env create -f environment.yml
conda activate pytorch
```

## Dataset

The preprocessed CASIA-Iris-Thousand dataset is from [AndrejHafner/iris-recognition-cnn](https://github.com/AndrejHafner/iris-recognition-cnn). Download from [Google Drive](https://drive.google.com/drive/folders/1US7deawGYcoEh0B92-IZOxUllj5Dp8By?usp=sharing) and place under `data/casia-iris-preprocessed/`:

```
data/casia-iris-preprocessed/
  CASIA_thousand_norm_256_64_e_nn_stacked/           # for training
  CASIA_thousand_norm_256_64_e_nn_open_set_stacked/  # for evaluation & ZK demo
```

Both `data/` and `models/` are gitignored.

## References

The open-set evaluation pipeline (`eval_open_set.py`) references the enrollment/matching methodology from [AndrejHafner/iris-recognition-cnn](https://github.com/AndrejHafner/iris-recognition-cnn).
