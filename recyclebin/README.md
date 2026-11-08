# RecycleBin: Transfer Learning for Household Waste Classification

Code, results and figures for the paper **"Data-Driven Evaluation of Transfer Learning for
AI-Powered Household Waste Classification"** (Rithika Pujari, DTSC 5082, University of North Texas).

The project compares a **CNN trained from scratch** with **MobileNetV2 (ImageNet initialisation)**
on the six-class [TrashNet](https://github.com/garythung/trashnet) task
(cardboard, glass, metal, paper, plastic, trash) and spot-checks the model on a small set of
real-world [TACO](https://github.com/pedropro/TACO) photos.

**Research question:** Can transfer learning classify common household waste accurately enough
to support useful disposal guidance?

## Reported results (from the team presentation)

| Metric (TrashNet test, n = 380) | Baseline CNN | MobileNetV2 |
|---|---|---|
| Accuracy | 53% | 83% (317/380) |
| Precision (macro) | 0.57 | 0.80 |
| Recall (macro) | 0.56 | 0.81 |
| Macro F1 | 0.50 | 0.80 |

Per-class accuracy (MobileNetV2): cardboard 0.97, paper 0.84, metal 0.84, glass 0.83, plastic 0.78, trash 0.60.
Full confusion matrix: `results/reported_confusion_matrix_mobilenetv2.csv`.

> **Status note.** The `reported_*` files are transcribed from the presentation. The TACO spot-check is
> unresolved: slide text says 50% (6/12) but the labelled panel shows 2/12 correct. Re-run
> `taco_eval` and update the paper with the verified number. See `results/README.md`.

## Repository layout

```
configs/config.yaml        all settings (seed, split, image size, hyper-parameters)
data/README.md             how to download TrashNet and TACO, label mapping template
src/recyclebin/
  data.py                  stratified split (saved to CSV) + tf.data pipeline
  models.py                baseline CNN and MobileNetV2 transfer model
  train.py                 train a model
  evaluate.py              test-set metrics, confusion matrix, per-image predictions
  taco_eval.py             real-world TACO spot-check
  stats.py                 Wilson CIs, two-proportion z-test, paired McNemar test
  figures.py               regenerate paper figures
results/                   reported values + figures (outputs of new runs go here too)
tests/                     unit tests for the statistics helpers
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
Python 3.10+ and TensorFlow 2.15+ are expected.

## Reproduce

```bash
# 1. Download data (see data/README.md), then:
python -m recyclebin.train    --model baseline
python -m recyclebin.train    --model mobilenetv2
python -m recyclebin.evaluate --model baseline
python -m recyclebin.evaluate --model mobilenetv2

# 2. Real-world spot-check (needs data/taco_spotcheck.csv)
python -m recyclebin.taco_eval --model mobilenetv2 --csv data/taco_spotcheck.csv

# 3. Statistics
python -m recyclebin.stats ci 317 380
python -m recyclebin.stats mcnemar results/baseline/predictions.csv results/mobilenetv2/predictions.csv

# 4. Figures (works without TensorFlow, using the reported values)
python -m recyclebin.figures
```

## Before you publish: confirm these match the paper

`configs/config.yaml` ships with **placeholder defaults** marked `CONFIRM`. The presentation did not
document them, so copy the exact values from the original notebook:

- random seed and split proportions (a 15% test fraction gives the reported 380 test images)
- image size, batch size, augmentation settings
- baseline CNN architecture (`models.py::build_baseline_cnn` is a generic example)
- MobileNetV2 frozen vs fine-tuned layers, optimizer, learning rate, epochs, early stopping
- TACO image selection and label mapping

The training and evaluation scripts have not been run against the datasets in this repository
snapshot; run them once and check that the metrics match the paper.

## Responsible use

The classifier predicts a **material category**, not a local recycling rule. Disposal guidance depends
on municipal policy and on the item's condition, so model output should be shown with its confidence
and combined with location-specific rules. Datasets are public image collections; follow their terms.

## Citation / references

Yang & Thung (2016) TrashNet; Proença & Simões (2020) TACO; Sandler et al. (2018) MobileNetV2.
See the paper for the full IEEE reference list.

## Licence

MIT (see `LICENSE`). Datasets keep their own licences.
