# Results

`reported_*` files contain the values transcribed from the team presentation. They let the
paper's tables and figures be regenerated without retraining (`python -m recyclebin.figures`).

> **Open issue:** the TACO slide text reports 50% (6/12) but the labelled photo panel
> (`reported_taco_spotcheck_slide.csv`) shows 2/12 correct (16.7%). Re-run
> `python -m recyclebin.taco_eval` to produce the authoritative number and update the paper.

Running `recyclebin.evaluate` writes new, authoritative outputs to `results/baseline/` and
`results/mobilenetv2/` (predictions, metrics, confusion matrix).
