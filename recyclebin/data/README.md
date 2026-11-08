# Data

Datasets are **not** committed to the repository. Download them and respect each licence/terms.

## TrashNet (primary dataset, 2,527 images, 6 classes)
1. Download from <https://github.com/garythung/trashnet> (the resized version `dataset-resized.zip`).
2. Unzip so the layout is:
   ```
   data/trashnet/dataset-resized/{cardboard,glass,metal,paper,plastic,trash}/*.jpg
   ```
Class counts: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.

## TACO (real-world spot-check)
1. See <https://github.com/pedropro/TACO> for the download script and annotations.
2. Pick the unseen photos for the spot-check and create `data/taco_spotcheck.csv`:
   ```
   path,true_label
   data/taco/batch_1/000001.jpg,plastic
   ```
   `true_label` must be one of `cardboard, glass, metal, paper, plastic, trash`.
3. TACO has its own fine-grained categories, so each image's label must be mapped to the six
   classes. Record the rule you used in `data/taco_label_mapping.md` (template below) and
   describe how the 12 images were selected (random? first N? by class?).

### taco_label_mapping.md template
| TACO category | Mapped class | Notes |
|---|---|---|
| (fill in) | (fill in) | |
