
Dataset summary for AVS_toothbrushes

Project objective: automatic detection of selected toothbrush bristle defects.

Focused defect for this work: Missing bristles (sparse areas).

Current repository layout:

- `.venv/` : Python virtual environment created during development.
- `train/` : original image folders used as source. Contains:
	- `train/good/` : PNG images `000.png` .. `059.png` (good samples)
	- `train/defective/` : PNG images `000.png` .. `029.png` (defective samples)
- `ground_truth/defective/` : mask images `000_mask.png` .. `029_mask.png` corresponding to defective images.
- `data/` : created by `scripts/data_split.py` when the demo is run. Contains `data/train/<class>` and `data/val/<class>`.
- `processed/` : created by `scripts/preprocess.py` (when run). Holds preprocessed images mirroring `data/` or `train/` structure.
- `scripts/` : helper scripts used by the project (see below).
- `run_demo.py` : top-level script that runs the pipeline (splitting, optional preprocessing, feature extraction, rule-based detection evaluation).
- `requirements.txt` : Python dependencies used for the scripts.
- `objective` : project objective text file.
- `README.md` : this file.

Scripts and what they do (current behaviour)

- `requirements.txt` : lists Python packages required by the project (`opencv-python`, `numpy`, `scikit-image`, `pandas`, `scikit-learn`).

- `scripts/data_split.py`
	- Current behaviour: reads `train/<class>` folders and copies images into `data/train/<class>` and `data/val/<class>` according to a `val_frac` parameter. This file is used by `run_demo.py`.

- `scripts/preprocess.py`
	- Current behaviour: reads images (expects a folder structure such as `data/train/<class>`), converts to grayscale, applies Gaussian blur and CLAHE, and writes outputs under `processed/<...>`.

- `scripts/features.py`
	- Current behaviour: provides functions to extract handcrafted features from a single image (and optional mask):
		- `density`: adaptive-threshold-based bristle pixel fraction.
		- `void_frac`: fraction of background component area (measures holes/gaps).
		- `edge_density`: Canny edge-based density.
		- `mean_intensity`: average grayscale intensity.
		- `extract_features(...)` returns a dict with all computed features.

- `scripts/detector.py`
	- Current behaviour: scans `data/train/<class>` (or another provided folder) to build a feature table, computes thresholds from the distribution of `good` images (`density` low threshold, `void_frac` high threshold), and classifies samples as defective when `density` < threshold OR `void_frac` > threshold. Also provides evaluation (classification report and confusion matrix).

- `run_demo.py`
	- Current behaviour: orchestrates the following steps when executed:
		1. runs `scripts/data_split.split_dataset` to create `data/` splits from `train/`;
		2. runs `scripts.preprocess.batch_preprocess` to produce `processed/` images from `data/train` (if present);
		3. extracts features (preferring `processed/` if it exists) and evaluates the rule-based detector, printing thresholds, classification report, and confusion matrix.

How to reproduce the current demo run (commands used during development)

1) create virtual environment and install dependencies:

	 python -m venv .venv
	 .venv\Scripts\python -m pip install --upgrade pip
	 .venv\Scripts\python -m pip install -r requirements.txt

2) run the demo from repository root:

	 .venv\Scripts\python run_demo.py

Current observed outputs (from the latest demo run):

- Rule-based detector printed thresholds and evaluation metrics (classification report and confusion matrix). The demo created `data/` and `processed/` during execution.

If you want changes to the repository layout or behaviour, tell me which specific files or structure you prefer and I will update the code and README accordingly.

