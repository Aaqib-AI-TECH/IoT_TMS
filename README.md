# XAI-TMS: Trust Management System for IoT Devices

Reproducibility package for *"Trust Management System for IoT Devices: An Autoencoder-Based
Approach with Model-Agnostic Techniques."* This repository reproduces every quantitative
result in the paper from raw data to tables and figures.

## 1. What this contains

| Script | Produces | Paper element |
|---|---|---|
| `validation_configA.py` | Unsupervised detector (AE + reconstruction-error threshold): DR/FAR/F1/accuracy/ROC-AUC/PR-AUC with 95% CIs, ROC & PR curves, threshold-sensitivity sweep | Table 4 (row A) |
| `autoencoder_configB.py` | Supervised variant (AE features + logistic regression) | Table 4 (row B) |
| `shap_analysis.py` | SHAP attribution of the anomaly score, feature ranking, background-size sensitivity | XAI-TMS-3 feature subset, Fig. (SHAP) |
| `counterfactual_generation.py` | Constrained counterfactuals + validity/proximity/sparsity/plausibility + case study | Table 5, Fig. 9 |
| `benchmark_xai_tms.py` | Inference latency (per-sample & batched) and memory footprint
| `statistical_test.py` | Friedman + Wilcoxon across feature sets, with effect sizes

The **data transformation / feature-engineering** code (PCAP parsing, throughput, delay,
latency, transmission-time computation, labelling) is released separately as an installable
package:

> **UNSWTrustworthinessPipeline** — https://github.com/Aaqib-AI-TECH/UNSWTrustworthinessPipeline
> (open access, MIT-licensed, with a tagged release of the transformed dataset).

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.11, TensorFlow 2.21 (CPU). A GPU is not required; the models are small.

## 3. Data

Obtain the transformed UNSW-NB15 dataset either from the dataset release in
`UNSWTrustworthinessPipeline`, or regenerate it from raw PCAPs with that package:

```bash
pip install git+https://github.com/Aaqib-AI-TECH/UNSWTrustworthinessPipeline
unswpipe --in <pcap_dir> --out ./data --name dataset.csv
```

Then set `DATA_PATH` at the top of each script (or run from a directory containing the CSV).
The label column is `label` (0 = trustworthy, 1 = untrustworthy).

## 4. Reproducing the results

Run everything:

```bash
python run_all.py
```

or a subset:

```bash
python run_all.py --only validation_configA.py shap_analysis.py
```

Outputs are written to `outputs_*/` directories as JSON (machine-readable metrics), CSV, and PNG.

## 5. Experimental protocol (splits, seeds, normalisation)

- **Splits.** 70% train+validation / 30% test, stratified by label. The validation set is 20%
  of the *trustworthy* training pool, used only to calibrate the detection threshold.
- **Unsupervised training.** The autoencoder is trained on trustworthy traffic only.
- **Normalisation.** Min-max scaling fitted on the **training data only** and applied to
  validation/test (no leakage). Set `NORM = "minmax"` to reproduce the scaler-sensitivity check.
- **Detection rule.** An instance is flagged untrustworthy iff its reconstruction error exceeds
  the 99th percentile of the validation reconstruction error.
- **Seeds.** All scripts set `PYTHONHASHSEED`, `random`, `numpy`, and `tensorflow` seeds
  (base seed 42). `validation_configA.py` runs 5 seeds (42–46) and reports mean ± 95% CI.
- **Counterfactuals.** Immutable features: `protocol_m`. Mutable features box-constrained to the
  [1st, 99th]-percentile range of trustworthy traffic. Loss weights p, q, r = 1.0, 5.0, 0.5.

## 6. Citation

If you use this code or the transformed dataset, please cite the paper and the
`UNSWTrustworthinessPipeline` repository.
