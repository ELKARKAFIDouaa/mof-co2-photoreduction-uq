# Uncertainty-aware machine learning for photocatalytic CO₂ reduction over MOFs

Code and data accompanying the paper:



## Overview

Experimental data on photocatalytic CO₂ reduction over metal–organic frameworks (MOFs) are scattered across the literature, heterogeneous, and not available in any public dataset. This repository contains everything needed to reproduce the study:

1. **An NLP pipeline** that extracts structured experimental records from article PDFs, with a validation layer (precision, recall, F1 against a manually annotated gold set).
2. **A curated dataset** of 316 entries × 40 variables extracted from 83 publications.
3. **A benchmark** of 8 regression and 5 classification models, evaluated on both accuracy (MAE, RMSE, R²; F1, AUC) and uncertainty calibration (ECE, log-likelihood, log-loss).
4. **SHAP interpretation** of the best tree-based model.
5. **A preliminary active-learning simulation** (UCB, EI, PI versus random selection) using a Gaussian process surrogate. This is a proof of concept on the literature dataset, not a validated discovery campaign.



## Installation

Tested with Python [3.x].

```bash
git clone [GitHub URL]
cd mof-co2-photoreduction-uq
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All random seeds are fixed (`SEED = 42`).



## Dataset

`data/curated/dataset.csv` contains one row per experiment (a MOF × reaction-conditions combination).

- **Size:** 316 entries, 40 variables, 83 publications. The regression task uses the 263 entries with a reported production rate.
- **Variable groups:** MOF, structure, electronic properties, reaction conditions, performance, stability, target.
- **Target:** production rate (µmol g⁻¹ h⁻¹), modelled as log(1 + rate).
- **Missing values:** variables with more than 60 % missing values were excluded from modelling (24 of 40 variables retained). An empty value is not always missing: for example, co-catalyst loading is recorded as 0 when no co-catalyst is used, and treated as missing only when a co-catalyst is present but its loading is not reported.
- **Provenance:** each record carries the DOI of its source publication.

A data dictionary describing every variable (name, type, unit, % missing, retained or excluded) is given in `data/curated/DATA_DICTIONARY.md`.

### Known limitations

- The dataset is concentrated on Zr-based UiO and Ti-based MIL materials, and on a typical experiment (300 W Xe lamp, no co-catalyst, no sacrificial agent).
- Metal and MOF family are strongly collinear.
- Reported rates mix several types (average, cumulative, initial) and come from different laboratories.
- Entries from the same publication are not independent ([average] entries per article).
- NLP validation is based on a gold set of [24] annotated records.

## NLP pipeline

The pipeline in `src/nlp/` turns PDFs into structured records in six layers: ingestion, preprocessing, extraction (regular expressions, gazetteers, linguistic rules, tables), normalisation, merging and output/validation. Each extracted value keeps its method and a pointer to its source.

To re-run it, place the PDFs in `data/raw/pdfs/` (not distributed) and run:

```bash
python -m src.nlp.pipeline
```

**Source articles.** The original PDFs are not redistributed because of publisher copyright. `data/raw/papers_doi.csv` lists the DOIs of all 83 publications. Free-text sentences from the articles are not included in the released dataset.

## Headline results

Values below are those reported in the manuscript; regenerate them with `notebooks/04_benchmark_models.ipynb`.

| Task | Result |
|---|---|
| Regression, best R² | GPR (Matérn), R² = [0.715], ECE = [0.083] |
| Regression, best MAE | XGBoost, MAE = [0.867] |
| Regression, best calibration | Ridge, ECE = [0.032] (R² ≈ [0.59]) |
| Classification | [LightGBM], F1 macro = [0.715], ECE = [0.055] |

Accuracy and calibration did not rank the models in the same way. All models were under-confident (prediction intervals wider than necessary). Metrics are computed on out-of-fold predictions from 5-fold cross-validation.

