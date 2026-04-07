# Solar Flare Intensity Prediction

Predicting solar flare peak X-ray flux using GOES satellite time-series data and SHARP magnetogram features from NASA's Solar Dynamics Observatory. Built as an end-to-end ML project covering data wrangling, feature engineering, classical ML, and deep learning.

---

## Project structure

```
solar-flare-prediction/
├── data/
│   ├── raw/            ← GOES .csv downloads, SHARP FITS files (DVC-tracked)
│   ├── interim/        ← partially cleaned data (DVC-tracked)
│   └── processed/      ← feature-engineered, model-ready (DVC-tracked)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_deep_learning.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
├── configs/            ← YAML hyperparameter configs
├── .dvc/               ← DVC metadata (committed)
├── .env.example        ← secrets template (committed, no values)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/solar-flare-prediction.git
cd solar-flare-prediction

conda create -n solar-flare python=3.11
conda activate solar-flare

pip install -r requirements.txt
```

### 2. Set up secrets

```bash
cp .env.example .env
# open .env and fill in your WANDB_API_KEY and JSOC_EMAIL
```

### 3. Pull data (requires DVC remote access)

```bash
dvc pull
```

### 4. Launch notebooks

```bash
jupyter lab
```

---

## Data sources

| Source | What it provides | Access |
|---|---|---|
| [NOAA GOES-R](https://www.ngdc.noaa.gov/stp/satellite/goes-r.html) | X-ray flux time series (1-min cadence) | Public, no key needed |
| [NASA JSOC / SHARP](http://jsoc.stanford.edu/) | Magnetogram features per active region | Free, email registration |
| [NOAA Solar Event Lists](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/) | Flare labels and classifications | Public |

---

## Pipeline overview

```
GOES X-ray flux + SHARP magnetograms
        ↓
  Data cleaning (NaNs, dropout, resampling)
        ↓
  Feature engineering (rolling stats, lag features, flux derivatives)
        ↓
  Baseline models (Random Forest → XGBoost → LSTM)
        ↓
  Deep learning extension (CNN on magnetograms + tabular fusion)
        ↓
  Experiment tracking (Weights & Biases)
```

---

## Results

| Model | RMSE | MAE | TSS |
|---|---|---|---|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| LSTM | — | — | — |
| CNN + Tabular Fusion | — | — | — |

*Results will be filled in as experiments complete.*

---

## Key design decisions

**Time-series cross-validation.** Standard k-fold would leak future data into training. All models use `TimeSeriesSplit` from scikit-learn.

**Evaluation metric.** TSS (True Skill Statistic) is the standard in solar flare forecasting — it handles class imbalance better than accuracy or F1.

**Data versioning.** Raw and processed data are tracked with DVC, not Git. Pointer `.dvc` files are committed so any collaborator can reproduce the exact dataset.

**Notebook hygiene.** `nbstripout` is installed as a pre-commit hook — all cell outputs are stripped before every commit.

---

## Environment

- Python 3.11
- PyTorch (CPU — no GPU required for baseline experiments)
- numpy < 2.0 (pinned for PyTorch compatibility)

Full dependency list in `requirements.txt`.

---

## References

- Bobra & Couvidat (2015) — [Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data](https://doi.org/10.1088/0004-637X/798/2/135)
- NOAA Space Weather Prediction Center — [Flare Classification](https://www.swpc.noaa.gov/phenomena/solar-flares-radio-blackouts)
