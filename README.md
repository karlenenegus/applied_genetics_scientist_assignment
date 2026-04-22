# README

## Applied Genetics Scientist Assignment

Take-home assignment for the Applied Genetics Scientist position. Includes exploratory data analysis, data preprocessing, feature extraction, and prediction modeling. Completed in 5.5 hours.

## Project Explanation

- Loads combined input data using `config/config.yml`. 
- Splits combined input data into genotype, phenotype, and environment tables based on column designations.
- Runs exploratory plots and generates exploratory data summary.
- Preprocesses data (genotype encoding, filtering, imputation, normalization, train/validation split) and extracts data type specific feature summaries.
- Trains models and writes models, predictions, and accuracy metrics.

## Project Structure

- `src/__main__.py` - Pipeline entrypoint (`explore_data -> preprocess_data -> model`).
- `src/assignment/utils.py` - Config and data loading/splitting helpers.
- `src/assignment/explore_data.py` - Exploratory data plots and dataset diagnostic summaries.
- `src/assignment/preprocess_data.py` - Preprocesses data by converting genotypes to numeric and normalizing phenotype and environment data. Extracts feature summaries for genotype, phenotype, and environment data.
- `src/assignment/model.py` - Model fitting, model saving, prediction, and accuracy metric export.
- `config/config.yml` - Paths, file definitions, and validation settings.

## Environment Setup

Using micromamba:

```bash
micromamba create -y -n assignment -f environment.yaml
micromamba activate assignment
```

Or alternatively using conda:

```bash
conda create -y -n assignment -f environment.yaml
conda activate assignment
```

## How to Run

From the repository root:

```bash
python -m src
```

This code runs these three processes in order:

1. `explore_data.main()`
2. `preprocess_data.main()`
3. `model.main()`

## Configuration

Edit `config/config.yml` for:

- Input file paths (`paths`)
- Input file characteristics (`files.input`)
- Validation split settings (`validation.split_ratio`, `validation.random_state`)
- Logging preferences

Currently the config defaults point to:

- Input: `./data/input`
- Output: `./data/output`
- Models: `./data/models`
- Logs: `./logs`

## Outputs

Typical files generated in `data/output` include:

From `explore_data.py`:
- `cucumber_weight_distribution.png`
- `environment_boxplots.png`
- `environment_heatmap.png`
- `exploratory_data_metrics.json`

From `preprocess_data.py`:
- `dataset_metrics.json`
- `normalization_values_phenotype.json`
- `normalization_values_environment.json`
- `train_data.pkl`
- `validation_data.pkl`

From `model.py`
- `predicted_phenotypes.csv`
- `accuracy_metrics.csv`


Typical files generated in `data/models` include:

From `model.py`
- `linear_model.pkl`
- `rrblup_model.pkl`
