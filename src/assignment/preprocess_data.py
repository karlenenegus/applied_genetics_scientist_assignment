import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import random
from typing import Optional
import json

from .utils import load_config, load_data

logger = logging.getLogger("PREPROCESS_DATA")

def filter_columns_for_missing(data: pd.DataFrame, min_missing_percentage: float = 0.75) -> pd.DataFrame:
    """
    Filter the columns for missing data.
    """
    missing_counts = data.isna().sum()
    total_observations = data.shape[0]
    min_missing_observations = total_observations * min_missing_percentage
    keep_columns_mask = missing_counts < min_missing_observations
    data = data.loc[:, keep_columns_mask]
    if sum(~keep_columns_mask) != 0:
        logger.info(f"{sum(~keep_columns_mask)} columns with less than {min_missing_percentage * 100}% of values were removed due to missing data.")
    
    return data, keep_columns_mask

def genotypes_to_numeric(data: pd.Series) -> pd.Series:
    """
    Convert the genotype data to allele dosage numeric data.
    """
    allele_1 = data.str[0]
    allele_2 = data.str[1]
    allele_counts = pd.concat([allele_1, allele_2], axis=0).value_counts()

    if len(allele_counts) == 0:
        logger.warning(f"Invalid genotype data for locus {data.name}. Setting to missing values.")
        return pd.Series([np.nan] * len(data))

    major_allele = allele_counts.index[0]
    homozygous_major = f"{major_allele}{major_allele}"
    dosage_dict = {homozygous_major: 0}

    if len(allele_counts) >= 2:
        if len(allele_counts) > 2:
            logger.warning(f"More than two alleles found in the genotype data. Using only the two most frequent alleles. Please check locus {data.name} for more information.")
        major_allele = allele_counts.index[0]
        minor_allele = allele_counts.index[1]
        homozygous_major = f"{major_allele}{major_allele}"
        homozygous_minor = f"{minor_allele}{minor_allele}"
        heterozygous = f"{major_allele}{minor_allele}"
        alt_heterozygous = f"{minor_allele}{major_allele}"

        dosage_dict.update({
            homozygous_minor: 2,
            heterozygous: 1,
            alt_heterozygous: 1
        })
    
    data = data.map(dosage_dict)
    return data

def _filter_and_impute_genotype_data(genotype_data, mode="fit", values_for_predict: Optional[dict] = None):
    """
    Filter and impute the genotype data.

    mode: str = "fit" or "predict"
    """
    if mode == "fit":
        min_missing_genos = genotype_data.shape[0] * 0.5
        mask_missing = [snp <= min_missing_genos for snp in genotype_data.isna().sum(axis=0)]
        genotype_data = genotype_data.loc[:, mask_missing]

        mask_allele_freq = genotype_data.apply(lambda x: x.nunique() > 1, axis=0)
        genotype_data = genotype_data.loc[:, mask_allele_freq]

        genotype_data_columns = genotype_data.columns.tolist()
        genotype_data_mean = genotype_data.mean()
        genotype_data = genotype_data.fillna(genotype_data_mean)

        values_for_predict = {'snp_columns': genotype_data_columns, 'snp_means': genotype_data_mean}

    if mode == "predict":
        if values_for_predict is None:
            raise ValueError("Values for predict must be provided when mode is predict.")

        genotype_data = genotype_data.reindex(columns=values_for_predict['snp_columns'])
        genotype_data = genotype_data.fillna(values_for_predict['snp_means'])

    return genotype_data, values_for_predict

def normalize_data(data: pd.DataFrame, mode: str = "fit", values_for_predict: Optional[dict] = None, output_file_path: str = None) -> pd.DataFrame:
    """
    Normalize the data.

    mode: str = "fit" or "predict"
    """
    if mode == "fit":
        values_for_predict = {}
        for column in data.columns:
            column_mean = data[column].mean()
            column_std = data[column].std()
            values_for_predict.update({column: {'mean': column_mean, 'std': column_std}})
            data[column] = (data[column] - column_mean) / column_std
        with open(output_file_path, 'w') as f:
            json.dump(values_for_predict, f, indent= 4)
    elif mode == "predict":
        if values_for_predict is None:
            raise ValueError("Values for predict must be provided when mode is predict.")
        for column in data.columns:
            if column not in values_for_predict.keys():
                raise ValueError(f"Column {column} not found in values for predict.")
            column_mean = values_for_predict[column]['mean']
            column_std = values_for_predict[column]['std']
            data[column] = (data[column] - column_mean) / column_std

    return data, values_for_predict

def calculate_genotype_metrics(genotype_data: pd.DataFrame) -> dict:
    """
    Calculate the genotype metrics.
    """
    genotype_data_long = genotype_data.melt()
    genotype_data_long.set_index('variable', inplace=True)

    total_genotypes = genotype_data_long.shape[0]

    completeness = ((~genotype_data_long['value'].isna()).sum() / total_genotypes)
    heterozygosity = ((genotype_data_long['value'] == 1.0).sum() / total_genotypes)
    minor_allele_frequency = (((genotype_data_long['value'] == 1.0).sum() + ((genotype_data_long['value'] == 2.0).sum()*2)) / (total_genotypes*2))
    genotype_metrics = {
        'completeness': f'{round(completeness, 4) * 100}%',
        'average_heterozygosity': f'{round(heterozygosity, 4) * 100}%',
        'average_minor_allele_frequency': f'{round(minor_allele_frequency, 4) * 100}%'
        }
    return genotype_metrics

def calculate_grouped_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate the grouped metrics.
    """
    data_long = data.melt()
    data_long['variable_group'] = data_long['variable'].str.replace(r"_\d+$", '', regex=True)
    grouped_metrics = data_long.drop(columns=['variable']).groupby('variable_group').agg({
        'value': ['mean', 'median', 'std', 'min', 'max']
    })

    grouped_metrics.columns = grouped_metrics.columns.get_level_values(1)
    return grouped_metrics.to_dict(orient='index')

def validation_split(
        genotype_data: pd.DataFrame, 
        phenotype_data: pd.DataFrame, 
        environment_data: pd.DataFrame, 
        validation_fraction: float, 
        random_state: int = 42
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:

    """
    Split the data into train and validation sets.
    """
    unique_genotype_ids = genotype_data.index.unique().tolist()
    validation_indices = random.sample(unique_genotype_ids, int(len(unique_genotype_ids) * validation_fraction))
    train_indices = set(unique_genotype_ids) - set(validation_indices)
    train_indices = list(train_indices)
    train_genotype_data = genotype_data.loc[train_indices]
    train_phenotype_data = phenotype_data.loc[train_indices]
    train_environment_data = environment_data.loc[train_indices]
    validation_genotype_data = genotype_data.loc[validation_indices]
    validation_phenotype_data = phenotype_data.loc[validation_indices]
    validation_environment_data = environment_data.loc[validation_indices]
    return (train_genotype_data, train_phenotype_data, train_environment_data), (validation_genotype_data, validation_phenotype_data, validation_environment_data)

def main():
    config = load_config("./config/config.yml")

    log_path = config['paths']['logs']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = f"{log_path}/job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_file, level=config['logging']['level'])

    genotype_data, phenotype_data, environment_data = load_data(config)
    genotype_data = genotype_data.apply(genotypes_to_numeric, axis=0)
    
    genotype_metrics = calculate_genotype_metrics(genotype_data)
    environment_metrics = calculate_grouped_metrics(environment_data)
    phenotype_metrics = calculate_grouped_metrics(phenotype_data)

    metrics = {
        'genotype_metrics': genotype_metrics,
        'environment_metrics': environment_metrics,
        'phenotype_metrics': phenotype_metrics
    }
    with open(f'{config["paths"]["output_data"]}/dataset_metrics.json', 'w') as f:
        json.dump(metrics, f, indent= 4)
    
    train_data, validation_data = validation_split(
        genotype_data=genotype_data, 
        phenotype_data=phenotype_data, 
        environment_data=environment_data, 
        validation_fraction=config['validation']['split_ratio'], 
        random_state=config['validation']['random_state']
    )

    train_genotype_data, train_phenotype_data, train_environment_data = train_data
    validation_genotype_data, validation_phenotype_data, validation_environment_data = validation_data
    
    train_genotype_data, values_for_predict = _filter_and_impute_genotype_data(train_genotype_data, mode="fit")
    validation_genotype_data, _ = _filter_and_impute_genotype_data(validation_genotype_data, mode="predict", values_for_predict=values_for_predict)

    train_environment_data, keep_columns_mask = filter_columns_for_missing(train_environment_data, min_missing_percentage=0.75)
    validation_environment_data = validation_environment_data.loc[:, keep_columns_mask]
    
    output_path = config['paths']['output_data']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    normalization_values_phenotype_file_path = f'{output_path}/normalization_values_phenotype.json'
    normalization_values_environment_file_path = f'{output_path}/normalization_values_environment.json'

    train_phenotype_data, values_for_predict = normalize_data(train_phenotype_data, mode="fit", output_file_path=normalization_values_phenotype_file_path)
    validation_phenotype_data, _ = normalize_data(validation_phenotype_data, mode="predict", values_for_predict=values_for_predict)

    train_environment_data, values_for_predict = normalize_data(train_environment_data, mode="fit", output_file_path=normalization_values_environment_file_path)
    validation_environment_data, _ = normalize_data(validation_environment_data, mode="predict", values_for_predict=values_for_predict)

    with open(f'{output_path}/train_data.pkl', 'wb') as f:
        pickle.dump({
            'genotype_data': train_genotype_data, 
            'phenotype_data': train_phenotype_data, 
            'environment_data': train_environment_data
        }, f)

    with open(f'{output_path}/validation_data.pkl', 'wb') as f:
        pickle.dump({
            'genotype_data': validation_genotype_data, 
            'phenotype_data': validation_phenotype_data, 
            'environment_data': validation_environment_data
        }, f)

if __name__ == "__main__":
    main()
    
    