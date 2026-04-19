import logging
import pandas as pd
import numpy as np
from typing import Optional
import os
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("EXPLORE_DATA")

def split_data(
        data: pd.DataFrame,
        trait_columns: list[str] | str,
        environment_variable_columns: list[str] | str,
        genotype_id_column: Optional[str] = None,
        marker_columns_prefix: Optional[str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into genotype data, phenotype data, and environment data.
    """
    if genotype_id_column is None:
        genotype_id_column = "genotype_id"
        data[genotype_id_column] = data.index
        logger.warning(f"Genotype ID column not provided. Using index as genotype ID.")
    elif genotype_id_column not in data.columns:
        logger.warning(f"Genotype ID column not found. Please check the genotype ID column name. Using index as genotype ID.")
        data[genotype_id_column] = data.index
    
    if isinstance(trait_columns, str):
        trait_columns = [trait_columns]
    if not all(col in data.columns for col in trait_columns):
        raise ValueError(f"Trait columns not found and required for splitting the data. Please check the trait columns names.")
    
    if isinstance(environment_variable_columns, str):
        environment_variable_columns = [environment_variable_columns]
    if not all(col in data.columns for col in environment_variable_columns):
        raise ValueError(f"Environment variable columns not found and required for splitting the data. Please check the environment variable columns names.")

    specified_columns = [genotype_id_column] + trait_columns + environment_variable_columns

    unspecified_columns = data.columns.drop(specified_columns)
    
    detected_marker_columns = [col for col in unspecified_columns if all(data[col].str.match(r'^[ACGT]+$'))]
    
    if marker_columns_prefix is not None:
        specified_marker_columns = [col for col in unspecified_columns if col.startswith(marker_columns_prefix)]
        marker_columns = list(set(specified_marker_columns).intersection(detected_marker_columns))
        if len(marker_columns) == 0:
            logger.warning(f"No marker columns found. Please check the marker columns prefix. Using columns that contain data likely to be nucleotide codes as markers.")
            marker_columns = detected_marker_columns
    else:
        marker_columns = detected_marker_columns

    genotype_data = data[[genotype_id_column] + marker_columns]
    phenotype_data = data[[genotype_id_column] + trait_columns]
    environment_data = data[[genotype_id_column] + environment_variable_columns]
    genotype_data.set_index(genotype_id_column, inplace=True)
    phenotype_data.set_index(genotype_id_column, inplace=True)
    environment_data.set_index(genotype_id_column, inplace=True)
    return genotype_data, phenotype_data, environment_data

def describe_data(data, name: str):
    logger.info(f"{name} contains {data.shape[0]} rows and {data.shape[1]} columns")
    
    missing_counts = data.apply(_check_missing, axis=0)
    logger.info(f"{name} has {missing_counts.sum()} missing values in {sum(missing_counts > 0)} columns.")
    
    data_types = data.dtypes.value_counts()
    if len(data_types) > 1:
        logger.info(f"{name} contains multiple data types ({data_types.index.tolist()}). Please check the data if this is unexpected.")
        data = data.apply(_convert_type, axis=0)
    else:
        logger.info(f"{name} contains one data type: {data_types.index[0]}.")


def _check_missing(data: pd.Series):
    return data.isnull().sum()

def _convert_type(data: pd.Series):
    type_counts = data.apply(type).value_counts()
    if len(type_counts) > 1:
        data = data.astype(type_counts.index[0])
    return data

# def _check_outliers(data: pd.Series):
#     if data.dtype != int and data.dtype != float:
#         return data
#     q1 = data.quantile(0.25)
#     q3 = data.quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     outlier_mask = (data < lower_bound) | (data > upper_bound)
#     if outlier_mask.sum() > 0:
#         logger.warning(f"Column {data.name} has {outlier_mask.sum()} potential outliers.")
#     return data[~outlier_mask]

def plot_phenotype_distribution(data: pd.DataFrame, output_path: str):
    for column_name in data.columns:
        plot_data = data[column_name].dropna()
        mean = plot_data.mean()
        std = plot_data.std()
        n = len(plot_data)
        mu, sigma = mean, std
        df = pd.DataFrame(np.random.normal(mu, sigma, size=n), columns=['normal_distribution'])
        plot_data = pd.concat([plot_data, df], axis=1)
        density_plot = plot_data.plot.kde(title=f"{column_name} Distribution")
        fig = density_plot.get_figure()
        fig.savefig(f"{output_path}/{column_name}_distribution.png")
        plt.close()

def plot_environment_heatmap(data: pd.DataFrame, output_path: str):
    correlation_matrix = data.corr()
    heatmap = sns.heatmap(correlation_matrix, cmap='bwr')
    heatmap.figure.tight_layout()
    plt.title('Environment Variables Correlation Heatmap')
    fig = heatmap.get_figure()
    fig.savefig(f"{output_path}/environment_heatmap.png")
    plt.close()

def plot_environment_boxplots(data: pd.DataFrame, output_path: str):
    data_standardized = data.apply(lambda x: (x - x.mean()) / x.std())
    long_data = data_standardized.melt(var_name='variable', value_name='value').dropna()
    boxplot = sns.boxplot(x = 'variable', y='value', data=long_data)
    plt.xticks(rotation=90, ha='right')
    plt.ylabel('Standardized Values')
    plt.xlabel('Environment Variables')
    plt.title('Environment Variables Boxplots')
    boxplot.figure.tight_layout()
    fig = boxplot.get_figure()
    fig.savefig(f"{output_path}/environment_boxplots.png")
    plt.close()

def main():
    config_file = './config/config.yml'
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    log_path = config['paths']['logs']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = f"{log_path}/job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    if len(config['files']['input']) > 1:
        genotype_file = config['files']['input']['genotype_data']
        phenotype_file = config['files']['input']['phenotype_data']
        environment_file = config['files']['input']['environment_data']
        genotype_data = pd.read_csv(f"{config['paths']['input_data']}/{genotype_file['file_name']}", sep=genotype_file['separator'])
        phenotype_data = pd.read_csv(f"{config['paths']['input_data']}/{phenotype_file['file_name']}", sep=phenotype_file['separator'])
        environment_data = pd.read_csv(f"{config['paths']['input_data']}/{environment_file['file_name']}", sep=environment_file['separator'])
    elif config['files']['input']['combined_data'] is not None:
        combined_file = config['files']['input']['combined_data']
        combined_data = pd.read_csv(f"{config['paths']['input_data']}/{combined_file['file_name']}")
        
        genotype_data, phenotype_data, environment_data = split_data(
            combined_data, 
            genotype_id_column=None, 
            trait_columns=combined_file['trait_columns'], 
            environment_variable_columns=combined_file['environment_variable_columns'],
            marker_columns_prefix=combined_file['marker_columns_prefix']
        )
    describe_data(genotype_data, name="Genotype Data")
    describe_data(phenotype_data, name="Phenotype Data")
    describe_data(environment_data, name="Environment Data")
    
    output_path = config['paths']['output_data']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_phenotype_distribution(phenotype_data.copy(), output_path=output_path)
    plot_environment_heatmap(environment_data.copy(), output_path=output_path)
    plot_environment_boxplots(environment_data.copy(), output_path=output_path)

if __name__ == "__main__":
    main()