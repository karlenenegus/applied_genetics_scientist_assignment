import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_config, load_data

logger = logging.getLogger("EXPLORE_DATA")


def describe_data(data, name: str) -> dict:
    logger.info(f"{name} contains {data.shape[0]} rows and {data.shape[1]} columns")
    
    missing_counts = data.apply(_check_missing, axis=0)
    logger.info(f"{name} has {missing_counts.sum()} missing values in {sum(missing_counts > 0)} columns.")
    
    data_types = data.dtypes.value_counts()
    converted_mixed_types = False
    if len(data_types) > 1:
        logger.info(f"{name} contains multiple data types ({data_types.index.tolist()}). Please check the data if this is unexpected.")
        data = data.apply(_convert_type, axis=0)
        converted_mixed_types = True
    else:
        logger.info(f"{name} contains one data type: {data_types.index[0]}.")

    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing_values": int(sum(missing_counts > 0)),
        "dtype_counts": {str(dtype): int(count) for dtype, count in data_types.items()},
        "converted_mixed_types": converted_mixed_types,
    }


def _check_missing(data: pd.Series):
    return data.isnull().sum()

def _convert_type(data: pd.Series):
    type_counts = data.apply(type).value_counts()
    if len(type_counts) > 1:
        data = data.astype(type_counts.index[0])
    return data

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
    config = load_config("./config/config.yml")
        
    log_path = config['paths']['logs']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = f"{log_path}/job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_file, level=config['logging']['level'])
    
    genotype_data, phenotype_data, environment_data = load_data(config)

    genotype_metrics = describe_data(genotype_data, name="Genotype Data")
    phenotype_metrics = describe_data(phenotype_data, name="Phenotype Data")
    environment_metrics = describe_data(environment_data, name="Environment Data")
    
    output_path = config['paths']['output_data']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    exploratory_data_metrics = {
        "genotype_data": genotype_metrics,
        "phenotype_data": phenotype_metrics,
        "environment_data": environment_metrics,
    }
    with open(f"{output_path}/exploratory_data_metrics.json", "w") as f:
        json.dump(exploratory_data_metrics, f, indent=4)

    plot_phenotype_distribution(phenotype_data.copy(), output_path=output_path)
    plot_environment_heatmap(environment_data.copy(), output_path=output_path)
    plot_environment_boxplots(environment_data.copy(), output_path=output_path)

if __name__ == "__main__":
    main()