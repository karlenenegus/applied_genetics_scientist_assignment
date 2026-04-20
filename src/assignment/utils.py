import yaml
import os
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger("UTILS")

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return genotype_data, phenotype_data, environment_data

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
