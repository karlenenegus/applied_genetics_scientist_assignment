import os
from datetime import datetime
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from .utils import load_config

logger = logging.getLogger("MODEL")

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan

def mixed_solve_rrblup(y, Z, K=None, return_hinv=False, method="REML"):
    """
    Wrapper for rrBLUP::mixed.solve via rpy2.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response vector.
    Z : array-like, shape (n, m)
        Marker/random-effect design matrix.
    K : array-like, optional
        Covariance matrix for random effects (m x m).
    return_hinv : bool, optional
        Whether to return H inverse from rrBLUP.
    method : str, optional
        Either 'REML' or 'ML'. Defaults to 'REML'.

    Returns
    -------
    dict
        Keys may include: 'u', 'beta', 'Ve', 'Vu', 'LL', 'Hinv'
    """
    if ro is None or pandas2ri is None or importr is None:
        raise ImportError(
            "rpy2 is required for mixed_solve_rrblup. "
            "Install rpy2 and ensure R is available."
        )
    if method not in {"REML", "ML"}:
        raise ValueError("method must be either 'REML' or 'ML'.")

    rrblup = importr("rrBLUP")
    y_df = pd.DataFrame({"y": np.asarray(y).reshape(-1)})
    Z_df = pd.DataFrame(np.asarray(Z))
    args = {"y": None, "Z": None, "SE": False, "return_Hinv": return_hinv, "method": method}

    with (ro.default_converter + pandas2ri.converter).context():
        args["y"] = ro.conversion.get_conversion().py2rpy(y_df["y"])
        args["Z"] = ro.conversion.get_conversion().py2rpy(Z_df)
        if K is not None:
            K_df = pd.DataFrame(np.asarray(K))
            args["K"] = ro.conversion.get_conversion().py2rpy(K_df)
        ans = rrblup.mixed_solve(**args)
        out = {}

        for name in ans.names():
            #ro.conversion.get_conversion().rpy2py(ans)
            out[str(name)] = ro.conversion.get_conversion().rpy2py(ans.getbyname(name))
    return out

def fit_models(train_data: dict, trait: str, output_path: str) -> dict:

    train_data['phenotype_data'] = train_data['phenotype_data'][trait].dropna()
    train_data['environment_data'] = train_data['environment_data'].dropna()
    combined_train_data = train_data['genotype_data'].merge(train_data['phenotype_data'], left_index=True, right_index=True, how='inner')
    combined_train_data = combined_train_data.merge(train_data['environment_data'], left_index=True, right_index=True, how='inner')

    env_columns = train_data['environment_data'].columns.tolist()
    y_train = combined_train_data[trait].to_numpy(dtype=float)
    g_train = combined_train_data.drop(columns=[trait]+env_columns).to_numpy(dtype=float) - 1
    rrblup_model = mixed_solve_rrblup(y_train, g_train)
    u = rrblup_model['u']
    beta = rrblup_model['beta']
    rrblup_model = {
        'u': u,
        'beta': beta
    }

    e_train = combined_train_data[env_columns].to_numpy(dtype=float)
    linear_model = LinearRegression()
    linear_model.fit(e_train, y_train)
    trait_model_path = f'{output_path}/{trait}'
    if not os.path.exists(trait_model_path):
        os.makedirs(trait_model_path)
    save_models(rrblup_model, linear_model, trait_model_path)

def predict_models(validation_data: dict, trait: str, model_path: str) -> dict: 

    validation_data['phenotype_data'] = validation_data['phenotype_data'][trait].dropna()
    validation_data['environment_data'] = validation_data['environment_data'].dropna()
    combined_validation_data = validation_data['genotype_data'].merge(validation_data['phenotype_data'], left_index=True, right_index=True, how='inner')
    combined_validation_data = combined_validation_data.merge(validation_data['environment_data'], left_index=True, right_index=True, how='inner')

    env_columns = validation_data['environment_data'].columns.tolist()
    trait_model_path = f'{model_path}/{trait}'
    rrblup_model, linear_model = load_models(trait_model_path)

    predictions_data = pd.DataFrame({f'observed_{trait}': combined_validation_data[trait]})

    g_val = combined_validation_data.drop(columns=[trait]+env_columns).to_numpy(dtype=float) - 1
    rrblup_y_pred = g_val @ rrblup_model['u'] + rrblup_model['beta']
    rrblup_y_pred = rrblup_y_pred.reshape(-1)
    predictions_data[f'predicted_{trait}_rrblup'] = rrblup_y_pred
    
    e_val = combined_validation_data[env_columns].to_numpy(dtype=float)
    linear_y_pred = linear_model.predict(e_val)
    predictions_data[f'predicted_{trait}_linear'] = linear_y_pred

    return predictions_data

def save_models(rrblup_model: dict, linear_model: dict, output_path: str) -> dict:
    with open(f'{output_path}/rrblup_model.pkl', 'wb') as f:
        pickle.dump(rrblup_model, f)
    
    with open(f'{output_path}/linear_model.pkl', 'wb') as f:
        pickle.dump(linear_model, f)

def load_models(model_path: str) -> dict:
    with open(f'{model_path}/rrblup_model.pkl', 'rb') as f:
        rrblup_model = pickle.load(f)
    
    with open(f'{model_path}/linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    return rrblup_model, linear_model
    

def main():
    config = load_config("./config/config.yml")

    log_path = config['paths']['logs']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = f"{log_path}/job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_file, level=config['logging']['level'])

    with open(f'{config["paths"]["output_data"]}/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'{config["paths"]["output_data"]}/validation_data.pkl', 'rb') as f:
        validation_data = pickle.load(f)
    
    predictions_data = pd.DataFrame()
    for trait in train_data['phenotype_data'].columns:
        fit_models(train_data, trait, config['paths']['models'])
        trait_predictions = predict_models(validation_data, trait, config['paths']['models'])
        predictions_data = pd.concat([predictions_data, trait_predictions], axis=1)

    accuracy_metrics = {}
    for column in predictions_data.columns:
        if column.startswith('predicted_'):
            rmse_value = rmse(predictions_data[column], predictions_data[f'observed_{trait}'])
            corr_value = correlation(predictions_data[column], predictions_data[f'observed_{trait}'])
            accuracy_metrics[column] = {"rmse": rmse_value, "correlation": corr_value}
        
    pd.DataFrame(accuracy_metrics).T.to_csv(f'{config["paths"]["output_data"]}/accuracy_metrics.csv')
        
if __name__ == "__main__":
    main()