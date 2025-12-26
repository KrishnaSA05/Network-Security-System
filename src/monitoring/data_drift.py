import pandas as pd
from scipy.stats import ks_2samp
from utils.logger import logger


def detect_data_drift(reference_path: str, current_path: str):
    ref = pd.read_csv(reference_path)
    curr = pd.read_csv(current_path)

    drift_report = {}
    for col in ref.columns:
        stat = ks_2samp(ref[col], curr[col])
        drift_report[col] = stat.pvalue < 0.05

    logger.info(f"Drift report: {drift_report}")
    return drift_report