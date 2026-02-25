"""
Utility functions for PBC clinical data analysis.
"""

import numpy as np
import random
from sklearn.metrics import roc_auc_score


# Reproducibility
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)


# Clinical sanity checks
def clinical_sanity_check(df) -> None:
    """
    Perform basic clinical plausibility checks on input dataframe.

    Raises AssertionError if implausible values are detected.
    """
    # Assuming Age might be in days (as in typical PBC datasets)
    # Convert to years for the range check if necessary
    age_in_years = df["Age"] if df["Age"].max() < 120 else df["Age"] / 365.25

    assert (age_in_years.between(18, 100)).all(), "Age out of plausible range"
    assert (df["Bilirubin"] >= 0).all(), "Bilirubin cannot be negative"
    assert (df["Albumin"] > 0).all(), "Albumin must be positive"
    assert (df["Platelets"].dropna() > 0).all(), "Platelet count must be positive"
    assert (df["SGOT"].dropna() >= 0).all(), "SGOT cannot be negative"
    assert (df["Alk_Phos"].dropna() >= 0).all(), "Alkaline phosphatase cannot be negative"


# Feature utilities
def get_feature_groups(df):
    """
    Split dataframe columns into numerical and categorical features.
    """
    num_features = df.select_dtypes(include="number").columns.tolist()
    cat_features = df.select_dtypes(exclude="number").columns.tolist()
    return num_features, cat_features


# Model evaluation
def evaluate_binary_classifier(model, X_test, y_test) -> float:
    """
    Evaluate binary classifier using ROC-AUC.

    Returns:
        float: ROC-AUC score
    """
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)