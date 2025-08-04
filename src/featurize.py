import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List

def extract_features_and_clean(
    db_path: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extract features from SQLite DB, perform cleaning and return X, y, and feature lists.
    Args:
        db_path: Path to SQLite database file.
    Returns:
        X: Feature DataFrame
        y: Target Series
        numeric: List of numeric feature names
        categorical: List of categorical feature names
    """
    query = """
    SELECT
        t.id AS tooth_id,
        t.patientId,
        p.age,
        p.sex,
        t.toothType,
        t.area AS tooth_area,
        COALESCE(c.cavity_count, 0) AS cavity_count,
        COALESCE(c.cavity_area_sum, 0) AS cavity_area_sum,
        COALESCE(c.cavity_area_mean, 0) AS cavity_area_mean,
        CASE WHEN tr.toothId IS NOT NULL THEN 1 ELSE 0 END AS needs_treatment
    FROM Teeth t
    JOIN Patients p ON t.patientId = p.id
    LEFT JOIN (
        SELECT
            toothId,
            COUNT(*) AS cavity_count,
            SUM(area) AS cavity_area_sum,
            AVG(area) AS cavity_area_mean
        FROM Cavities
        GROUP BY toothId
    ) c ON t.id = c.toothId
    LEFT JOIN Treatments tr ON t.id = tr.toothId
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Standardize sex column to uppercase for consistency
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.upper()

    # Drop rows with missing toothType
    df = df.dropna(subset=["toothType"])

    # Impute area by group mean (toothType, sex), fallback to global mean
    group_means = df.groupby(["toothType", "sex"])['tooth_area'].mean().to_dict()
    global_mean = df['tooth_area'].mean()
    def impute_area(row):
        if pd.isna(row['tooth_area']):
            key = (row['toothType'], row['sex'])
            return group_means.get(key, global_mean)
        return row['tooth_area']
    df['tooth_area'] = df.apply(impute_area, axis=1)

    numeric = ["age", "tooth_area", "cavity_count", "cavity_area_sum", "cavity_area_mean"]
    categorical = ["sex", "toothType"]
    X = df[numeric + categorical]
    y = df["needs_treatment"]
    return X, y, numeric, categorical
