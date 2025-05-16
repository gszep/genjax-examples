from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def preprocess_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Japanese gender labels to English and drop the original column.
    """
    df = df.copy()
    sex_mapping = {"男性": "male", "女性": "female"}
    df["sex"] = df["macromill_sex"].map(sex_mapping)
    df.drop(columns=["macromill_sex"], inplace=True)
    return df


def filter_frequent_referers(
    df: pd.DataFrame, tuuid_col: str = "tuuid", referer_col: str = "referer", min_count: int = 100
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return filtered DataFrame with referers that appear >= min_count times per tuuid.
    """
    unique_pairs = df.drop_duplicates(subset=[tuuid_col, referer_col])
    counts = unique_pairs[referer_col].value_counts()
    frequent = counts[counts >= min_count].index.tolist()
    return unique_pairs[unique_pairs[referer_col].isin(frequent)], frequent


def compute_chi2_scores(
    df: pd.DataFrame,
    referers: List[str],
    sex_col: str = "sex",
    age_col: str = "macromill_age",
    referer_col: str = "referer",
) -> List[Tuple[str, float]]:
    """
    Compute chi-squared influence scores of referer usage on gender and age.
    """
    scores = []
    for referer in referers:
        df["has_referer"] = df[referer_col] == referer

        chi2_gender = (
            chi2_contingency(pd.crosstab(df[sex_col], df["has_referer"]))[0] if df[sex_col].nunique() > 1 else 0
        )
        chi2_age = chi2_contingency(pd.crosstab(df[age_col], df["has_referer"]))[0] if df[age_col].nunique() > 1 else 0
        scores.append((referer, chi2_gender + chi2_age))

    return scores


def plot_top_referer_scores(referer_scores: List[Tuple[str, float]], top_n: int = 20, show: bool = True) -> List[str]:
    """
    Plot top N referers by chi-squared score. Return the list of top referers.
    """
    df = pd.DataFrame(referer_scores, columns=["referer", "score"]).sort_values("score", ascending=False).head(top_n)

    if show:
        plt.figure(figsize=(10, 6))
        plt.bar(df["referer"], df["score"])
        plt.title("Top Referers by Chi-square Score")
        plt.xlabel("Referer")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return df["referer"].tolist()


def plot_referer_heatmaps(
    df: pd.DataFrame,
    referers: List[str],
    age_col: str = "macromill_age",
    gender_col: str = "sex",
    referer_col: str = "referer",
    figsize: tuple = (6, 5),
    cmap: str = "Blues",
) -> None:
    """
    Plot age x gender heatmaps for each referer.
    """
    all_ages = sorted(df[age_col].dropna().unique())
    all_genders = sorted(df[gender_col].dropna().unique())

    for referer in referers:
        subset = df[df[referer_col] == referer]
        pivot = pd.crosstab(subset[age_col], subset[gender_col])
        pivot = pivot.reindex(index=all_ages, columns=all_genders, fill_value=0)

        plt.figure(figsize=figsize)
        sns.heatmap(pivot, annot=True, fmt="d", cmap=cmap)
        plt.title(f"Referer: {referer}")
        plt.xlabel("Gender")
        plt.ylabel("Age")
        plt.tight_layout()
        plt.show()


def analyze_referer_influence(
    df: pd.DataFrame,
    top_n: int = 20,
    min_count: int = 100,
    tuuid_col: str = "tuuid",
    referer_col: str = "referer",
    age_col: str = "macromill_age",
    sex_col: str = "sex",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Tuple[str, float]], List[str]]:
    """
    Full pipeline: preprocess data, compute influence scores, and extract top referers.
    """
    preprocessed = preprocess_demographics(df)
    filtered_df, frequent_referers = filter_frequent_referers(preprocessed, tuuid_col, referer_col, min_count)
    scores = compute_chi2_scores(filtered_df, frequent_referers, sex_col, age_col, referer_col)
    top_referers = plot_top_referer_scores(scores, top_n, show=verbose)
    return filtered_df[filtered_df[referer_col].isin(top_referers)], scores, top_referers
