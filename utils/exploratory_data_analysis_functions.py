import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups"""
    pooled_std = np.sqrt(
        ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var())
        / (len(group1) + len(group2) - 2)
    )
    return (group2.mean() - group1.mean()) / pooled_std


def plot_numerical_feature_distribution(data, feature, target_col="target", ax=None):
    """
    Create line plot showing distribution of numerical feature by target groups

    Args:
        data (DataFrame): Input dataframe
        feature (str): Feature column name
        target_col (str): Target column name
        ax (matplotlib.axes): Axis to plot on

    Returns:
        dict: Statistics for the feature
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    no_default = data[data[target_col] == 0][feature].dropna()
    default = data[data[target_col] == 1][feature].dropna()

    x_min = min(no_default.min(), default.min())
    x_max = min(no_default.quantile(0.95), default.quantile(0.95))  # Remove outliers

    x_range = np.linspace(x_min, x_max, 100)

    try:
        kde_no_default = gaussian_kde(no_default)
        kde_default = gaussian_kde(default)

        density_no_default = kde_no_default(x_range)
        density_default = kde_default(x_range)

        ax.plot(
            x_range,
            density_no_default,
            color="skyblue",
            linewidth=2.5,
            label="No Default",
            alpha=0.8,
        )
        ax.plot(
            x_range,
            density_default,
            color="salmon",
            linewidth=2.5,
            label="Default",
            alpha=0.8,
        )

        ax.fill_between(x_range, density_no_default, alpha=0.3, color="skyblue")
        ax.fill_between(x_range, density_default, alpha=0.3, color="salmon")

    except Exception as e:
        ax.hist(
            [no_default, default],
            bins=30,
            alpha=0.7,
            label=["No Default", "Default"],
            color=["skyblue", "salmon"],
            density=True,
        )

    ax.set_title(f"{feature.replace('_', ' ').title()}", fontweight="bold", fontsize=12)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate statistics
    cohens_d = calculate_cohens_d(no_default, default)
    _, p_value = stats.ttest_ind(default, no_default)

    effect_size = (
        "Small" if abs(cohens_d) < 0.2 else "Medium" if abs(cohens_d) < 0.5 else "Large"
    )
    significance = (
        "***"
        if p_value < 0.001
        else "**"
        if p_value < 0.01
        else "*"
        if p_value < 0.05
        else ""
    )

    return {
        "feature": feature,
        "no_default_mean": no_default.mean(),
        "no_default_std": no_default.std(),
        "default_mean": default.mean(),
        "default_std": default.std(),
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "p_value": p_value,
        "significance": significance,
        "mean_difference": abs(default.mean() - no_default.mean()),
    }


def plot_categorical_feature_distribution(data, feature, target_col="target", ax=None):
    """
    Create bar plot showing default rates for categorical feature

    Args:
        data (DataFrame): Input dataframe
        feature (str): Feature column name
        target_col (str): Target column name
        ax (matplotlib.axes): Axis to plot on

    Returns:
        dict: Statistics for the feature
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    default_rates = data.groupby(feature)[target_col].agg(["count", "mean"]).round(3)
    default_rates.columns = ["Count", "Default_Rate"]

    categories = default_rates.index
    rates = default_rates["Default_Rate"]
    counts = default_rates["Count"]

    bars = ax.bar(
        range(len(categories)),
        rates,
        color=[
            "lightcoral" if rate > data[target_col].mean() else "lightsteelblue"
            for rate in rates
        ],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (bar, rate, count) in enumerate(zip(bars, rates, counts)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{rate:.1%}\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    overall_rate = data[target_col].mean()
    ax.axhline(
        y=overall_rate,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Overall Rate: {overall_rate:.1%}",
    )

    ax.set_title(f"{feature.replace('_', ' ').title()}", fontweight="bold", fontsize=12)
    ax.set_ylabel("Default Rate")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    min_rate, max_rate = rates.min(), rates.max()
    range_diff = max_rate - min_rate

    return {
        "feature": feature,
        "categories": list(categories),
        "default_rates": list(rates),
        "counts": list(counts),
        "min_rate": min_rate,
        "max_rate": max_rate,
        "range_difference": range_diff,
    }


def create_comprehensive_bivariate_analysis(
    data, numerical_features, categorical_features, target_col="target"
):
    """
    Create comprehensive bivariate analysis with both numerical and categorical features

    Args:
        data (DataFrame): Input dataframe
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target_col (str): Target column name

    Returns:
        dict: Analysis results
    """
    print("BIVARIATE ANALYSIS")
    print("=" * 60)

    total_features = len(numerical_features) + len(categorical_features)
    n_cols = 3
    n_rows = int(np.ceil(total_features / n_cols))

    fig = plt.figure(figsize=(20, 8 * n_rows))

    plot_idx = 1
    numerical_stats = []
    categorical_stats = []

    print("NUMERICAL FEATURES - LINE DENSITY PLOTS")
    print("-" * 50)

    for feature in numerical_features:
        if feature in data.columns:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            stats_result = plot_numerical_feature_distribution(
                data, feature, target_col, ax
            )
            numerical_stats.append(stats_result)

            print(f"{feature.upper()}:")
            print(
                f"  No Default: mean={stats_result['no_default_mean']:.1f}, std={stats_result['no_default_std']:.1f}"
            )
            print(
                f"  Default:    mean={stats_result['default_mean']:.1f}, std={stats_result['default_std']:.1f}"
            )
            print(
                f"  Cohen's d:  {stats_result['cohens_d']:.3f} ({stats_result['effect_size']}) {stats_result['significance']}"
            )
            print()

            plot_idx += 1

    print("CATEGORICAL FEATURES - DEFAULT RATE BARS")
    print("-" * 50)

    for feature in categorical_features:
        if feature in data.columns:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            stats_result = plot_categorical_feature_distribution(
                data, feature, target_col, ax
            )
            categorical_stats.append(stats_result)

            # Print statistics
            print(f"{feature.upper()}:")
            for cat, rate, count in zip(
                stats_result["categories"],
                stats_result["default_rates"],
                stats_result["counts"],
            ):
                print(f"  {cat}: {rate:.1%} (n={count:,})")
            print(
                f"  Range: {stats_result['min_rate']:.1%} - {stats_result['max_rate']:.1%} "
                f"(diff: {stats_result['range_difference']:.1%})"
            )
            print()

            plot_idx += 1

    plt.tight_layout()
    plt.show()

    return {
        "numerical_stats": numerical_stats,
        "categorical_stats": categorical_stats,
        "overall_default_rate": data[target_col].mean(),
    }


def print_feature_importance_summary(analysis_results):
    """Print summary of most important features based on analysis"""

    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)

    numerical_stats = analysis_results["numerical_stats"]
    categorical_stats = analysis_results["categorical_stats"]

    numerical_sorted = sorted(
        numerical_stats, key=lambda x: abs(x["cohens_d"]), reverse=True
    )

    print("NUMERICAL FEATURES - RANKED BY EFFECT SIZE:")
    print("-" * 50)
    for i, stat in enumerate(numerical_sorted, 1):
        print(
            f"{i:2d}. {stat['feature']:20s}: Cohen's d = {stat['cohens_d']:6.3f} "
            f"({stat['effect_size']:6s}) {stat['significance']}"
        )

    categorical_sorted = sorted(
        categorical_stats, key=lambda x: x["range_difference"], reverse=True
    )

    print(f"\nCATEGORICAL FEATURES - RANKED BY DEFAULT RATE RANGE:")
    print("-" * 50)
    for i, stat in enumerate(categorical_sorted, 1):
        print(
            f"{i:2d}. {stat['feature']:20s}: {stat['min_rate']:.1%} - {stat['max_rate']:.1%} "
            f"(range: {stat['range_difference']:.1%})"
        )


def analyze_mutual_information(
    df,
    target_col='defaulted',
    feature_groups=None,
    binary_encoding=None,
    n_bins=20,
    balance_data=True,
    figsize=(18, 16),
    random_state=0
):
    """
    Compute and visualize mutual information matrix for features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default='defaulted'
        Name of target column
    feature_groups : dict, optional
        Dictionary with keys 'numerical', 'binary', 'categorical' containing lists of feature names
        If None, will auto-detect from dtypes
    binary_encoding : dict, optional
        Dictionary mapping binary feature names to (target_value, code) tuples
        Example: {'gender': ('M', 1), 'owns_car': ('Y', 1)}
    n_bins : int, default=20
        Number of bins for discretizing numerical features
    balance_data : bool, default=True
        Whether to balance classes by undersampling majority
    figsize : tuple, default=(18, 16)
        Figure size for heatmap
    random_state : int, default=0
        Random state for reproducibility
        
    Returns
    -------
    mi_df : pd.DataFrame
        Normalized mutual information matrix
    discretized_df : pd.DataFrame
        Discretized and balanced dataframe used for MI calculation
    """
    if feature_groups is None:
        feature_groups = {
            'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'binary': [],
            'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        for group in feature_groups.values():
            if target_col in group:
                group.remove(target_col)
    
    numerical_cols = feature_groups.get('numerical', [])
    binary_cols = feature_groups.get('binary', [])
    categorical_cols = feature_groups.get('categorical', [])
    
    if binary_encoding is None:
        binary_encoding = {}
        
    analysis_df = df.copy()
    
    if balance_data:
        df_majority = analysis_df[analysis_df[target_col] == 0]
        df_minority = analysis_df[analysis_df[target_col] == 1]
        df_majority_downsampled = resample(
            df_majority, replace=False, n_samples=len(df_minority), random_state=random_state
        )
        analysis_df = pd.concat([df_majority_downsampled, df_minority]).sample(
            frac=1, random_state=random_state
        )
    
    discretized_df = analysis_df.copy()
    
    for col in binary_cols:
        if col in binary_encoding:
            target_val, _ = binary_encoding[col]
            discretized_df[col] = (discretized_df[col] == target_val).astype(int)
        else:
            discretized_df[col] = discretized_df[col].astype('category').cat.codes
    
    for col in numerical_cols:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
        discretized_df[col] = kbd.fit_transform(discretized_df[[col]]).ravel().astype(int)
    
    for col in categorical_cols:
        discretized_df[col] = discretized_df[col].astype('category').cat.codes
    
    all_features = numerical_cols + binary_cols + categorical_cols + [target_col]
    
    n = len(all_features)
    mi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mi_matrix[i, j] = mutual_info_score(
                    discretized_df[all_features[i]], 
                    discretized_df[all_features[i]]
                )
            else:
                mi_score = mutual_info_score(
                    discretized_df[all_features[i]], 
                    discretized_df[all_features[j]]
                )
                mi_matrix[i, j] = mi_score
                mi_matrix[j, i] = mi_score
    
    mi_normalized = np.zeros_like(mi_matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                mi_normalized[i, j] = 1.0
            else:
                h_i = mi_matrix[i, i]
                h_j = mi_matrix[j, j]
                nmi = 2 * mi_matrix[i, j] / (h_i + h_j) if (h_i + h_j) > 0 else 0
                mi_normalized[i, j] = nmi
    
    mi_df = pd.DataFrame(mi_normalized, index=all_features, columns=all_features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        mi_df,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        square=True,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": .8, "label": "Normalized MI (NMI)"},
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    
    yticklabels = []
    for feat in mi_df.index:
        if feat == target_col:
            yticklabels.append(f"{feat} (TARGET)")
        elif feat in numerical_cols:
            yticklabels.append(f"{feat} (N)")
        elif feat in binary_cols:
            yticklabels.append(f"{feat} (B)")
        else:
            yticklabels.append(f"{feat} (C)")
    
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=9)
    ax.set_xticklabels(yticklabels, rotation=90, fontsize=9)
    
    for i, label in enumerate(ax.get_yticklabels()):
        feat = mi_df.index[i]
        if feat == target_col:
            label.set_color('red')
            label.set_weight('bold')
        elif feat in numerical_cols:
            label.set_color('blue')
        elif feat in binary_cols:
            label.set_color('green')
        else:
            label.set_color('orange')
    
    for i, label in enumerate(ax.get_xticklabels()):
        feat = mi_df.columns[i]
        if feat == target_col:
            label.set_color('red')
            label.set_weight('bold')
        elif feat in numerical_cols:
            label.set_color('blue')
        elif feat in binary_cols:
            label.set_color('green')
        else:
            label.set_color('orange')
    
    balance_text = "Balanced Data" if balance_data else "Original Data"
    plt.title(
        f'Mutual Information Matrix ({balance_text}, n_bins={n_bins})\n'
        f'TARGET=Red, N=Numerical, B=Binary, C=Categorical',
        fontsize=16, weight='bold', pad=20
    )
    plt.tight_layout()
    plt.show()
    
    print("SUMMARY STATISTICS")
    print("="*80)
    
    target_mi = mi_df[target_col].drop(target_col).sort_values(ascending=False)
    print(f"\nTop 10 features by MI with {target_col}:")
    print(f"{'Feature':<25} {'Type':<6} {'NMI':<10}")

    return mi_df, discretized_df
