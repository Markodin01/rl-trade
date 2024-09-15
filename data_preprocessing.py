import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timeit
def load_data():
    df_list = []
    for i in tqdm(range(1, 5), desc="Loading data"):
        df = pd.read_parquet(f'data/batch{i}.parquet')
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

from datetime import datetime

@timeit
def preprocess_data(df):
    # Convert timestamp from seconds to datetime string
    df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    
    # Convert the string back to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Drop the __index_level_0__ column if it exists
    df = df.drop(columns=['__index_level_0__'], errors='ignore')
    
    # Calculate 'next_return' if it doesn't exist
    if 'next_return' not in df.columns:
        df['next_return'] = df['close'].pct_change().shift(-1)
    
    df.dropna(inplace=True)
    
    return df

@timeit
def stratified_time_series_sample(df, sample_size, n_consecutive=180):
    if 'market_condition' not in df.columns:
        df['market_condition'] = pd.qcut(df['next_return'], q=3, labels=['bearish', 'neutral', 'bullish'])
    
    sampled_df = pd.DataFrame()
    for condition in df['market_condition'].unique():
        condition_df = df[df['market_condition'] == condition]
        n_samples = sample_size // 3 // n_consecutive
        
        start_indices = np.random.choice(len(condition_df) - n_consecutive, n_samples, replace=False)
        
        for start in start_indices:
            sampled_df = pd.concat([sampled_df, condition_df.iloc[start:start+n_consecutive]])
    
    return sampled_df.sort_index()

@timeit
def iterative_feature_importance(df, n_iterations=5, sample_size=180000):
    # Create 'market_condition' column
    df['market_condition'] = pd.qcut(df['next_return'], q=3, labels=['bearish', 'neutral', 'bullish'])
    
    features = df.columns.drop(['next_return', 'target', 'market_condition'])
    importance_scores = pd.DataFrame(index=features)
    
    for i in tqdm(range(n_iterations), desc="Calculating feature importance"):
        sample = stratified_time_series_sample(df, sample_size)
        X = sample[features]
        y = sample['next_return']
        
        mi_scores = calculate_feature_importance(X, y)
        importance_scores[f'iteration_{i}'] = mi_scores
    
    mean_importance = importance_scores.mean(axis=1).sort_values(ascending=False)
    return mean_importance

@timeit
def calculate_feature_importance(X, y):
    return mutual_info_regression(X, y)

@timeit
def plot_feature_importance(mi_scores):
    plt.figure(figsize=(12, 20))
    mi_scores.plot.barh()
    plt.title("Feature Importance based on Mutual Information")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

@timeit
def plot_correlation_heatmap(X):
    corr_matrix = X.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    start_time = time.time()

    print("Loading all data...")
    df = load_data()

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Calculating feature importance iteratively...")
    mi_scores = iterative_feature_importance(df, n_iterations=5, sample_size=180000)
    print("Plotting feature importance...")
    plot_feature_importance(mi_scores)

    print("Top 20 Important Features:")
    print(mi_scores.head(20))

    print("Plotting correlation heatmap for top 30 features...")
    top_30_features = mi_scores.head(30).index.tolist()
    plot_correlation_heatmap(df[top_30_features])

    print("Selecting top features and saving...")
    top_features = mi_scores.head(30).index.tolist()
    if 'close' not in top_features:
        top_features.append('close')
    df_selected = df[top_features + ['next_return', 'target']]
    df_selected.to_csv('selected_features.csv')

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print("Data preprocessing and feature selection completed.")
    print(f"Selected {len(top_features)} features.")
    print("Selected features dataframe saved as 'selected_features.csv'")

    # Optional: Print feature importances
    print("\nFeature Importances:")
    for feature, importance in mi_scores.items():
        print(f"{feature}: {importance:.4f}")