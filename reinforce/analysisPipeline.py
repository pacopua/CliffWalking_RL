import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- Configuration for REINFORCE Analysis ---
CSV_FILEPATH = 'results/reinforce_param_sweep_resultsFiguras.csv' # INPUT: Path to your REINFORCE CSV
RESULTS_DIR = 'results_reinforce' # INPUT: Directory to save plots (use a new one)
import os
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# Target metric for Random Forest
# Option 1: Steps (fewer is better) - if 'avg_steps' represents steps to goal for successful runs
TARGET_METRIC_FOR_ANALYSIS = 'avg_steps'
# Option 2: Reward (higher is better) - if 'final_reward' is a good overall performance measure
# TARGET_METRIC_FOR_ANALYSIS = 'final_reward'
# Option 3: Success Rate (higher is better)
# TARGET_METRIC_FOR_ANALYSIS = 'success_rate'


# For 'avg_steps', only consider runs with at least this success rate
# (if 'avg_steps' includes non-successful runs, this filtering is important)
MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS = 0.1 # e.g., 10% - adjust as needed

# --- 1. Load and Basic Numeric Conversion ---
try:
    df = pd.read_csv(CSV_FILEPATH)
    print(f"Successfully loaded data from: {CSV_FILEPATH}")
    print(f"Original data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILEPATH}")
    exit()

# Rename columns to match some of the existing logic if desired, or use new names directly
# For REINFORCE, let's use its native names and adapt the feature_cols list later.
# df.rename(columns={'training_episodes': 'num_train_episodes',
#                    'learning_rate': 'alpha',
#                    'execution_time': 'mean_training_time_per_sample',
#                    'avg_steps': 'eval_mean_steps_if_successful', # If analogous
#                    'success_rate': 'eval_mean_success_rate'}, # If analogous
#           inplace=True)


# Convert potentially relevant columns to numeric
# List all columns that should be numeric from the REINFORCE CSV
reinforce_numeric_cols = [
    'gamma', 'training_episodes', 'finish_reward', 'fall_reward', 'step_reward',
    'learning_rate', 'learning_rate_decay',
    'avg_steps', 'std_steps', 'final_reward', 'final_loss', 'success_rate', 'execution_time'
]
for col in reinforce_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Expected column '{col}' not found in REINFORCE CSV.")

# --- 2. Define Reward Categories and Filter Data (Optional for REINFORCE, but can be useful) ---
# This part might be less critical if all REINFORCE runs used similar reward philosophies,
# but we can keep the structure.
def categorize_reward_reinforce(row):
    try:
        # Adapt this logic based on how REINFORCE rewards were structured/varied
        if row['finish_reward'] > 0 and row['step_reward'] < 0 and row['fall_reward'] <= -50:
            return 'Standard Goal-Oriented'
        elif row['finish_reward'] == 0 and row['step_reward'] == 0:
            return 'No Goal/Step Incentive'
        else:
            return 'Other Custom Rewards'
    except TypeError:
        return 'Undefined Rewards (due to NaN)'

reward_param_cols_reinforce = ['finish_reward', 'step_reward', 'fall_reward']
if all(col in df.columns for col in reward_param_cols_reinforce):
    df['reward_category'] = df.apply(categorize_reward_reinforce, axis=1)
    print("\nReward categories value counts (REINFORCE):")
    print(df['reward_category'].value_counts())

    df_main_analysis = df[df['reward_category'].isin(['Standard Goal-Oriented', 'Other Custom Rewards'])].copy()
    print(f"\nShape after selecting 'good' reward categories for main analysis: {df_main_analysis.shape}")
else:
    print("\nWarning: Reward parameter columns not found for categorization. Using full DataFrame.")
    df_main_analysis = df.copy()

# --- Clean Training Times (if there's an equivalent column like 'execution_time') ---
time_col_reinforce = 'execution_time'
if time_col_reinforce in df_main_analysis.columns:
    negative_time_mask = df_main_analysis[time_col_reinforce] < 0
    num_negative_times = negative_time_mask.sum()
    if num_negative_times > 0:
        print(f"\nWarning: Found {num_negative_times} runs with negative '{time_col_reinforce}'.")
        df_main_analysis = df_main_analysis[~negative_time_mask] # Drop these rows
        print(f"Dropped {num_negative_times} rows with negative execution times.")
        print(f"Shape of df_main_analysis after dropping negative time rows: {df_main_analysis.shape}")

# --- 3. Handle NaNs in the Target Metric for df_main_analysis ---
if TARGET_METRIC_FOR_ANALYSIS not in df_main_analysis.columns:
    print(f"Error: Target metric '{TARGET_METRIC_FOR_ANALYSIS}' not found in df_main_analysis. Exiting.")
    exit()

if TARGET_METRIC_FOR_ANALYSIS == 'avg_steps': # Analogous to 'eval_mean_steps_if_successful'
    # REINFORCE 'avg_steps' might be NaN if 'std_steps' is NaN (e.g. only 1 successful episode or no successes)
    # Or if success_rate is 0.
    # We also need a t_max equivalent if not directly in REINFORCE CSV. Assume a default or look for it.
    # For CliffWalking, t_max is usually 200.
    t_max_cliffwalking = 200 # Assume this if not in CSV
    
    if 'success_rate' in df_main_analysis.columns:
        condition_for_imputation = (df_main_analysis['success_rate'] < MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS) | \
                                   (df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].isnull())
        
        impute_value_steps = t_max_cliffwalking * 1.1 # Penalize for failure/low success
        
        df_main_analysis.loc[condition_for_imputation, TARGET_METRIC_FOR_ANALYSIS] = impute_value_steps
        print(f"\nFor '{TARGET_METRIC_FOR_ANALYSIS}': Imputed {condition_for_imputation.sum()} values (where success < {MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS*100}% or NaN) with {impute_value_steps:.1f}.")
    else:
        print(f"Warning: Cannot accurately impute '{TARGET_METRIC_FOR_ANALYSIS}' for REINFORCE as 'success_rate' column is missing. Dropping NaNs for this metric.")
        df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS], inplace=True)
elif TARGET_METRIC_FOR_ANALYSIS in ['final_reward', 'success_rate']: # For these, just drop NaNs if they exist
    df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS], inplace=True)

print(f"Shape of df_main_analysis after NaN handling for target metric: {df_main_analysis.shape}")


# --- 4. Visual Analysis (Skipping for this request, as focus is RF) ---
print("\nSkipping detailed visual analysis plots for REINFORCE as per request (focus on RF).")


# --- 5. Random Forest Analysis (on df_main_analysis) ---
if df_main_analysis.empty or (TARGET_METRIC_FOR_ANALYSIS in df_main_analysis.columns and df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].isnull().all()):
    print("\nNo valid data available for Random Forest analysis after cleaning and filtering.")
else:
    print(f"\n--- Starting Random Forest Analysis for REINFORCE (Target: {TARGET_METRIC_FOR_ANALYSIS}) ---")
    
    # Define features (hyperparameters from REINFORCE CSV)
    feature_cols_reinforce = [
        'gamma', 'training_episodes', 'finish_reward', 'fall_reward', 'step_reward',
        'learning_rate', 'learning_rate_decay'
        # 't_max' if it was varied and is in the CSV; otherwise, it's constant and not a useful feature
    ]
    if 't_max' in df_main_analysis.columns and df_main_analysis['t_max'].nunique() > 1:
         feature_cols_reinforce.append('t_max')

    existing_feature_cols = [col for col in feature_cols_reinforce if col in df_main_analysis.columns]
    
    if not existing_feature_cols:
        print("Error: No feature columns available for Random Forest.")
    elif TARGET_METRIC_FOR_ANALYSIS not in df_main_analysis.columns:
         print(f"Error: Target metric '{TARGET_METRIC_FOR_ANALYSIS}' not found in the dataframe for RF.")
    else:
        X = df_main_analysis[existing_feature_cols].copy()
        y = df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].copy()

        combined_for_rf = pd.concat([X, y], axis=1)
        combined_for_rf.dropna(inplace=True) 

        if combined_for_rf.empty:
            print("Error: Data became empty after final NaN drop for RF features/target.")
        else:
            X_rf = combined_for_rf[existing_feature_cols]
            y_rf = combined_for_rf[TARGET_METRIC_FOR_ANALYSIS]
            print(f"Training Random Forest with {len(X_rf)} samples for REINFORCE.")

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
            rf_model.fit(X_rf, y_rf)

            print(f"\nREINFORCE Random Forest OOB Score: {rf_model.oob_score_:.4f}")

            importances = rf_model.feature_importances_
            feature_names = X_rf.columns
            sorted_indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8)) # Adjusted size
            plt.title(f"REINFORCE Hyperparameter Importances for {TARGET_METRIC_FOR_ANALYSIS.replace('_', ' ').title()}")
            bars = plt.barh(range(X_rf.shape[1]), importances[sorted_indices], align='center') # Horizontal bar plot
            plt.yticks(range(X_rf.shape[1]), feature_names[sorted_indices])
            plt.xlabel("Importance")
            plt.ylabel("Hyperparameter")
            plt.gca().invert_yaxis() # Display most important at the top
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_rf_feature_importances_reinforce_{TARGET_METRIC_FOR_ANALYSIS}.png', bbox_inches='tight')
            plt.show()

            print("\nREINFORCE Feature Importances (descending):")
            for i in sorted_indices:
                print(f"  {feature_names[i]}: {importances[i]:.4f}")

# --- 6. Analysis of Top Performing Combinations (Skipping for this request) ---
print("\nSkipping top combinations analysis plot for REINFORCE as per request.")
            
print("\nREINFORCE analysis script complete.")