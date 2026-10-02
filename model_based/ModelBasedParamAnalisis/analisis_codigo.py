import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split # Only if you want to evaluate RF prediction accuracy
import os

# --- Configuration ---
CSV_FILEPATH = '../fractional_results.csv' # Updated to your CSV file
RESULTS_DIR = 'results' # Directory to save plots
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists

# Target metric for initial line plots and primary focus for efficiency plots
TARGET_METRIC_FOR_PLOTS = 'avg_steps' # Changed from 'eval_mean_steps_if_successful'

# For 'avg_steps', only consider runs with at least this success rate for certain interpretations,
# or impute a penalty if success is lower.
MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS = 0.1 # e.g., 10%

# Column name for success rate in your CSV
SUCCESS_METRIC_COL = 'finish_percent' # Changed from 'eval_mean_success_rate'

# --- 1. Load and Basic Numeric Conversion ---
try:
    df = pd.read_csv(CSV_FILEPATH)
    print(f"Successfully loaded data from: {CSV_FILEPATH}")
    print(f"Original data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILEPATH}")
    exit()

all_potential_numeric_cols = [
    'gamma', 'num_trajectories', 'epsilon',
    'finish_reward', 'fall_reward', 'step_reward',
    'avg_elapsed_time', 'std_training_time_per_sample',
    'avg_steps', 'eval_std_steps', 'finish_percent'
]
for col in all_potential_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Expected column '{col}' not found in CSV.")

# --- 2. Define Reward Categories and Filter Data ---
def categorize_reward(row):
    try:
        if row['finish_reward'] > 0 and row['step_reward'] < 0 and row['fall_reward'] <= -50:
            return 'Standard Goal-Oriented'
        elif row['finish_reward'] == 0 and row['step_reward'] == 0: # Handles cases like (0,0,X)
            return 'No Goal/Step Incentive'
        elif row['finish_reward'] == 0 and row['step_reward'] < 0: # Handles (0, negative, X)
             return 'Step Penalty Only' # Example of a more specific custom category
        else:
            return 'Other Custom Rewards'
    except TypeError:
        return 'Undefined Rewards (due to NaN)'

reward_param_cols = ['finish_reward', 'step_reward', 'fall_reward']
if all(col in df.columns for col in reward_param_cols):
    df['reward_category'] = df.apply(categorize_reward, axis=1)
    print("\nReward categories value counts:")
    print(df['reward_category'].value_counts())
    df_problematic_rewards = df[df['reward_category'].isin(['No Goal/Step Incentive', 'Step Penalty Only'])].copy()
    df_main_analysis = df[df['reward_category'].isin(['Standard Goal-Oriented', 'Other Custom Rewards'])].copy()
    print(f"\nShape after selecting 'good' reward categories for main analysis: {df_main_analysis.shape}")
else:
    print("\nWarning: One or more reward parameter columns not found. Using full DataFrame for analysis.")
    df_problematic_rewards = pd.DataFrame()
    df_main_analysis = df.copy()

# --- 3. Handle NaNs in the TARGET_METRIC_FOR_PLOTS for df_main_analysis ---
if TARGET_METRIC_FOR_PLOTS not in df_main_analysis.columns:
    print(f"Error: Target metric for plots '{TARGET_METRIC_FOR_PLOTS}' not found in df_main_analysis. Exiting.")
    exit()

if TARGET_METRIC_FOR_PLOTS == 'avg_steps':
    if SUCCESS_METRIC_COL in df_main_analysis.columns:
        IMPUTE_PENALTY_STEPS = 165.0
        condition_for_imputation = (df_main_analysis[SUCCESS_METRIC_COL] < MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS) | \
                                   (df_main_analysis[TARGET_METRIC_FOR_PLOTS].isnull())
        df_main_analysis.loc[condition_for_imputation, TARGET_METRIC_FOR_PLOTS] = IMPUTE_PENALTY_STEPS
        print(f"\nFor '{TARGET_METRIC_FOR_PLOTS}': Imputed {condition_for_imputation.sum()} values (where {SUCCESS_METRIC_COL} < {MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS*100}% or NaN) with {IMPUTE_PENALTY_STEPS:.1f}.")
    else:
        print(f"Warning: Cannot accurately impute '{TARGET_METRIC_FOR_PLOTS}'. '{SUCCESS_METRIC_COL}' missing. Dropping NaNs for this metric.")
        df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_PLOTS], inplace=True)
else:
    df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_PLOTS], inplace=True)

print(f"Shape of df_main_analysis after NaN handling for '{TARGET_METRIC_FOR_PLOTS}': {df_main_analysis.shape}")


# --- 4. Visual Analysis (on df_main_analysis, using TARGET_METRIC_FOR_PLOTS) ---
params_to_plot = ['epsilon', 'num_trajectories', 'gamma']
hue_params = {'epsilon': 'gamma', 'num_trajectories': 'epsilon', 'gamma': 'epsilon'}
lower_is_better_plots = True if TARGET_METRIC_FOR_PLOTS == 'avg_steps' else False # For y-axis inversion

for param_x in params_to_plot:
    if param_x not in df_main_analysis.columns:
        print(f"Skipping plot for '{param_x} vs {TARGET_METRIC_FOR_PLOTS}': '{param_x}' column not found.")
        continue
    plt.figure(figsize=(10, 6))
    plot_cols_to_check = [TARGET_METRIC_FOR_PLOTS, param_x]
    current_hue = hue_params.get(param_x)
    if current_hue and current_hue in df_main_analysis.columns:
        plot_cols_to_check.append(current_hue)
    elif current_hue:
        print(f"Hue parameter '{current_hue}' not found for plotting with '{param_x}'. Plotting without hue.")
        current_hue = None
    plot_df_param = df_main_analysis.dropna(subset=plot_cols_to_check)
    if plot_df_param.empty:
        print(f"Not enough data to plot {param_x} vs {TARGET_METRIC_FOR_PLOTS} after filtering NaNs.")
        plt.close()
        continue
    sns.lineplot(data=plot_df_param, x=param_x, y=TARGET_METRIC_FOR_PLOTS,
                 hue=current_hue, marker='o', err_style="band", errorbar=('ci', 95))
    if param_x == 'epsilon':
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
    else:
        plt.grid(True)
    plt.xlabel(f'{param_x.replace("_", " ").title()}')
    plt.ylabel(f'{TARGET_METRIC_FOR_PLOTS.replace("_", " ").title()}')
    plt.title(f'Effect of {param_x.replace("_", " ").title()} on {TARGET_METRIC_FOR_PLOTS.replace("_", " ").title()}')
    if current_hue: plt.legend(title=current_hue.replace("_", " ").title())
    if lower_is_better_plots: plt.gca().invert_yaxis()
    plt.savefig(f'{RESULTS_DIR}/plot_{param_x}_vs_{TARGET_METRIC_FOR_PLOTS}.png', bbox_inches='tight')
    plt.show()

# Plot for Reward Categories
if 'reward_category' in df.columns:
    if SUCCESS_METRIC_COL in df.columns:
        plt.figure(figsize=(12, 7))
        plot_df_reward_sr = df.dropna(subset=[SUCCESS_METRIC_COL, 'reward_category'])
        if not plot_df_reward_sr.empty:
            sns.boxplot(data=plot_df_reward_sr, x='reward_category', y=SUCCESS_METRIC_COL)
            plt.xlabel('Reward Category'); plt.ylabel(f'{SUCCESS_METRIC_COL.replace("_", " ").title()}')
            plt.title(f'Impact of Reward Structure on {SUCCESS_METRIC_COL.replace("_", " ").title()}')
            plt.xticks(rotation=25, ha='right'); plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_reward_structure_vs_{SUCCESS_METRIC_COL}.png', bbox_inches='tight')
            plt.show()
    if TARGET_METRIC_FOR_PLOTS in df.columns:
        plt.figure(figsize=(12, 7))
        plot_df_reward_target = df.dropna(subset=[TARGET_METRIC_FOR_PLOTS, 'reward_category'])
        if not plot_df_reward_target.empty :
            sns.boxplot(data=plot_df_reward_target, x='reward_category', y=TARGET_METRIC_FOR_PLOTS)
            plt.xlabel('Reward Category'); plt.ylabel(f'{TARGET_METRIC_FOR_PLOTS.replace("_", " ").title()}')
            plt.title(f'Impact of Reward Structure on {TARGET_METRIC_FOR_PLOTS.replace("_", " ").title()}')
            plt.xticks(rotation=25, ha='right')
            if lower_is_better_plots: plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_reward_structure_vs_{TARGET_METRIC_FOR_PLOTS}.png', bbox_inches='tight')
            plt.show()

# --- 5. Random Forest Analysis (on df_main_analysis) ---
# Define features (hyperparameters from your CSV)
feature_cols_rf = ['gamma', 'num_trajectories', 'epsilon',
                'finish_reward', 'fall_reward', 'step_reward']
existing_feature_cols_rf = [col for col in feature_cols_rf if col in df_main_analysis.columns]

if not existing_feature_cols_rf:
    print("\nError: No feature columns available for Random Forest.")
elif df_main_analysis.empty:
    print("\nNo valid data in df_main_analysis for Random Forest analysis.")
else:
    # List of target metrics for RF
    rf_target_metrics_list = ['avg_steps', 'avg_elapsed_time', SUCCESS_METRIC_COL]

    for current_rf_target in rf_target_metrics_list:
        print(f"\n--- Starting Random Forest Analysis (Target: {current_rf_target}) ---")

        if current_rf_target not in df_main_analysis.columns:
            print(f"Target metric '{current_rf_target}' not found in df_main_analysis. Skipping RF for this target.")
            continue

        # Prepare data for the current RF target from df_main_analysis
        # Select features (X) and current target (y), then drop rows with any NaNs in these specific columns
        cols_for_current_rf = existing_feature_cols_rf + [current_rf_target]
        temp_df_for_rf = df_main_analysis[cols_for_current_rf].copy()
        temp_df_for_rf.dropna(inplace=True)

        if temp_df_for_rf.empty:
            print(f"Data for RF became empty after NaN drop for target '{current_rf_target}'. Skipping.")
            continue
        
        if temp_df_for_rf[current_rf_target].nunique() < 2 :
             print(f"Warning: Target metric '{current_rf_target}' has {temp_df_for_rf[current_rf_target].nunique()} unique value(s) after filtering. RF requires at least 2. Skipping.")
             continue

        X_rf = temp_df_for_rf[existing_feature_cols_rf]
        y_rf = temp_df_for_rf[current_rf_target]

        print(f"Training Random Forest for '{current_rf_target}' with {len(X_rf)} samples.")

        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
            rf_model.fit(X_rf, y_rf)

            print(f"Random Forest OOB Score for '{current_rf_target}': {rf_model.oob_score_:.4f}")

            importances = rf_model.feature_importances_
            feature_names_for_plot = X_rf.columns
            sorted_indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 7))
            plt.title(f"Hyperparameter Importances for {current_rf_target.replace('_', ' ').title()}")
            plt.bar(range(X_rf.shape[1]), importances[sorted_indices], align='center')
            plt.xticks(range(X_rf.shape[1]), feature_names_for_plot[sorted_indices], rotation=45, ha="right")
            plt.ylabel("Importance")
            plt.xlabel("Hyperparameter")
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_rf_feature_importances_{current_rf_target}.png', bbox_inches='tight')
            plt.show()

            print(f"\nFeature Importances for '{current_rf_target}' (descending):")
            for i in sorted_indices:
                print(f"  {feature_names_for_plot[i]}: {importances[i]:.4f}")
        except Exception as e:
            print(f"Error during Random Forest for target '{current_rf_target}': {e}")


print("\nAnalysis complete.")