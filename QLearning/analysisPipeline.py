import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split # Only if you want to evaluate RF prediction accuracy

# --- Configuration ---
CSV_FILEPATH = 'results/training_results_with_evaluation.csv' # Make sure this is correct
RESULTS_DIR = 'results' # Directory to save plots

# Target metric for Random Forest and primary focus for efficiency plots
# Option 1: Steps (fewer is better) - use this if success rate is mostly high
TARGET_METRIC_FOR_ANALYSIS = 'eval_mean_steps_if_successful'
# Option 2: Reward (higher is better) - good general metric
# TARGET_METRIC_FOR_ANALYSIS = 'eval_mean_reward'

# For 'eval_mean_steps_if_successful', only consider runs with at least this success rate
MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS = 0.1 # e.g., 10% - adjust as needed based on your data

# --- 1. Load and Basic Numeric Conversion ---
try:
    df = pd.read_csv(CSV_FILEPATH)
    print(f"Successfully loaded data from: {CSV_FILEPATH}")
    print(f"Original data shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILEPATH}")
    exit()

# Convert potentially relevant columns to numeric, coercing errors
# This list should include all hyperparameters and all performance metrics
all_potential_numeric_cols = [
    'gamma', 'alpha', 'epsilon', 'epsilon_decay', 'epsilon_end',
    'num_train_episodes', 't_max',
    'finish_reward', 'fall_reward', 'step_reward', # These define the reward structure
    'eval_mean_reward', 'eval_std_reward',
    'eval_mean_success_rate', 'eval_std_success_rate',
    'eval_mean_steps_if_successful', 'eval_std_steps_if_successful',
    'samples_per_param_set' # Though likely constant for all rows in one CSV
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
        elif row['finish_reward'] == 0 and row['step_reward'] == 0:
            return 'No Goal/Step Incentive'
        # Add more specific categories if your reward parameters vary more
        # For example, to distinguish between different levels of positive finish_reward:
        # elif row['finish_reward'] == 100 and row['step_reward'] < 0 and row['fall_reward'] <= -50:
        #     return 'High Finish Reward'
        else:
            return 'Other Custom Rewards'
    except TypeError: # Handles if any reward component is NaN after coercion
        return 'Undefined Rewards (due to NaN)'

reward_param_cols = ['finish_reward', 'step_reward', 'fall_reward']
if all(col in df.columns for col in reward_param_cols):
    df['reward_category'] = df.apply(categorize_reward, axis=1)
    print("\nReward categories value counts:")
    print(df['reward_category'].value_counts())

    # Data for discussing problematic rewards (e.g., 0% success)
    df_problematic_rewards = df[df['reward_category'] == 'No Goal/Step Incentive'].copy()

    # Main dataset for performance/efficiency analysis (focus on 'good' reward structures)
    # Adjust this list based on your reward categories and what you consider "good for learning"
    df_main_analysis = df[df['reward_category'].isin(['Standard Goal-Oriented', 'Other Custom Rewards'])].copy()
    # If 'Other Custom Rewards' also includes problematic ones, be more specific:
    # df_main_analysis = df[df['reward_category'] == 'Standard Goal-Oriented'].copy()
    
    print(f"\nShape after selecting 'good' reward categories for main analysis: {df_main_analysis.shape}")
else:
    print("\nWarning: One or more reward parameter columns (finish_reward, fall_reward, step_reward) not found. Using full DataFrame for analysis.")
    df_problematic_rewards = pd.DataFrame()
    df_main_analysis = df.copy()

# --- 3. Handle NaNs in the Target Metric for df_main_analysis ---
if TARGET_METRIC_FOR_ANALYSIS not in df_main_analysis.columns:
    print(f"Error: Target metric '{TARGET_METRIC_FOR_ANALYSIS}' not found in df_main_analysis. Exiting.")
    exit()

if TARGET_METRIC_FOR_ANALYSIS == 'eval_mean_steps_if_successful':
    if 'eval_mean_success_rate' in df_main_analysis.columns and 't_max' in df_main_analysis.columns:
        condition_for_imputation = (df_main_analysis['eval_mean_success_rate'] < MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS) | \
                                   (df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].isnull())
        
        default_t_max = df_main_analysis['t_max'].median() if not df_main_analysis['t_max'].empty else 200
        impute_value_steps = default_t_max * 1.1 # Penalize slightly more than just timing out
        
        df_main_analysis.loc[condition_for_imputation, TARGET_METRIC_FOR_ANALYSIS] = impute_value_steps
        print(f"\nFor '{TARGET_METRIC_FOR_ANALYSIS}': Imputed {condition_for_imputation.sum()} values (where success < {MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS*100}% or NaN) with {impute_value_steps:.1f}.")
    else:
        print(f"Warning: Cannot accurately impute '{TARGET_METRIC_FOR_ANALYSIS}'. 'eval_mean_success_rate' or 't_max' missing. Dropping NaNs for this metric.")
        df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS], inplace=True)
else: # For other target metrics like 'eval_mean_reward'
    df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS], inplace=True)

print(f"Shape of df_main_analysis after NaN handling for target metric: {df_main_analysis.shape}")


# --- 4. Visual Analysis (on df_main_analysis) ---
# Define hyperparameters to plot against the target metric
params_to_plot = ['epsilon_decay', 'alpha', 'num_train_episodes', 'gamma', 'epsilon']
hue_params = {'epsilon_decay': 'epsilon', 'alpha': 'gamma', 'num_train_episodes': None, 'gamma': 'alpha', 'epsilon': 'epsilon_decay'}

# Determine if lower is better for the target metric (for inverting y-axis)
lower_is_better = True if TARGET_METRIC_FOR_ANALYSIS == 'eval_mean_steps_if_successful' else False

for param_x in params_to_plot:
    if param_x not in df_main_analysis.columns:
        print(f"Skipping plot for '{param_x} vs {TARGET_METRIC_FOR_ANALYSIS}': '{param_x}' column not found.")
        continue

    plt.figure(figsize=(10, 6))
    plot_df_param = df_main_analysis.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS, param_x])

    if plot_df_param.empty:
        print(f"Not enough data to plot {param_x} vs {TARGET_METRIC_FOR_ANALYSIS} after filtering NaNs.")
        plt.close() # Close empty figure
        continue

    current_hue = hue_params.get(param_x)
    if current_hue and current_hue not in df_main_analysis.columns:
        print(f"Hue parameter '{current_hue}' not found for plotting with '{param_x}'. Plotting without hue.")
        current_hue = None

    sns.lineplot(data=plot_df_param, x=param_x, y=TARGET_METRIC_FOR_ANALYSIS,
                 hue=current_hue,
                 marker='o', err_style="band", errorbar=('ci', 95))
    
    if param_x == 'epsilon_decay':
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
    else:
        plt.grid(True)

    plt.xlabel(f'{param_x.replace("_", " ").title()}')
    plt.ylabel(f'{TARGET_METRIC_FOR_ANALYSIS.replace("_", " ").title()}')
    plt.title(f'Effect of {param_x.replace("_", " ").title()} on {TARGET_METRIC_FOR_ANALYSIS.replace("_", " ").title()}')
    if current_hue:
        plt.legend(title=current_hue.replace("_", " ").title())
    
    if lower_is_better:
        plt.gca().invert_yaxis()

    plt.savefig(f'{RESULTS_DIR}/plot_{param_x}_vs_{TARGET_METRIC_FOR_ANALYSIS}.png', bbox_inches='tight')
    plt.show()


# Plot for Reward Categories (Success Rate and Target Metric)
if 'reward_category' in df.columns: # Use original df for this to show all categories
    # Plot Success Rate by Reward Category
    plt.figure(figsize=(12, 7))
    plot_df_reward_sr = df.dropna(subset=['eval_mean_success_rate', 'reward_category'])
    if not plot_df_reward_sr.empty:
        sns.boxplot(data=plot_df_reward_sr, x='reward_category', y='eval_mean_success_rate')
        plt.xlabel('Reward Category')
        plt.ylabel('Mean Success Rate (Eval)')
        plt.title('Impact of Reward Structure on Success Rate')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/plot_reward_structure_vs_success_rate.png', bbox_inches='tight')
        plt.show()

    # Plot Target Metric by Reward Category (using df_main_analysis or full df as appropriate)
    # If target is steps, it's already filtered. If reward, maybe use full df or a specific filter.
    # For this example, using the original df to see all categories, but be mindful of NaNs if steps is target
    if TARGET_METRIC_FOR_ANALYSIS in df.columns:
        plt.figure(figsize=(12, 7))
        plot_df_reward_target = df.dropna(subset=[TARGET_METRIC_FOR_ANALYSIS, 'reward_category'])
        if not plot_df_reward_target.empty :
            sns.boxplot(data=plot_df_reward_target, x='reward_category', y=TARGET_METRIC_FOR_ANALYSIS)
            plt.xlabel('Reward Category')
            plt.ylabel(f'{TARGET_METRIC_FOR_ANALYSIS.replace("_", " ").title()}')
            plt.title(f'Impact of Reward Structure on {TARGET_METRIC_FOR_ANALYSIS.replace("_", " ").title()}')
            plt.xticks(rotation=25, ha='right')
            if lower_is_better:
                plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_reward_structure_vs_{TARGET_METRIC_FOR_ANALYSIS}.png', bbox_inches='tight')
            plt.show()


# --- 5. Random Forest Analysis (on df_main_analysis) ---
if df_main_analysis.empty or df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].isnull().all():
    print("\nNo valid data available for Random Forest analysis after cleaning and filtering.")
else:
    print(f"\n--- Starting Random Forest Analysis (Target: {TARGET_METRIC_FOR_ANALYSIS}) ---")
    
    # Define features (hyperparameters)
    # Ensure these are the columns you actually varied and want to assess
    feature_cols = ['gamma', 'alpha', 'epsilon', 'epsilon_decay', 'epsilon_end', 
                    'num_train_episodes', 
                    'finish_reward', 'fall_reward', 'step_reward', # Reward params are features too!
                    't_max']
    
    # Filter to only existing columns in df_main_analysis
    existing_feature_cols = [col for col in feature_cols if col in df_main_analysis.columns]
    
    if not existing_feature_cols:
        print("Error: No feature columns available for Random Forest.")
    else:
        X = df_main_analysis[existing_feature_cols].copy()
        y = df_main_analysis[TARGET_METRIC_FOR_ANALYSIS].copy()

        # Final check for NaNs that might have been introduced or missed
        # (e.g. if a feature column had NaNs after numeric conversion)
        combined_for_rf = pd.concat([X, y], axis=1)
        combined_for_rf.dropna(inplace=True) # Drop rows with any NaNs in features or target

        if combined_for_rf.empty:
            print("Error: Data became empty after final NaN drop for RF features/target.")
        else:
            X_rf = combined_for_rf[existing_feature_cols]
            y_rf = combined_for_rf[TARGET_METRIC_FOR_ANALYSIS]
            print(f"Training Random Forest with {len(X_rf)} samples.")

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
            rf_model.fit(X_rf, y_rf)

            print(f"\nRandom Forest OOB Score: {rf_model.oob_score_:.4f}")

            importances = rf_model.feature_importances_
            feature_names = X_rf.columns
            sorted_indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 7))
            plt.title(f"Hyperparameter Importances for {TARGET_METRIC_FOR_ANALYSIS.replace('_', ' ').title()}")
            plt.bar(range(X_rf.shape[1]), importances[sorted_indices], align='center')
            plt.xticks(range(X_rf.shape[1]), feature_names[sorted_indices], rotation=45, ha="right")
            plt.ylabel("Importance")
            plt.xlabel("Hyperparameter")
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/plot_rf_feature_importances_{TARGET_METRIC_FOR_ANALYSIS}.png', bbox_inches='tight')
            plt.show()

            print("\nFeature Importances (descending):")
            for i in sorted_indices:
                print(f"  {feature_names[i]}: {importances[i]:.4f}")

print("\nAnalysis complete.")