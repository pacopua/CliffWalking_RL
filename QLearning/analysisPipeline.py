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

# --- Clean Training Times ---
if 'mean_training_time_per_sample' in df_main_analysis.columns:
    original_time_count = len(df_main_analysis)
    negative_time_mask = df_main_analysis['mean_training_time_per_sample'] < 0
    num_negative_times = negative_time_mask.sum()

    if num_negative_times > 0:
        print(f"\nWarning: Found {num_negative_times} runs with negative 'mean_training_time_per_sample'.")
        
        # Option A: Set to NaN (they will then be dropped or imputed by later steps if time is used)
        # df_main_analysis.loc[negative_time_mask, 'mean_training_time_per_sample'] = np.nan
        # print("Negative times have been set to NaN.")

        # Option B: Drop these rows entirely if they are few and you suspect other issues with them
        df_main_analysis = df_main_analysis[~negative_time_mask]
        print(f"Dropped {num_negative_times} rows with negative training times.")
        print(f"Shape of df_main_analysis after dropping negative time rows: {df_main_analysis.shape}")
    
    # Optional: Cap extremely high unrealistic times if any (e.g. > 3 * median or something)
    # This is more about outlier handling than fixing errors.
    # q_high = df_main_analysis['mean_training_time_per_sample'].quantile(0.99)
    # df_main_analysis.loc[df_main_analysis['mean_training_time_per_sample'] > q_high, 'mean_training_time_per_sample'] = q_high

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


# --- 6. Analysis of Top Performing Combinations ---
if df_main_analysis.empty or TARGET_METRIC_FOR_ANALYSIS not in df_main_analysis.columns:
    print("\nSkipping top combinations analysis: No valid data.")
else:
    print(f"\n--- Analyzing Top Performing Combinations (Target: {TARGET_METRIC_FOR_ANALYSIS}) ---")

    # Ensure 'mean_training_time_per_sample' exists and is numeric
    if 'mean_training_time_per_sample' not in df_main_analysis.columns:
        print("Warning: 'mean_training_time_per_sample' column not found. Cannot include in combined score.")
        # Proceed without time in score, or handle as error
        # For this example, we'll proceed, and time_weight will effectively be 0 if col is missing
        df_main_analysis['mean_training_time_per_sample'] = np.nan # Ensure column exists to avoid KeyError

    df_main_analysis['mean_training_time_per_sample'] = pd.to_numeric(df_main_analysis['mean_training_time_per_sample'], errors='coerce')

    # Create a copy for this specific analysis to avoid modifying df_main_analysis too much
    df_top_analysis = df_main_analysis.copy()

    # --- Define weights for the combined score ---
    # We want to MINIMIZE steps and MINIMIZE time.
    # If TARGET_METRIC_FOR_ANALYSIS is 'eval_mean_reward', then higher is better for reward.
    # We need to normalize them to be on a similar scale before combining.

    steps_col = 'eval_mean_steps_if_successful' # Assuming this is what you mean by "pasos hasta el destino"
    time_col = 'mean_training_time_per_sample'

    # Check if necessary columns exist
    if steps_col not in df_top_analysis.columns:
        print(f"Warning: '{steps_col}' not found. Cannot compute combined score based on steps.")
    elif time_col not in df_top_analysis.columns:
        print(f"Warning: '{time_col}' not found. Cannot compute combined score based on time.")
    else:
        # --- Normalization (Min-Max Scaling to [0, 1]) ---
        # For steps: lower is better. So, (max - x) / (max - min) will make higher scores better.
        # Or, more simply, scale so 0 is best, 1 is worst, then subtract from 1 for score.
        # For this combined score, let's make lower values better for both.
        
        # Filter out rows where steps or time is NaN for fair normalization and scoring
        df_top_analysis.dropna(subset=[steps_col, time_col], inplace=True)

        if not df_top_analysis.empty:
            min_steps = df_top_analysis[steps_col].min()
            max_steps = df_top_analysis[steps_col].max()
            if max_steps == min_steps: # Avoid division by zero if all values are the same
                 df_top_analysis['norm_steps'] = 0.0 if max_steps is not np.nan else np.nan
            else:
                df_top_analysis['norm_steps'] = (df_top_analysis[steps_col] - min_steps) / (max_steps - min_steps)

            min_time = df_top_analysis[time_col].min()
            max_time = df_top_analysis[time_col].max()
            if max_time == min_time:
                 df_top_analysis['norm_time'] = 0.0 if max_time is not np.nan else np.nan
            else:
                df_top_analysis['norm_time'] = (df_top_analysis[time_col] - min_time) / (max_time - min_time)

            # --- Combined Score (lower is better) ---
            # Ensure columns exist before trying to use them
            if 'norm_steps' in df_top_analysis.columns and 'norm_time' in df_top_analysis.columns:
                steps_weight = 0.5
                time_weight = 0.5
                df_top_analysis['combined_score'] = (steps_weight * df_top_analysis['norm_steps'] +
                                                     time_weight * df_top_analysis['norm_time'])
                
                # Sort by the combined score (ascending, as lower is better)
                top_n = 10
                df_top_n = df_top_analysis.sort_values(by='combined_score', ascending=True).head(top_n)

                print(f"\nTop {top_n} combinations based on combined score (steps and time):")
                # Select a few key hyperparameters to display along with the scores
                display_cols = ['gamma', 'alpha', 'epsilon', 'epsilon_decay', 'epsilon_end', 'num_train_episodes',
                                'step_reward', 'finish_reward', 'fall_reward',
                                steps_col, time_col, 'combined_score', 'eval_mean_success_rate']
                # Filter display_cols to only those present in df_top_n
                actual_display_cols = [col for col in display_cols if col in df_top_n.columns]
                print(df_top_n[actual_display_cols])

                # --- Plotting the Top N Combinations ---
                if not df_top_n.empty:
                    # Create labels for the y-axis (e.g., by concatenating some key params)
                    # Shorten for better display
                    df_top_n['label'] = df_top_n.apply(
                        lambda row: f"G:{row.get('gamma', 'N/A')}, A:{row.get('alpha', 'N/A')}, ED:{row.get('epsilon_decay', 'N/A')}\nEpsN:{row.get('num_train_episodes', 'N/A')}, SR:{row.get('step_reward', 'N/A')}",
                        axis=1
                    )
                    
                    fig, ax1 = plt.subplots(figsize=(14, 8)) # Increased figure size

                    # Bar plot for steps on primary y-axis
                    color_steps = 'tab:blue'
                    ax1.set_xlabel('Performance Metrics')
                    ax1.set_ylabel(steps_col.replace('_',' ').title(), color=color_steps)
                    bars_steps = ax1.barh(df_top_n['label'], df_top_n[steps_col], color=color_steps, alpha=0.7, label=steps_col.replace('_',' ').title())
                    ax1.tick_params(axis='y', labelcolor='black') # Keep y-axis labels black
                    ax1.invert_yaxis() # Display best (top of df) at the top of plot

                    # Add data labels for steps
                    for bar in bars_steps:
                        width = bar.get_width()
                        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                                 f'{width:.1f}', va='center', ha='left', color=color_steps)


                    # Instantiate a second y-axis for time
                    ax2 = ax1.twiny() # Share the same y-axis

                    color_time = 'tab:red'
                    # We need to plot time on the same horizontal axis direction.
                    # Since bars_steps are plotted from left to right, ax2 also plots from left to right.
                    # We can use a scatter plot or thin bars for time to differentiate.
                    # For this example, let's use scatter points aligned with the bars.
                    # ax2.set_ylabel(time_col.replace('_',' ').title(), color=color_time) # Not needed if using top x-axis
                    ax2.set_xlabel(time_col.replace('_',' ').title(), color=color_time)
                    
                    # Plot time as points or small bars on the secondary x-axis (top)
                    # To make them visually distinct and not overlapping too much,
                    # we can plot them as points.
                    points_time = ax2.scatter(df_top_n[time_col], df_top_n['label'], color=color_time, marker='o', s=100, label=time_col.replace('_',' ').title(), zorder=3)
                    
                    ax2.tick_params(axis='x', labelcolor=color_time)
                    ax2.grid(False) # Turn off grid for secondary axis if desired

                    fig.suptitle(f'Top {top_n} Hyperparameter Combinations (Lower Combined Score is Better)', fontsize=16)
                    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    
                    # Add a single legend for both
                    # handles, labels = [], []
                    # for ax in [ax1, ax2]:
                    #     h, l = ax.get_legend_handles_labels()
                    #     handles.extend(h)
                    #     labels.extend(l)
                    # if handles: # Only create legend if there are items
                    #    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                    # Simpler: just label axes
                    
                    plt.savefig(f'{RESULTS_DIR}/plot_top_{top_n}_combinations_score.png', bbox_inches='tight')
                    plt.show()
                else:
                    print("No top N combinations to plot (dataframe might be empty after sorting).")
            else:
                print("Normalized steps or time columns not found. Skipping combined score calculation.")
        else:
            print("DataFrame is empty after filtering NaNs for steps and time. Cannot calculate top combinations.")
            
print("\nAnalysis pipeline complete.")