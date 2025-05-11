import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for saving plots if it doesn't exist
os.makedirs('analysis_plots', exist_ok=True)

# Load the data
df = pd.read_csv('fractional_results.csv')

# Basic data cleaning
df['finish_percent'] = df['finish_percent'] * 100  # Convert to percentage
df['avg_elapsed_time'] = df['avg_elapsed_time'] / 60  # Convert to minutes

# Set up visualization style
plt.style.use('seaborn')
sns.set_palette("husl")

def save_figure(fig, filename):
    """Helper function to save figures"""
    fig.savefig(f'analysis_plots/{filename}.png', bbox_inches='tight')
    plt.close(fig)

def plot_performance_metrics():
    """Plot key performance metrics against different parameters"""
    # Success Rate by Gamma
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='gamma', y='finish_percent', data=df, ax=ax)
    ax.set_title('Success Rate by Gamma')
    ax.set_ylabel('Completion Percentage')
    save_figure(fig, 'success_rate_by_gamma')

    # Steps by Epsilon
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='epsilon', y='avg_steps', data=df, ax=ax)
    ax.set_title('Average Steps by Epsilon')
    ax.set_ylabel('Steps')
    save_figure(fig, 'steps_by_epsilon')

    # Reward structure impact
    df['reward_combo'] = df.apply(lambda row: f"F:{row['finish_reward']},Fl:{row['fall_reward']},S:{row['step_reward']}", axis=1)
    top_rewards = df['reward_combo'].value_counts().head(10).index
    reward_df = df[df['reward_combo'].isin(top_rewards)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    box = sns.boxplot(x='reward_combo', y='finish_percent', data=reward_df, ax=ax)
    ax.set_title('Success Rate by Reward Combination (Top 10)')
    ax.set_ylabel('Completion Percentage')
    box.set_xticklabels(box.get_xticklabels(), rotation=45, ha='right')
    save_figure(fig, 'success_rate_by_reward_combo')

    # Training time vs trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='num_trajectories', y='avg_elapsed_time', 
                   hue='finish_percent', size='finish_percent',
                   sizes=(20, 200), data=df, ax=ax)
    ax.set_title('Training Time vs. Trajectories (Colored by Success)')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_xlabel('Number of Trajectories')
    save_figure(fig, 'training_time_vs_trajectories')

    # Correlation heatmap
    best_combinations = df.sort_values('finish_percent', ascending=False).head(10)
    heatmap_data = best_combinations[['gamma', 'epsilon', 'num_trajectories', 
                                    'finish_reward', 'fall_reward', 'step_reward',
                                    'finish_percent', 'avg_steps']]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Top 10 Configurations Correlation Heatmap')
    save_figure(fig, 'correlation_heatmap')

    # Gamma vs Epsilon
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['gamma'], df['epsilon'], c=df['finish_percent'],
                        s=df['num_trajectories']/100, cmap='viridis')
    ax.grid(False)
    ax.set_title('Gamma vs Epsilon (Size=Trajectories, Color=Success)')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Epsilon')
    plt.colorbar(scatter, ax=ax, label='Completion Percentage')
    save_figure(fig, 'gamma_vs_epsilon')

def plot_parameter_interactions():
    """Visualize how parameters interact with each other"""
    numeric_cols = ['gamma', 'epsilon', 'num_trajectories', 'finish_percent']
    g = sns.PairGrid(df[numeric_cols], diag_sharey=False)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, warn_singular=False)
    g.map_diag(sns.histplot, kde=True)
    g.fig.suptitle('Parameter Interactions', y=1.02)
    save_figure(g.fig, 'parameter_interactions')

def plot_top_performers():
    """Highlight the top performing parameter combinations"""
    top_n = 15
    top_df = df.sort_values('finish_percent', ascending=False).head(top_n)
    top_df['config'] = top_df.apply(
        lambda row: f"γ={row['gamma']}, ε={row['epsilon']}, N={row['num_trajectories']}\n"
                   f"R=(fin:{row['finish_reward']}, fall:{row['fall_reward']}, step:{row['step_reward']})",
        axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='finish_percent', y='config', hue='config', 
                data=top_df, palette='viridis', dodge=False, legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Parameter Combinations by Success Rate')
    ax.set_xlabel('Completion Percentage')
    ax.set_ylabel('Parameter Combination')
    ax.set_xlim(90, 101)
    save_figure(fig, 'top_performers')

def plot_tradeoff_analysis():
    """Analyze tradeoffs between different metrics"""
    # Success vs Steps
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='avg_steps', y='finish_percent', 
                   hue='num_trajectories', size='gamma',
                   sizes=(20, 200), data=df, ax=ax)
    ax.set_title('Success Rate vs. Steps Taken')
    ax.set_xlabel('Average Steps')
    ax.set_ylabel('Completion Percentage')
    save_figure(fig, 'success_vs_steps')

    # Success vs Training Time
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='avg_elapsed_time', y='finish_percent', 
                   hue='epsilon', style='gamma',
                   data=df, ax=ax)
    ax.set_title('Success Rate vs. Training Time')
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Completion Percentage')
    save_figure(fig, 'success_vs_training_time')

def plot_reward_structure_impact():
    """Analyze how different reward structures affect performance"""
    df['reward_structure'] = df.apply(
        lambda row: f"Fin:{row['finish_reward']},Fl:{row['fall_reward']},St:{row['step_reward']}", 
        axis=1)
    
    reward_order = df.groupby('reward_structure')['finish_percent'].mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    box = sns.boxplot(x='reward_structure', y='finish_percent', data=df, order=reward_order, ax=ax)
    ax.set_title('Impact of Reward Structure on Success Rate')
    ax.set_xlabel('Reward Structure (Finish, Fall, Step)')
    ax.set_ylabel('Completion Percentage')
    box.set_xticklabels(box.get_xticklabels(), rotation=45, ha='right')
    save_figure(fig, 'reward_structure_impact')

# Generate all visualizations
plot_performance_metrics()
plot_parameter_interactions()
plot_top_performers()
plot_tradeoff_analysis()
plot_reward_structure_impact()

# Create summary CSV
best_configs = df.sort_values(['finish_percent', 'avg_steps'], ascending=[False, True]).head(10)
best_configs.to_csv('analysis_plots/best_configurations.csv', index=False)

print("All plots saved to 'analysis_plots' directory")
print("Top 5 Configurations:")
print(best_configs[['gamma', 'epsilon', 'num_trajectories', 
                    'finish_reward', 'fall_reward', 'step_reward',
                    'finish_percent', 'avg_steps', 'avg_elapsed_time']].head(5).to_string())