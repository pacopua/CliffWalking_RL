import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the data
df = pd.read_csv('fractional_results.csv')

# Create a directory for saving figures
import os
if not os.path.exists('figures'):
    os.makedirs('figures')

# Set the style for the plots
plt.style.use('ggplot')
sns.set_context("talk")

# 1. Overview of the distribution of key metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(y='finish_percent', data=df, ax=axes[0])
axes[0].set_title('Distribution of Finish Percentage')
axes[0].set_ylim(-0.05, 1.05)

sns.boxplot(y='avg_steps', data=df, ax=axes[1])
axes[1].set_title('Distribution of Average Steps')

sns.boxplot(y='avg_elapsed_time', data=df, ax=axes[2])
axes[2].set_title('Distribution of Average Elapsed Time')
axes[2].set_yscale('log')  # Log scale due to wide range

plt.tight_layout()
plt.savefig('figures/metrics_overview.png')
plt.close()

# 2. Analyze effect of gamma on key metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(x='gamma', y='finish_percent', data=df, ax=axes[0])
axes[0].set_title('Effect of Gamma on Finish Percentage')
axes[0].set_ylim(-0.05, 1.05)

sns.boxplot(x='gamma', y='avg_steps', data=df, ax=axes[1])
axes[1].set_title('Effect of Gamma on Average Steps')

sns.boxplot(x='gamma', y='avg_elapsed_time', data=df, ax=axes[2])
axes[2].set_title('Effect of Gamma on Average Elapsed Time')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig('figures/gamma_effects.png')
plt.close()

# 3. Analyze effect of epsilon on key metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(x='epsilon', y='finish_percent', data=df, ax=axes[0])
axes[0].set_title('Effect of Epsilon on Finish Percentage')
axes[0].set_ylim(-0.05, 1.05)

sns.boxplot(x='epsilon', y='avg_steps', data=df, ax=axes[1])
axes[1].set_title('Effect of Epsilon on Average Steps')

sns.boxplot(x='epsilon', y='avg_elapsed_time', data=df, ax=axes[2])
axes[2].set_title('Effect of Epsilon on Average Elapsed Time')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig('figures/epsilon_effects.png')
plt.close()

# 4. Analyze effect of number of trajectories on key metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(x='num_trajectories', y='finish_percent', data=df, ax=axes[0])
axes[0].set_title('Effect of Trajectories on Finish Percentage')
axes[0].set_ylim(-0.05, 1.05)

sns.boxplot(x='num_trajectories', y='avg_steps', data=df, ax=axes[1])
axes[1].set_title('Effect of Trajectories on Average Steps')

sns.boxplot(x='num_trajectories', y='avg_elapsed_time', data=df, ax=axes[2])
axes[2].set_title('Effect of Trajectories on Average Elapsed Time')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig('figures/trajectories_effects.png')
plt.close()

# 5. Analyze effect of reward structure
# Create a reward structure column for grouping
df['reward_structure'] = 'F:' + df['finish_reward'].astype(str) + ', FL:' + df['fall_reward'].astype(str) + ', S:' + df['step_reward'].astype(str)

# Get the top 10 most frequent reward structures
top_rewards = df['reward_structure'].value_counts().nlargest(10).index

# Filter data to include only top reward structures (for readability)
df_top_rewards = df[df['reward_structure'].isin(top_rewards)]

# Plot the effect of reward structure
plt.figure(figsize=(12, 8))
sns.boxplot(x='reward_structure', y='finish_percent', data=df_top_rewards)
plt.title('Effect of Reward Structure on Finish Percentage')
plt.xticks(rotation=45, ha='right')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('figures/reward_structure_finish.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(x='reward_structure', y='avg_steps', data=df_top_rewards)
plt.title('Effect of Reward Structure on Average Steps')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/reward_structure_steps.png')
plt.close()

# 6. Heatmap of finish percentage by gamma and epsilon
heatmap_data = df.pivot_table(
    values='finish_percent', 
    index='gamma', 
    columns='epsilon', 
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1)
plt.title('Average Finish Percentage by Gamma and Epsilon')
plt.tight_layout()
plt.savefig('figures/gamma_epsilon_heatmap.png')
plt.close()

# 7. Heatmap of average steps by gamma and epsilon
heatmap_steps = df.pivot_table(
    values='avg_steps', 
    index='gamma', 
    columns='epsilon', 
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green', 'yellow', 'red'])
sns.heatmap(heatmap_steps, annot=True, cmap=cmap, vmin=60, vmax=150)
plt.title('Average Steps by Gamma and Epsilon')
plt.tight_layout()
plt.savefig('figures/gamma_epsilon_steps_heatmap.png')
plt.close()

# 8. Analyze the relationship between average steps and finish percentage
plt.figure(figsize=(10, 8))
sns.scatterplot(x='avg_steps', y='finish_percent', hue='gamma', size='epsilon', 
                sizes=(50, 200), alpha=0.7, data=df)
plt.title('Relationship between Average Steps and Finish Percentage')
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/steps_vs_finish.png')
plt.close()

# 9. Analyze training time efficiency
plt.figure(figsize=(10, 8))
sns.scatterplot(x='avg_elapsed_time', y='finish_percent', hue='gamma', 
                size='num_trajectories', sizes=(50, 200), alpha=0.7, data=df)
plt.title('Training Time vs. Finish Percentage')
plt.xscale('log')
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/time_vs_finish.png')
plt.close()

# 10. Analyze the effect of fall reward on performance
plt.figure(figsize=(10, 8))
sns.boxplot(x='fall_reward', y='finish_percent', data=df)
plt.title('Effect of Fall Reward on Finish Percentage')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('figures/fall_reward_effect.png')
plt.close()

# 11. Analyze the effect of finish reward on performance
plt.figure(figsize=(10, 8))
sns.boxplot(x='finish_reward', y='finish_percent', data=df)
plt.title('Effect of Finish Reward on Finish Percentage')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('figures/finish_reward_effect.png')
plt.close()

# 12. Analyze the effect of step reward on performance
plt.figure(figsize=(10, 8))
sns.boxplot(x='step_reward', y='finish_percent', data=df)
plt.title('Effect of Step Reward on Finish Percentage')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('figures/step_reward_effect.png')
plt.close()

# 13. Create a pairplot for parameter interactions
# Select relevant columns for the pairplot
columns_for_pairplot = ['gamma', 'epsilon', 'finish_percent', 'avg_steps']
plt.figure(figsize=(12, 10))
sns.pairplot(df[columns_for_pairplot], hue='gamma', diag_kind='kde', corner=True)
plt.savefig('figures/parameter_pairplot.png')
plt.close()

# 14. Create radar charts for the best configurations
# Define a function to create radar charts
def create_radar_chart(data, title):
    # Define the parameters to plot on the radar chart (exclude metrics)
    params = ['gamma', 'num_trajectories', 'finish_reward', 'fall_reward', 'step_reward', 'epsilon']
    
    # Normalize the parameters to be between 0 and 1 for radar chart
    min_values = df[params].min()
    max_values = df[params].max()
    
    normalized_data = {}
    for param in params:
        # Avoid division by zero
        if max_values[param] == min_values[param]:
            normalized_data[param] = 1.0
        else:
            # Special handling for negative values (invert for fall_reward and step_reward)
            if param in ['fall_reward', 'step_reward'] and data[param] < 0:
                # For negative values, we want more negative to be closer to 0 on the radar
                range_val = abs(min_values[param] - max_values[param])
                if range_val > 0:
                    norm_val = abs(data[param] - min_values[param]) / range_val
                    normalized_data[param] = 1 - norm_val
                else:
                    normalized_data[param] = 0.5
            else:
                normalized_data[param] = (data[param] - min_values[param]) / (max_values[param] - min_values[param])
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Add parameter names
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params)
    
    # Plot the data
    values = [normalized_data[param] for param in params]
    values += values[:1]  # Close the polygon
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    plt.title(title, size=15, pad=20)
    plt.tight_layout()
    
    return fig

# Find the best configuration by finish_percent and then by avg_steps
best_configs = df.sort_values(['finish_percent', 'avg_steps'], ascending=[False, True]).head(3)

for i, (_, config) in enumerate(best_configs.iterrows()):
    fig = create_radar_chart(config, f"Best Configuration #{i+1}\nFinish: {config['finish_percent']:.2f}, Steps: {config['avg_steps']:.2f}")
    plt.savefig(f'figures/best_config_{i+1}_radar.png')
    plt.close(fig)

# 15. Create a correlation matrix to see relationships between all parameters
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Parameters and Metrics')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png')
plt.close()

# 16. Analyze the interaction between gamma and fall_reward
plt.figure(figsize=(12, 8))
sns.boxplot(x='gamma', y='finish_percent', hue='fall_reward', data=df)
plt.title('Interaction between Gamma and Fall Reward')
plt.ylim(-0.05, 1.05)
plt.legend(title='Fall Reward')
plt.tight_layout()
plt.savefig('figures/gamma_fall_reward_interaction.png')
plt.close()

# 17. Create a parallel coordinates plot for high-performing configurations
# Filter data for configurations with high finish percentage
high_perf = df[df['finish_percent'] >= 0.9].copy()

# Normalize all columns for parallel coordinates
for column in high_perf.columns:
    if high_perf[column].dtype in [np.int64, np.float64]:
        min_val = high_perf[column].min()
        max_val = high_perf[column].max()
        if max_val > min_val:
            high_perf[column] = (high_perf[column] - min_val) / (max_val - min_val)

# Plot parallel coordinates
plt.figure(figsize=(14, 8))
pd.plotting.parallel_coordinates(
    high_perf, 
    'gamma', 
    cols=['epsilon', 'finish_reward', 'fall_reward', 'step_reward', 'num_trajectories', 'avg_steps'],
    colormap=plt.cm.viridis
)
plt.title('Parallel Coordinates Plot of High-Performing Configurations')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.savefig('figures/parallel_coordinates.png')
plt.close()

# 18. Analyze the best parameter combinations by finish_percent
# Create a summary table of average finish_percent for different parameter combinations
summary_gamma_epsilon = df.groupby(['gamma', 'epsilon'])['finish_percent'].mean().reset_index()
summary_gamma_epsilon = summary_gamma_epsilon.pivot(index='gamma', columns='epsilon', values='finish_percent')

plt.figure(figsize=(10, 6))
sns.heatmap(summary_gamma_epsilon, annot=True, cmap='viridis', vmin=0, vmax=1)
plt.title('Average Finish Percentage by Gamma and Epsilon')
plt.tight_layout()
plt.savefig('figures/gamma_epsilon_finish_heatmap.png')
plt.close()

# 19. Create a plot to show how many trajectories affect training time for different gamma values
plt.figure(figsize=(12, 8))
for gamma in df['gamma'].unique():
    subset = df[df['gamma'] == gamma]
    sns.regplot(x='num_trajectories', y='avg_elapsed_time', data=subset, 
                label=f'gamma={gamma}', scatter_kws={'alpha': 0.5}, line_kws={'linewidth': 2})

plt.title('Trajectories vs. Training Time by Gamma')
plt.xlabel('Number of Trajectories')
plt.ylabel('Average Elapsed Time (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/trajectories_time_by_gamma.png')
plt.close()

# 20. Create a 3D plot to visualize the interaction between gamma, epsilon, and finish_percent
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot
scatter = ax.scatter(df['gamma'], df['epsilon'], df['finish_percent'],
                     c=df['avg_steps'], cmap='viridis', s=50, alpha=0.7)

ax.set_xlabel('Gamma')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Finish Percentage')
ax.set_title('3D Visualization of Gamma, Epsilon, and Finish Percentage')

# Add a color bar to show the avg_steps scale
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Average Steps')

plt.tight_layout()
plt.savefig('figures/3d_gamma_epsilon_finish.png')
plt.close()

print("Analysis completed! All figures saved to the 'figures' directory.")