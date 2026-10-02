import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

# Load the data
df = pd.read_csv('../fractional_results.csv')

# Display basic statistics
print(df.describe())

# 1. Create a 3D visualization for parameter configurations vs avg_elapsed_time and avg_steps
# We'll use a bubble plot where color and size represent different dimensions

plt.figure(figsize=(14, 10))
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Normalize times for color mapping (using log scale since times vary widely)
norm = Normalize(vmin=np.log1p(df['avg_elapsed_time'].min()), 
                 vmax=np.log1p(df['avg_elapsed_time'].max()))
colors = cm.viridis(norm(np.log1p(df['avg_elapsed_time'])))

# Size based on finish percentage
sizes = df['finish_percent'] * 100

# Plot points
scatter = ax.scatter(df['gamma'], 
                     df['epsilon'], 
                     df['avg_steps'],
                     c=colors,
                     s=sizes,
                     alpha=0.7)

# Add labels
ax.set_xlabel('Gamma (Discount Factor)')
ax.set_ylabel('Epsilon (Exploration Rate)')
ax.set_zlabel('Average Steps')

# Add a color bar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), 
                    ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Log(Avg Time)')

plt.title('3D Visualization: Parameter Impact on Performance', fontsize=14)
plt.tight_layout()
plt.savefig('3d_parameter_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create boxplots for individual hyperparameters vs avg_time and avg_steps

# Function to create boxplots for each hyperparameter
def create_boxplots(df, param_name, target_cols):
    for target in target_cols:
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Sort the parameter values for more logical ordering
        param_values = sorted(df[param_name].unique())
        
        # Create boxplot
        ax = sns.boxplot(x=param_name, y=target, data=df, order=param_values)
        
        # Add swarmplot for individual data points
        sns.swarmplot(x=param_name, y=target, data=df, color='black', alpha=0.5, 
                      size=4, order=param_values)
        
        # Improve readability
        plt.xticks(rotation=45 if param_name != 'gamma' else 0)
        plt.title(f'Impact of {param_name} on {target}', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'boxplot_{param_name}_{target.replace("/", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

# List of parameters to analyze
parameters = ['gamma', 'num_trajectories', 'finish_reward', 'fall_reward', 
              'step_reward', 'epsilon']

# Target variables
targets = ['avg_elapsed_time', 'avg_steps']

# Create boxplots for each parameter
for param in parameters:
    create_boxplots(df, param, targets)

# 3. Create a ranking of the top 10 configurations

# First, we need to normalize both metrics to make them comparable
df['normalized_time'] = 1 - ((df['avg_elapsed_time'] - df['avg_elapsed_time'].min()) / 
                           (df['avg_elapsed_time'].max() - df['avg_elapsed_time'].min()))

# For steps, we want fewer steps, but only if the agent finishes
# If finish_percent is 0, we penalize it
df['normalized_steps'] = np.where(
    df['finish_percent'] > 0,
    1 - ((df['avg_steps'] - df['avg_steps'].min()) / 
         (df['avg_steps'].max() - df['avg_steps'].min())),
    0
)

# Create a combined score (equal weighting)
df['combined_score'] = 0.5 * df['normalized_time'] + 0.5 * df['normalized_steps']

# Get the top 10 configurations
top_10 = df.sort_values('combined_score', ascending=False).head(10)

# Display the top 10 configurations
print("\nTop 10 Parameter Configurations:")
print(top_10[['gamma', 'num_trajectories', 'finish_reward', 'fall_reward', 
              'step_reward', 'epsilon', 'avg_elapsed_time', 'avg_steps', 
              'finish_percent', 'combined_score']])

# Create a horizontal bar chart for the top 10 configurations
plt.figure(figsize=(14, 8))

# Create configuration labels - reverse the order so best is at the top
reversed_top_10 = top_10.iloc[::-1]
config_labels = [
    f"γ={row['gamma']}, n={int(row['num_trajectories'])}, " +
    f"r_f={int(row['finish_reward'])}, r_fall={int(row['fall_reward'])}, " +
    f"r_s={int(row['step_reward'])}, ε={row['epsilon']}"
    for _, row in reversed_top_10.iterrows()
]

# Create the plot
y_pos = np.arange(len(reversed_top_10))
plt.barh(y_pos, reversed_top_10['combined_score'], align='center', alpha=0.8, 
         color=plt.cm.viridis(np.linspace(0.8, 0, len(reversed_top_10))))  # Reversed color order too

plt.yticks(y_pos, config_labels)
plt.xlabel('Combined Score')
plt.title('Top 10 Parameter Configurations (Best at Top)', fontsize=14)

# Add time and steps as annotations
for i, (_, row) in enumerate(reversed_top_10.iterrows()):
    plt.text(0.01, i, f"Time: {row['avg_elapsed_time']:.2f}s, Steps: {row['avg_steps']:.1f}, " +
             f"Success: {row['finish_percent']*100:.0f}%", 
             va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('top_10_configurations.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a parallel coordinates plot to visualize all dimensions of the top configurations
plt.figure(figsize=(16, 8))

# Prepare the data
parallel_cols = ['gamma', 'num_trajectories', 'finish_reward', 'fall_reward', 
                'step_reward', 'epsilon', 'avg_elapsed_time', 'avg_steps', 'finish_percent']
parallel_data = top_10[parallel_cols].copy()

# Normalize all values for the parallel plot
for col in parallel_cols:
    if parallel_data[col].min() != parallel_data[col].max():
        parallel_data[col] = (parallel_data[col] - parallel_data[col].min()) / (
            parallel_data[col].max() - parallel_data[col].min())

# Create the parallel coordinates plot
pd.plotting.parallel_coordinates(
    parallel_data, 'finish_percent', colormap=plt.cm.viridis, 
    alpha=0.7, axvlines=True
)

plt.title('Parallel Coordinates Plot of Top 10 Configurations', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('parallel_coordinates_top10.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a heatmap showing correlation between parameters and performance metrics
plt.figure(figsize=(12, 10))
correlation = df[parameters + targets + ['finish_percent']].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Parameters vs. Performance', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create interactive 3D scatter plot for top configurations
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Use a different color for each configuration
scatter = ax.scatter(top_10['gamma'], 
                    top_10['epsilon'], 
                    top_10['avg_steps'],
                    c=top_10['avg_elapsed_time'],
                    s=top_10['finish_percent'] * 100,
                    alpha=0.8,
                    cmap='viridis')

# Add labels for each point
for i, row in top_10.iterrows():
    ax.text(row['gamma'], row['epsilon'], row['avg_steps'], 
            f"{i+1}", fontsize=12)

ax.set_xlabel('Gamma')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Avg Steps')

cbar = plt.colorbar(scatter)
cbar.set_label('Avg Time (seconds)')

plt.title('Top 10 Configurations: 3D View', fontsize=14)
plt.savefig('top10_3d_view.png', dpi=300, bbox_inches='tight')
plt.close()