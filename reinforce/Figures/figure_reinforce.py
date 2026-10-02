import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load the data
df = pd.read_csv('reinforce_param_sweep_results.csv')

# Clean the data - replace 'nan' strings with actual NaN values
df = df.replace('nan', np.nan)

# Convert numeric columns to appropriate types
numeric_cols = ['avg_steps', 'std_steps', 'final_reward', 'final_loss', 'success_rate', 'execution_time']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Create a composite score considering success rate, avg steps, and execution time
# Normalize each metric to 0-1 scale
df['norm_success'] = (df['success_rate'] - df['success_rate'].min()) / (df['success_rate'].max() - df['success_rate'].min())
df['norm_steps'] = 1 - (df['avg_steps'] - df['avg_steps'].min()) / (df['avg_steps'].max() - df['avg_steps'].min())
df['norm_time'] = 1 - (df['execution_time'] - df['execution_time'].min()) / (df['execution_time'].max() - df['execution_time'].min())

# Create a composite score (weighted average)
weights = {'success': 0.5, 'steps': 0.3, 'time': 0.2}  # Adjust weights as needed
df['composite_score'] = (weights['success'] * df['norm_success'] + 
                         weights['steps'] * df['norm_steps'] + 
                         weights['time'] * df['norm_time'])

# Get top 10 configurations
top_10 = df.sort_values('composite_score', ascending=False).head(10)

# Save top 10 to CSV
top_10.to_csv('top_10_configurations.csv', index=False)

# Create a horizontal bar chart for the top 10 configurations
plt.figure(figsize=(14, 8))

# Create configuration labels - reverse the order so best is at the top
reversed_top_10 = top_10.iloc[::-1]
config_labels = [
    f"γ={row['gamma']}, ep={int(row['training_episodes'])}, " +
    f"r_fin={int(row['finish_reward'])}, r_fall={int(row['fall_reward'])}, " +
    f"r_step={int(row['step_reward'])}, lr={row['learning_rate']}, " +
    f"decay={row['learning_rate_decay']}"
    for _, row in reversed_top_10.iterrows()
]

# Create the plot
y_pos = np.arange(len(reversed_top_10))
plt.barh(y_pos, reversed_top_10['composite_score'], align='center', alpha=0.8, 
         color=plt.cm.viridis(np.linspace(0.8, 0, len(reversed_top_10))))  # Reversed color order too

plt.yticks(y_pos, config_labels)
plt.xlabel('Composite Score')
plt.title('Top 10 Parameter Configurations (Best at Top)', fontsize=14)

# Add performance metrics as annotations
for i, (_, row) in enumerate(reversed_top_10.iterrows()):
    plt.text(0.01, i, f"Time: {row['execution_time']:.1f}s, Steps: {row['avg_steps']:.1f}, " +
             f"Success: {row['success_rate']*100:.0f}%", 
             va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('top_10_configurations.png', dpi=300, bbox_inches='tight')
plt.close()

# Display the top 10 configurations
print("\nTop 10 Parameter Configurations:")
print(top_10[['gamma', 'training_episodes', 'finish_reward', 'fall_reward', 
              'step_reward', 'learning_rate', 'learning_rate_decay', 'execution_time', 
              'avg_steps', 'success_rate', 'composite_score']].to_string())

# Create individual 3D plots
def save_3d_plot(x, y, z, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df[x], df[y], df[z], c=df[z], cmap='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=z)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_3d_plot('avg_steps', 'execution_time', 'success_rate', 
             'Configurations by Steps, Time and Success', '3d_steps_time_success.png')
save_3d_plot('gamma', 'learning_rate', 'success_rate', 
             'Key Parameters vs Success', '3d_gamma_lr_success.png')
save_3d_plot('training_episodes', 'finish_reward', 'success_rate', 
             'Training Setup vs Success', '3d_episodes_finish_success.png')

# Create individual boxplots for each hyperparameter
def save_boxplot(param, metric, title, filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=param, y=metric, data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Success rate boxplots
for param in ['gamma', 'training_episodes', 'finish_reward', 'fall_reward', 
              'step_reward', 'learning_rate', 'learning_rate_decay']:
    save_boxplot(param, 'success_rate', 
                f'Effect of {param} on Success Rate', 
                f'boxplot_{param}_success.png')

# Execution time boxplots
for param in ['gamma', 'training_episodes', 'finish_reward', 'fall_reward', 
              'step_reward', 'learning_rate', 'learning_rate_decay']:
    save_boxplot(param, 'execution_time', 
                f'Effect of {param} on Execution Time', 
                f'boxplot_{param}_time.png')

# Avg steps boxplots
for param in ['gamma', 'training_episodes', 'finish_reward', 'fall_reward', 
              'step_reward', 'learning_rate', 'learning_rate_decay']:
    save_boxplot(param, 'avg_steps', 
                f'Effect of {param} on Avg Steps', 
                f'boxplot_{param}_steps.png')
