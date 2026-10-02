import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

# Optimized Configuration
T_MAX = 200         # Max steps per episode
NUM_EPISODES = 2000 # Number of episodes
GAMMA = 0.99        # Discount factor
LEARNING_RATE = 0.1 # Learning rate
EPSILON_START = 0.5 # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.01# Decay rate per episode (larger value = faster decay)

class OptimizedQLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon_start, epsilon_end, epsilon_decay, t_max=None):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        # Pre-allocate Q-table with small random values to break ties initially
        self.Q = np.random.uniform(low=0, high=0.01, size=(self.n_states, self.n_actions))
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Epsilon decay parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.t_max = t_max
        
        # Pre-compute action space for random sampling
        self.action_space = np.arange(self.n_actions)
    
    def decay_epsilon(self):
        """More efficient epsilon decay using exponential formula"""
        self.epsilon = max(self.epsilon_end, self.epsilon * (1 - self.epsilon_decay))
    
    def select_action(self, state, training=True):
        """Optimized action selection with vectorized operations"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            # Vectorized argmax - handles ties automatically by taking the first max
            return np.argmax(self.Q[state])
    
    def update_Q(self, state, action, reward, next_state, is_done):
        """Optimized Q-value update"""
        # Vectorized max operation
        q_next_max = 0 if is_done else np.max(self.Q[next_state])
        
        # Q-learning update - combined into single formula
        self.Q[state, action] += self.learning_rate * (reward + self.gamma * q_next_max - self.Q[state, action])
    
    def learn_from_episode(self):
        """Run a single episode with optimized termination logic"""
        state, _ = self.env.reset()
        total_reward = 0
        
        for step in range(self.t_max):
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            total_reward += reward
            self.update_Q(state, action, reward, next_state, terminated)
            
            state = next_state
            
            if terminated or truncated:
                break
                
        return total_reward
    
    def policy(self):
        """Extract greedy policy from Q-values (vectorized)"""
        return np.argmax(self.Q, axis=1)


def draw_rewards(rewards, title="Rewards Over Episodes", display=True):
    """Plot rewards with optimized rendering"""
    window_size = 50
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    data['Rolling Average'] = data['Reward'].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Episode', y='Reward', data=data, alpha=0.6, label='Episode Reward')
    sns.lineplot(x='Episode', y='Rolling Average', data=data, label=f'Rolling Average (w={window_size})')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if display:
        plt.show()

def print_policy(policy):
    """Display policy in grid format for CliffWalking"""
    rows, cols = 4, 12
    visual_map = {0:'^', 1:'>', 2:'v', 3:'<', -1:'?'}
    policy_arrows = [visual_map.get(int(action), '?') for action in policy]
    
    print("Learned Policy:")
    try:
        policy_grid = np.array(policy_arrows).reshape(rows, cols)
        print(policy_grid)
    except ValueError as e:
        print(f"Error reshaping policy: {e}")

def run_training(with_timing=True):
    """Run training with timing measurements"""
    env = gym.make("CliffWalking-v0", is_slippery=True)
    
    start_time = time.time()
    
    # Initialize agent with optimized parameters
    agent = OptimizedQLearningAgent(
        env=env,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        t_max=T_MAX
    )
    
    rewards_history = np.zeros(NUM_EPISODES)
    
    # Training loop
    for i in range(NUM_EPISODES):
        rewards_history[i] = agent.learn_from_episode()
        agent.decay_epsilon()
        
        # Print progress less frequently to reduce overhead
        if (i + 1) % 500 == 0:
            print(f"Episode {i+1}/{NUM_EPISODES} - " +
                  f"Avg Reward (last 100): {np.mean(rewards_history[max(0, i-99):i+1]):.2f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    if with_timing:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Average time per episode: {training_time/NUM_EPISODES*1000:.2f} ms")
    
    # Only draw rewards and policy after training (not during)
    draw_rewards(rewards_history, title="Q-Learning Training Rewards (Optimized)")
    
    # Print final policy
    learned_policy = agent.policy()
    print_policy(learned_policy)
    
    return agent, rewards_history, training_time

def evaluate_agent(agent, num_eval_episodes=50):
    """Evaluate trained agent"""
    eval_env = gym.make("CliffWalking-v0", is_slippery=True)
    eval_rewards = np.zeros(num_eval_episodes)
    
    for n_ep in range(num_eval_episodes):
        state, _ = eval_env.reset()
        total_reward = 0
        done = False
        
        for step in range(T_MAX):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
                
        eval_rewards[n_ep] = total_reward
    
    eval_env.close()
    
    print(f"\nEvaluation Results:")
    print(f"Average reward over {num_eval_episodes} episodes: {np.mean(eval_rewards):.2f}")

    draw_rewards(eval_rewards, title=f"Q-Learning Evaluation Rewards ({num_eval_episodes} episodes)")
    
    return eval_rewards

# Run optimized training
print("Starting Optimized Training...")
trained_agent, rewards, training_time = run_training()

# Evaluate the trained agent
eval_rewards = evaluate_agent(trained_agent)

# Show Q-values for start state
print("\nQ-values for start state (36):")
print(f"Actions (0:^, 1:>, 2:v, 3:<): {trained_agent.Q[36]}")

# Performance comparison (if running against original)
print(f"\nOptimized training time for {NUM_EPISODES} episodes: {training_time:.2f} seconds")