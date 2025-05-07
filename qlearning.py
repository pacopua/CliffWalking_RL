# --- START OF ADAPTED Q-LEARNING FOR CLIFFWALKING ---

import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper # Keep Wrapper in case we need it later

# 1. Configuration and Dependencise
# Adjusted Hyperparameters for CliffWalking (can be tuned further)
# SLIPPERY = True # Standard CliffWalking doesn't support this flag. Using deterministic version.
T_MAX = 200         # Max steps per episode (CliffWalking can be long if agent wanders)
NUM_EPISODES = 4000 # Might need more episodes than FrozenLake
GAMMA = 0.99        # Discount factor, standard value
LEARNING_RATE = 0.1 # Learning rate, often needs tuning (0.1 is common)
EPSILON = 0.5       # Exploration rate, lower epsilon can be good if steps are costly

# Install dependencies (if not already installed)
# !pip install gymnasium seaborn numpy matplotlib

# 2. Environment Setup and Helper Functions

# Create the CliffWalking environment
# Note: Ignoring 'is_slippery=True' from PDF as standard env doesn't support it.
env = gym.make("CliffWalking-v0", is_slippery = True) # render_mode="human" for visualization during eval


# Observation space: 48 states (0-47)
# Action space: 4 actions (0: up, 1: right, 2: down, 3: left)

def draw_rewards(rewards, title="Rewards Over Episodes"):
    """Plots the rewards per episode and a rolling average."""
    window_size = 50 # Adjust window size for rolling average if needed
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
    plt.show()

def print_policy(policy):
    """Prints the policy in a grid format for CliffWalking."""
    rows, cols = 4, 12 # CliffWalking grid dimensions
    if len(policy) != rows * cols:
        print(f"Warning: Policy length ({len(policy)}) doesn't match grid dimensions ({rows}x{cols}).")
        # Attempt to reshape anyway or handle error
    # Action mapping for CliffWalking: 0: ^, 1: >, 2: v, 3: <
    visual_help = {0:'^', 1:'>', 2:'v', 3:'<', -1:'?'} # Add default for unexpected values
    policy_arrows = [visual_help.get(int(action), '?') for action in policy]
    try:
        print("Learned Policy:")
        print(np.array(policy_arrows).reshape(rows, cols))
    except ValueError as e:
         print(f"Error reshaping policy: {e}")
         print("Policy array:", policy_arrows)


# 3. Q-Learning Agent Implementation (Adapted from 04_qlearning.ipynb)

class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max=None): # t_max optional now
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max # Store t_max if provided, though learn_from_episode handles termination

    def select_action(self, state, training=True):
        """Selects an action using epsilon-greedy strategy."""
        if training and random.random() <= self.epsilon:
            return self.env.action_space.sample() # Use random action
        else:
            # Handle ties by choosing randomly among the best actions
            best_actions = np.flatnonzero(self.Q[state,] == np.max(self.Q[state,]))
            return np.random.choice(best_actions)

    def update_Q(self, state, action, reward, next_state, is_done):
        """Updates the Q-value for the state-action pair using the Q-learning rule."""
        if is_done:
             q_next_max = 0 # Terminal state has no future reward
        else:
             q_next_max = np.max(self.Q[next_state,]) # Value of the best action from next state

        # Q-learning update rule
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def learn_from_episode(self):
        """Runs a single episode, updating Q-values."""
        state, _ = self.env.reset()
        total_reward = 0
        is_done = False
        truncated = False
        steps = 0

        # Run until episode ends naturally or max steps reached (if t_max is set)
        while not is_done and not truncated:
            if self.t_max is not None and steps >= self.t_max:
                truncated = True # Force truncation if max steps exceeded

            action = self.select_action(state, training=True)
            new_state, new_reward, is_done, truncated_step, info = self.env.step(action)
            # Note: Gymnasium returns truncated separately, combine flags for termination check
            terminated = is_done or truncated_step or truncated

            total_reward += new_reward
            self.update_Q(state, action, new_reward, new_state, is_done) # Pass is_done for correct target calculation

            state = new_state
            steps += 1

            if terminated: # Break loop if episode finished (done or truncated)
                break

        return total_reward

    def policy(self):
        """Extracts the greedy policy from the learned Q-values."""
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
             # Use select_action in non-training mode to get the best action
             policy[s] = self.select_action(s, training=False)
        return policy

# 4. Training Process

print("Starting Training...")
agent = QLearningAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=T_MAX)
rewards_history = []

for i in range(NUM_EPISODES):
    episode_reward = agent.learn_from_episode()
    rewards_history.append(episode_reward)
    agent.epsilon *= 0.999999 # Decay epsilon (optional, can be tuned)
    if (i + 1) % 100 == 0: # Print progress
        print(f"Episode {i+1}/{NUM_EPISODES} - Avg Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")

print("Training finished.")

# Visualize training rewards
draw_rewards(rewards_history, title="Q-Learning Training Rewards on CliffWalking-v0")

# Print the learned policy
learned_policy = agent.policy()
print_policy(learned_policy)


# 5. Evaluation Process

print("\nStarting Evaluation...")
EVAL_EPISODES = 50
eval_rewards = []

eval_env = gym.make("CliffWalking-v0", is_slippery=True) # Use  , render_mode="human" to watch

for n_ep in range(EVAL_EPISODES):
    state, _ = eval_env.reset()
    total_reward = 0
    is_done = False
    truncated = False
    steps = 0
    while not is_done and not truncated:
        if T_MAX is not None and steps >= T_MAX: # Use T_MAX for safety during eval too
             truncated = True
        action = agent.select_action(state, training=False) # Use greedy policy
        state, reward, is_done, _, _ = eval_env.step(action)
        total_reward += reward
        steps += 1
        # eval_env.render() # Uncomment to watch the agent
        if is_done or truncated:
            break
    eval_rewards.append(total_reward)
    # print(f"Evaluation Episode {n_ep+1}: Reward = {total_reward}") # Optional: print reward per episode

eval_env.close()
print("Evaluation finished.")
print(f"Average reward over {EVAL_EPISODES} evaluation episodes: {np.mean(eval_rewards):.2f}")

# Visualize evaluation rewards
draw_rewards(eval_rewards, title=f"Q-Learning Evaluation Rewards ({EVAL_EPISODES} episodes)")


# Optional: Look at the Q-table for a specific state (e.g., start state 36)
print("\nQ-values for start state (36):")
print(f"Actions (0:^, 1:>, 2:v, 3:<): {agent.Q[36]}")

# --- END OF ADAPTED Q-LEARNING FOR CLIFFWALKING ---


