import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time
import csv
from scipy.special import softmax

# Hyperparameters with better defaults based on our analysis
SLIPPERY = True
T_MAX = 200  # Max steps per episode
EVAL_EPISODES = 100  # For reliable performance measurement

class ReinforceAgent:
    def __init__(self, env, gamma=0.95, learning_rate=0.01, lr_decay=0.999, 
                 baseline=False, entropy_coef=0.01, seed=42):
        self.env = env
        self.gamma = gamma
        self.initial_lr = learning_rate
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Initialize policy parameters (logits)
        self.policy_logits = np.zeros((env.observation_space.n, env.action_space.n))
        np.random.seed(seed)
        
        # For baseline computation
        if baseline:
            self.state_values = np.zeros(env.observation_space.n)
            self.value_lr = 0.01  # Separate learning rate for value function

    def select_action(self, state, training=True):
        action_probs = softmax(self.policy_logits[state])
        if training:
            return np.random.choice(len(action_probs), action_probs)
        return np.argmax(action_probs), action_probs

    def update_policy(self, states, actions, rewards):
        # Calculate discounted returns
        discounted_returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            discounted_returns[t] = running_return
        
        # Normalize returns for stability
        if self.baseline:
            # Compute advantages using baseline
            advantages = discounted_returns - np.array([self.state_values[s] for s in states])
            # Update value function
            for t, s in enumerate(states):
                self.state_values[s] += self.value_lr * (discounted_returns[t] - self.state_values[s])
        else:
            advantages = discounted_returns - np.mean(discounted_returns)
            advantages /= (np.std(discounted_returns) + 1e-8)
        
        # Policy gradient update
        policy_gradients = np.zeros_like(self.policy_logits)
        for t, (s, a) in enumerate(zip(states, actions)):
            # Get current action probabilities
            action_probs = softmax(self.policy_logits[s])
            
            # Policy gradient
            policy_gradients[s, a] = advantages[t] * (1 - action_probs[a])
            
            # Entropy bonus for exploration
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            policy_gradients[s] += self.entropy_coef * entropy
        
        # Update policy parameters
        self.policy_logits += self.learning_rate * policy_gradients
        
        # Learning rate decay
        self.learning_rate = max(self.initial_lr * (self.lr_decay ** len(states)), 1e-5)
        
        return np.mean(advantages ** 2)  # Return mean squared advantage as loss

    def train_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        total_reward = 0
        
        while not done and len(episode) < T_MAX:
            action, probs = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward, probs[action]))
            state = next_state
            total_reward += reward
        
        # Extract components from episode
        states, actions, rewards, action_probs = zip(*episode)
        
        # Update policy
        loss = self.update_policy(states, actions, rewards)
        
        return total_reward, loss, len(episode)

    def evaluate(self, n_episodes=EVAL_EPISODES):
        successes = 0
        steps = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            step = 0
            
            while not done and step < T_MAX:
                action, _ = self.select_action(state, training=False)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                step += 1
                
                if terminated and step < T_MAX:  # Reached goal
                    successes += 1
                    steps.append(step)
                    break
        
        success_rate = successes / n_episodes
        avg_steps = np.mean(steps) if steps else T_MAX
        std_steps = np.std(steps) if steps else 0
        
        return success_rate, avg_steps, std_steps

    def get_policy(self):
        policy = np.argmax(softmax(self.policy_logits), axis=1)
        return policy

def run_training(env, params, n_episodes=1000):
    agent = ReinforceAgent(
        env,
        gamma=params['gamma'],
        learning_rate=params['learning_rate'],
        lr_decay=params['learning_rate_decay'],
        baseline=True,  # Using baseline reduces variance
        entropy_coef=0.01
    )
    
    rewards = []
    losses = []
    episode_lengths = []
    best_success_rate = 0
    
    # Training loop
    for episode in range(n_episodes):
        reward, loss, length = agent.train_episode()
        rewards.append(reward)
        losses.append(loss)
        episode_lengths.append(length)
        
        # Periodic evaluation
        if (episode + 1) % 100 == 0:
            success_rate, avg_steps, _ = agent.evaluate()
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_policy = agent.get_policy()
            
            print(f"Episode {episode+1}: Reward={np.mean(rewards[-100:]):.1f}, "
                  f"Success={success_rate:.2f}, Avg Steps={avg_steps:.1f}")
    
    final_success, final_avg_steps, final_std = agent.evaluate()
    return {
        **params,
        'final_reward': np.mean(rewards[-100:]),
        'final_loss': np.mean(losses[-100:]),
        'success_rate': final_success,
        'avg_steps': final_avg_steps,
        'std_steps': final_std,
        'policy': best_policy.tolist(),
        'rewards': rewards,
        'losses': losses
    }

def visualize_training(rewards, losses, window=50):
    """Plot training progress with moving averages."""
    plt.figure(figsize=(12, 5))
    
    # Rewards plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(pd.Series(rewards).rolling(window).mean(), 'b-', label=f'{window}-episode Avg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.3, label='Episode Loss')
    plt.plot(pd.Series(losses).rolling(window).mean(), 'r-', label=f'{window}-episode Avg')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([4, 12]))

# Example usage:
if __name__ == "__main__":
    # Create environment with custom rewards
    env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
    
    # Set custom rewards (modify as needed)
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            new_transitions = []
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                if done:
                    new_reward = 10 if reward == -1 else -100  # Finish/Fall rewards
                else:
                    new_reward = -1  # Step penalty
                new_transitions.append((prob, next_s, new_reward, done))
            env.unwrapped.P[s][a] = new_transitions
    
    # Training parameters
    params = {
        'gamma': 0.95,
        'learning_rate': 0.01,
        'learning_rate_decay': 0.999,
        'finish_reward': 10,
        'fall_reward': -100,
        'step_reward': -1
    }
    
    # Train and evaluate
    results = run_training(env, params, n_episodes=5000)
    print(f"\nFinal Success Rate: {results['success_rate']:.2f}")
    print(f"Average Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
    
    # Visualize training
    visualize_training(results['rewards'], results['losses'])
    
    # Show learned policy
    print("\nLearned Policy:")

    print_policy(results['policy'])