import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import timeit
import csv 
from scipy.stats import norm
from sklearn.model_selection import ParameterSampler
from itertools import product

# Default parameters
SLIPPERY = True
TRAINING_EPISODES = 1000
NUM_EPISODES = 5
GAMMA = 0.9
T_MAX = 200
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.99

def draw_history(history, title):
    window_size = 50
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    data['rolling_avg'] = data[title].rolling(window_size).mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)
    sns.lineplot(x='Episode', y='rolling_avg', data=data)

    plt.title(title + ' Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([4, 12]))

class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, entropy_bonus=0.01, seed=0):
        self.entropy_bonus = entropy_bonus
        self.env = env
        self.gamma = gamma
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.episode_count = 0
        # Objeto que representa la política (J(theta)) como una matriz estados X acciones,
        # con una probabilidad inicial para cada par estado accion igual a: pi(a|s) = 1/|A|
        self.policy_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        np.random.seed(seed)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            # Escogemos la acción según el vector de policy_table correspondiente a la acción,
            # con una distribución de probabilidad igual a los valores actuales de este vector
            return np.random.choice(np.arange(self.env.action_space.n), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    '''def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        loss = -np.sum(np.log(self.policy_table[states, actions]) * discounted_rewards) / len(states)
        policy_logits = np.log(self.policy_table)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient
            # Alternativa:
            # policy_gradient = 1.0 / action_probs[actions[t]]
            # policy_logits[states[t], actions[t]] += self.learning_rate * G_t * policy_gradient
        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss'''

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        
        # Calculate discounted rewards
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        
        # Subtract baseline and normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards) > 0:
            discounted_rewards /= np.std(discounted_rewards)
        
        # Calculate loss (for monitoring)
        loss = -np.sum(np.log(self.policy_table[states, actions] + 1e-10) * discounted_rewards) / len(states)
        
        # Convert to logits for numerical stability
        logits = np.log(self.policy_table + 1e-10)
        
        # Update policy parameters
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            advantage = discounted_rewards[t]
            
            # Calculate gradient
            grad_log_p = -self.policy_table[state]  # ∂/∂θ log π(a|s)
            grad_log_p[action] += 1
            
            # Update with advantage
            logits[state] += self.learning_rate * advantage * grad_log_p
        
        # Entropy regularization (now correctly outside the per-timestep loop)
        for state in set(states):
            probs = self.policy_table[state]
            logits[state] += self.learning_rate * self.entropy_bonus * (1 + np.log(probs + 1e-10))
        
        # Convert back to probabilities
        self.policy_table = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        self.policy_table /= np.sum(self.policy_table, axis=1, keepdims=True)
        
        # Update learning rate
        self.episode_count += 1
        self.learning_rate = self.initial_learning_rate / (1.0 + self.lr_decay * self.episode_count)
        
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        episode = []
        done = False
        step = 0
        total_reward = 0
        while not done and step < T_MAX:
            action = self.select_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward = total_reward + reward
            step = step + 1
        loss = self.update_policy(zip(*episode))
        return total_reward, loss

    def policy(self):
        policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table

# Define the parameter space
params = {
    'gamma': [0.9, 0.95, 0.99],
    'training_episodes': [1000, 10000, 50000],
    'finish_reward': [0, 10, 100],
    'fall_reward': [-10, -100, -1000],
    'step_reward': [0, -1, -10],
    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
    'learning_rate_decay': [0.9, 0.95, 0.99, 0.999, 1.0],
    'entropy_bonus' : [0.01, 0.1, 0.2]
}

# Function to create a fractional factorial design
def create_fractional_factorial_design(params, n_samples=100):
    # Calculate the full factorial size
    full_factorial_size = np.prod([len(values) for values in params.values()])
    
    # If requested samples is more than full factorial, return full factorial
    if n_samples >= full_factorial_size:
        return list(ParameterSampler(params, n_iter=full_factorial_size, random_state=42))
    
    # Generate a balanced fractional factorial design
    all_combinations = list(product(*params.values()))
    np.random.seed(42)
    indices = np.sort(np.random.choice(len(all_combinations), n_samples, replace=False))
    
    # Create the parameter combinations
    param_combinations = []
    for idx in indices:
        combo = dict(zip(params.keys(), all_combinations[idx]))
        param_combinations.append(combo)
    
    return param_combinations

# Generate the parameter combinations
n_combinations = 150  # Number of combinations to test
param_combinations = create_fractional_factorial_design(params, n_combinations)

# Function to evaluate a parameter combination
def evaluate_param_combination(params_dict):
    # Extract parameters
    gamma = params_dict['gamma']
    training_episodes = params_dict['training_episodes']
    finish_reward = params_dict['finish_reward']
    fall_reward = params_dict['fall_reward']
    step_reward = params_dict['step_reward']
    learning_rate = params_dict['learning_rate']
    learning_rate_decay = params_dict['learning_rate_decay']
    entropy_bonus = params_dict['entropy_bonus'] 
    
    # Create environment
    env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
    
    # Modify rewards
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            new_transitions = []
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                # Adjust the reward
                if done:
                    if reward == -1:  # Reached the goal
                        new_reward = finish_reward
                    elif reward == -100:  # Fell off cliff
                        new_reward = fall_reward
                else:
                    new_reward = step_reward  # Standard step
                new_transitions.append((prob, next_s, new_reward, done))
            env.unwrapped.P[s][a] = new_transitions
    
    # Create agent
    agent = ReinforceAgent(env, gamma=gamma, learning_rate=learning_rate,
                          lr_decay=learning_rate_decay, entropy_bonus=entropy_bonus,seed=42)
    
    # Train the agent
    rewards = []
    losses = []
    
    for i in range(training_episodes):
        reward, loss = agent.learn_from_episode()
        rewards.append(reward)
        losses.append(loss)
    
    # Evaluate the agent
    total_steps = 0
    n_eval_episodes = 100  # Fewer episodes for faster evaluation
    
    succesful_steps = []
    succes_rate = 0
    for n_ep in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            action = agent.select_action(state, False)
            state, reward, done, truncated, _ = env.step(action)

            step_count += 1
            if done:
                succesful_steps.append(step_count)
                succes_rate += 1
                break
        
        total_steps += step_count
    
    avg_steps = total_steps / n_eval_episodes
    policy, _ = agent.policy()
    
    return {
        'gamma': gamma,
        'training_episodes': training_episodes,
        'finish_reward': finish_reward,
        'fall_reward': fall_reward,
        'step_reward': step_reward,
        'learning_rate': learning_rate,
        'learning_rate_decay': learning_rate_decay,
        'entropy_bonus': entropy_bonus,
        'avg_steps': avg_steps,
        'std_steps': np.std(succesful_steps),
        'final_reward': rewards[-1],
        'final_loss': losses[-1],
        'policy': policy.tolist(),
        'success_rate': succes_rate / n_eval_episodes
    }

# Main execution
if __name__ == "__main__":
    print(f"Running parameter sweep with {len(param_combinations)} combinations...")
    
    # File to save results
    results_file = "reinforce_param_sweep_results.csv"
    
    # Run parameter combinations and save results
    results = []
    for i, params_dict in enumerate(param_combinations):
        print(f"Running combination {i+1}/{len(param_combinations)}: {params_dict}")
        start_time = timeit.default_timer()
        result = evaluate_param_combination(params_dict)
        end_time = timeit.default_timer()
        result['execution_time'] = end_time - start_time
        results.append(result)
        
        # Save results incrementally
        if i == 0:  # First iteration, create file with headers
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[k for k in result.keys() if k != 'policy'])
                writer.writeheader()
                writer.writerow({k: result[k] for k in result.keys() if k != 'policy'})
        else:  # Append to existing file
            with open(results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[k for k in result.keys() if k != 'policy'])
                writer.writerow({k: result[k] for k in result.keys() if k != 'policy'})
    
    # Find the best combination
    best_idx = np.argmin([r['avg_steps'] for r in results])
    best_params = results[best_idx]
    
    print("\nBest parameter combination:")
    for key, value in best_params.items():
        if key != 'policy':
            print(f"{key}: {value}")
    
    print("\nBest policy:")
    best_policy = np.array(best_params['policy']).astype(int)
    print_policy(best_policy)
    
    # Plot parameter effects
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame([{k: r[k] for k in r.keys() if k != 'policy'} for r in results])
    
    # Plot the distribution of average steps by parameter
    for param in params.keys():
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=param, y='avg_steps', data=results_df)
        plt.title(f'Effect of {param} on Average Steps')
        plt.xlabel(param)
        plt.ylabel('Average Steps')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'param_effect_{param}.png')
        plt.close()
    
    # Optional: Run a single demonstration with the best parameters
    print("\nRunning demonstration with best parameters...")
    env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY, render_mode="human")
    
    # Apply the best rewards
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            new_transitions = []
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                if done:
                    if reward == -1:  # Reached the goal
                        new_reward = best_params['finish_reward']
                    elif reward == -100:  # Fell off cliff
                        new_reward = best_params['fall_reward']
                else:
                    new_reward = best_params['step_reward']
                new_transitions.append((prob, next_s, new_reward, done))
            env.unwrapped.P[s][a] = new_transitions
    
    # Create and train agent with best parameters
    agent = ReinforceAgent(env, gamma=best_params['gamma'], 
                          learning_rate=best_params['learning_rate'],
                          lr_decay=best_params['learning_rate_decay'], 
                          entropy_bonus=best_params['entropy_bonus'],
                          seed=42)
    
    rewards = []
    losses = []
    
    for i in range(best_params['training_episodes']):
        reward, loss = agent.learn_from_episode()
        rewards.append(reward)
        losses.append(loss)
    
    # Show the learning curves
    draw_history(rewards, "Reward")
    draw_history(losses, "Loss")
    
    # Demonstrate the learned policy
    state, _ = env.reset()
    done = False
    step = 0
    
    while not done and step < 100:
        action = agent.select_action(state, False)
        state, reward, done, truncated, _ = env.step(action)
        env.render()
        step += 1
        if done:
            break
    
    env.close()