SLIPPERY = False
TRAINING_EPISODES = 100000  # Might need more for slippery
NUM_EPISODES = 5 # For evaluation after training
GAMMA = 0.99 # Increased
T_MAX = 200
LEARNING_RATE = 0.005 # Drastically reduced
LEARNING_RATE_DECAY = 0.9999 # Keep or slightly increase decay rate if LR is small

import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gymnasium import Wrapper

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

class ReinforceAgent:'''
    def __init__(self, env, gamma, learning_rate, lr_decay=1, seed=0):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
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

    def update_policy(self, episode):
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
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table'''

class ReinforceAgent:
    def __init__(self, env, gamma, learning_rate, lr_decay=1, seed=0):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        np.random.seed(seed)

        # Store logits as the primary parameters
        # Initialize to zeros for uniform initial probabilities after softmax
        self.policy_logits = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def _get_action_probabilities(self, state):
        logits_s = self.policy_logits[state]
        # Softmax for probabilities (with stabilization)
        exp_logits_s = np.exp(logits_s - np.max(logits_s))
        return exp_logits_s / np.sum(exp_logits_s)

    def select_action(self, state, training=True):
        action_probabilities = self._get_action_probabilities(state)
        if training:
            return np.random.choice(self.env.action_space.n, p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode_data): # episode_data is from zip(*episode)
        states_list, actions_list, rewards_list = [list(t) for t in episode_data]
        
        states = np.array(states_list)
        actions = np.array(actions_list)
        rewards = np.array(rewards_list, dtype=float)

        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        # Normalize discounted rewards (helps stabilize training - acts as a simple baseline)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)

        total_objective_terms = 0.0 # For calculating loss (sum of log_prob * G_t)

        for t in range(len(states)):
            s_t = states[t]
            a_t = actions[t]
            G_t = discounted_rewards[t]

            current_action_probs_st = self._get_action_probabilities(s_t)
            
            # Calculate log_prob for the loss objective *before* updating logits
            log_prob_action_taken = np.log(current_action_probs_st[a_t] + 1e-9) # Add epsilon for stability
            total_objective_terms += log_prob_action_taken * G_t

            # Gradient update for all actions' logits in state s_t
            for a_k in range(self.env.action_space.n):
                if a_k == a_t:
                    gradient_component = (1 - current_action_probs_st[a_k])
                else:
                    gradient_component = (0 - current_action_probs_st[a_k]) # i.e., -current_action_probs_st[a_k]
                
                self.policy_logits[s_t, a_k] += self.learning_rate * G_t * gradient_component
        
        # Loss is the negative of the sum of (log_prob * G_t) terms, usually averaged
        loss = -total_objective_terms / len(states) 
        return loss

    def learn_from_episode(self):
        state, _ = self.env.reset()
        episode_buffer = [] # Changed name from 'episode' for clarity
        done = False
        terminated = False # Gymnasium update
        step = 0
        total_reward = 0
        while not done and not terminated and step < T_MAX:
            action = self.select_action(state, training=True) # Ensure training=True
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated # In CliffWalking, truncated usually means T_MAX hit by wrapper
            
            episode_buffer.append((state, action, reward))
            state = next_state
            total_reward += reward # Use +=
            step += 1
        
        if not episode_buffer: # Handle empty episodes if T_MAX=0 or immediate termination
            return total_reward, 0.0 

        # Pass the zipped episode data correctly
        loss = self.update_policy(list(zip(*episode_buffer)))
        self.learning_rate *= self.lr_decay # Use *=
        return total_reward, loss

    def policy(self):
        # Derive deterministic policy and policy table (probabilities) from logits
        policy_table_probs = np.zeros_like(self.policy_logits)
        for s in range(self.env.observation_space.n):
            policy_table_probs[s] = self._get_action_probabilities(s)
        
        deterministic_policy = np.argmax(policy_table_probs, axis=1)
        return deterministic_policy, policy_table_probs

env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        new_transitions = []
        for prob, next_s, reward, done in env.unwrapped.P[s][a]:
            # ajustamos la recompensa:
            if done:
                if reward == -1:
                    new_reward = 0
            elif reward == -100:
                new_reward = -100
            else:
                new_reward = -1
            new_transitions.append((prob, next_s, new_reward, done))
        env.unwrapped.P[s][a] = new_transitions


agent = ReinforceAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE,
                       lr_decay=LEARNING_RATE_DECAY, seed=8)
rewards = []
losses = []

for i in range(TRAINING_EPISODES):
    reward, loss = agent.learn_from_episode()
    policy, policy_table = agent.policy()
    rewards.append(reward)
    losses.append(loss)
    
    # Print more detailed information periodically
    if (i + 1) % 100 == 0 or i < 10:
        print(f"Episode {i+1}/{TRAINING_EPISODES}:")
        print(f"  Reward: {reward:.2f}, Loss: {loss:.6f}, Learning rate: {agent.learning_rate:.6f}")
        
        # Check policy entropy to monitor exploration
        entropy = -np.sum(policy_table * np.log(policy_table + 1e-10)) / policy_table.shape[0]
        print(f"  Policy entropy: {entropy:.6f}")
        
        # Check if policy has converged too much
        max_probs = np.max(policy_table, axis=1)
        highly_certain = np.sum(max_probs > 0.95)
        print(f"  States with >95% certainty: {highly_certain}/{policy_table.shape[0]}")
        
        # Print a sample of the policy
        print("  Policy sample:")
        print_policy(policy)
        print()

is_done = False
rewards = []
steps = 0
for n_ep in range(500):
    state, _ = env.reset()
    #print('Episode: ', n_ep)
    total_reward = 0
    for i in range(200):
        action = agent.select_action(state, False)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        env.render()
        steps += 1
        if is_done:
            break
        rewards.append(total_reward)
        #draw_rewards(rewards)
average_steps = steps/500
print(average_steps)
print_policy(policy)

draw_history(rewards, "Reward")
draw_history(losses, "Loss")