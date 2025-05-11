# Declaración de constantes
SLIPPERY = True
TRAINING_EPISODES = 100000  # Might need more for slippery
NUM_EPISODES = 5 # For evaluation after training
GAMMA = 0.99 # Increased
T_MAX = 200
LEARNING_RATE = 0.005 # Drastically reduced
LEARNING_RATE_DECAY = 0.9999 # Keep or slightly increase decay rate if LR is small

# Imports necesarios
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gymnasium import Wrapper

class CustomFrozenLakeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
        if is_done and reward < 1:
            reward = -0.2
        if not is_done:
            reward = reward - 0.01
        return state, reward, is_done, truncated, info

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
    print(np.array(policy_arrows).reshape([-1, 4]))

class ReinforceAgent:
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
        print("Here is jonny")
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table
'''
env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
agent = ReinforceAgent(env_fast, gamma=GAMMA, learning_rate=LEARNING_RATE,
                       lr_decay=LEARNING_RATE_DECAY, seed=8)
rewards = []
losses = []
for i in range(TRAINING_EPISODES):
    reward, loss = agent.learn_from_episode()
    rewards.append(reward)
    losses.append(loss)
draw_history(rewards, "Reward")
draw_history(losses, "Loss")
print_policy(agent.policy()[0])
'''


env = gym.make("CliffWalking-v0", is_slippery=False)
agent = ReinforceAgent(env, gamma=GAMMA, learning_rate=LEARNING_RATE, lr_decay=LEARNING_RATE_DECAY, seed=0)
rewards = []
losses = []
for i in range(TRAINING_EPISODES):
    reward, loss = agent.learn_from_episode()
    policy, policy_table = agent.policy()
    print(policy_table)
    print(f"Last reward: {reward}, last loss: {loss}, new lr: {agent.learning_rate}")
    print_policy(policy)
    print(f"End of iteration [{i + 1}/{TRAINING_EPISODES}]")
    rewards.append(reward)
    losses.append(loss)

draw_history(rewards, "Reward")
draw_history(losses, "Loss")