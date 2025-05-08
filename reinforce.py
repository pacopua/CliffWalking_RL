SLIPPERY = True
TRAINING_EPISODES = 1000
NUM_EPISODES = 5
GAMMA = 0.9
T_MAX = 50
LEARNING_RATE = 1.0
LEARNING_RATE_DECAY = 0.99

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
        self.learning_rate = self.learning_rate * self.lr_decay
        return total_reward, loss

    def policy(self):
        policy = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n):
            action_probabilities = self.policy_table[s]
            policy[s] = np.argmax(action_probabilities)
        return policy, self.policy_table

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
    #print(policy_table)
    #print(f"Last reward: {reward}, last loss: {loss}, new lr: {agent.learning_rate}")
    #print_policy(policy)
    #print(f"End of iteration [{i + 1}/{TRAINING_EPISODES}]")
    rewards.append(reward)
    losses.append(loss)

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