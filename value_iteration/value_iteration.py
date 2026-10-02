import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import csv

SLIPPERY = True
MOD_REWARD = True
T_MAX = 200
MEDIA = 500
max_iterations = 10000
gamma = [0.90, 0.95, 0.99]
convergencia = [1, 0.01, 1e-4, 1e-6]
finish_reward = [0, 10, 100]
fall_reward = [-10, -100, -1000]
step_reward = [0.0, -1, -10]



def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

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



FALL_REWARD = -10
FINISH_REWARD = 0

class ValueIterationAgent:
    def __init__(self, env, gamma, max_iter):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.max_iter = max_iter
        
    def calc_action_value(self, state, action):
        if state == 47:
            return FINISH_REWARD
        if state in range(37, 47):
            return FALL_REWARD
        action_value = sum([prob * (reward + self.gamma * self.V[next_state])
                            for prob, next_state, reward, _ 
                            in self.env.unwrapped.P[state][action]]) 
        return action_value

    def select_action(self, state):
        best_action = best_value = None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if not best_value or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def print_V(self):
        rows, cols = 4, 12 
        policy_arrows = [V_value for V_value in self.V]
        try:
            print(np.array(policy_arrows).reshape(rows, cols))
        except ValueError as e:
            print(f"Error reshaping policy: {e}")
            print("Policy array:", policy_arrows)

    def value_iteration(self):

        max_diff = 0
        for state in range(self.env.observation_space.n):
            #digamos que lo que está haciendo es calcular para la nueva iteracion los valores de V*
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            #self.V es para la t-1, new_V es t
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff

    '''
    def value_iteration(self):
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):  
                state_values.append(self.calc_action_value(state, action))
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    '''
    def policy(self):   
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy



def train(agent): 
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0
     
    for i in range(agent.max_iter):
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        #print("After value iteration, max_diff = " + str(max_diff))
        t += 1

        if max_diff < CONVERGENCIA:
            break

    #print("final")
    return rewards, max_diffs



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




is_done = False
paso = 1
steps = 0
resultados = []


for GAMMA in gamma:
    for CONVERGENCIA in convergencia:
        for FINISH_REWARD in finish_reward:
            for FALL_REWARD in fall_reward:
                for STEP_REWARD in step_reward:
                    env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
                    if MOD_REWARD:
                        for s in range(env.observation_space.n):
                            for a in range(env.action_space.n):
                                new_transitions = []
                                for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                                    # ajustamos la recompensa:
                                    if done:
                                        if reward == -1:
                                            new_reward = FINISH_REWARD
                                    elif reward == -100:
                                        new_reward = FALL_REWARD
                                    else:
                                        new_reward = STEP_REWARD
                                    new_transitions.append((prob, next_s, new_reward, done))
                                env.unwrapped.P[s][a] = new_transitions
                    agent = ValueIterationAgent(env, gamma=GAMMA, max_iter = max_iterations)
                    start_time = time.time()
                    train(agent)
                    train_time = time.time() - start_time
                    steps_list = []
                    for n_ep in range(MEDIA):
                        state, _ = env.reset()
                        #print('Episode: ', n_ep)
                        total_reward = 0
                        steps = 0
                        for i in range(T_MAX):
                            action = agent.select_action(state)
                            state, reward, is_done, truncated, _ = env.step(action)
                            total_reward = total_reward + reward
                            env.render()
                            steps += 1
                            if is_done:
                                break
                        steps_list.append(steps)
                        


                    avg_steps = np.mean(steps_list)
                    std_steps = np.std(steps_list)

                    resultados.append({
                        'gamma': GAMMA,
                        'convergencia': CONVERGENCIA,
                        'finish_reward': FINISH_REWARD,
                        'fall_reward': FALL_REWARD,
                        'step_reward': STEP_REWARD,
                        'avg_steps': avg_steps,
                        'std_steps': std_steps,
                        'train_time_sec': train_time
                    })
                    
                    print(paso)
                    paso += 1
                        
keys = resultados[0].keys()
with open('value_iteration_results2.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(resultados)

print("Resultados guardados en value_iteration_results.csv")    




'''
Best_config = [0.99 0.01 10 -1000 -10]
Worst_succ_config = [0.9 1 10 -1000 -1]
Worst_config = [0.9 1 0 -10 0]
'''
'''
env = gym.make("CliffWalking-v0", is_slippery=SLIPPERY)
if MOD_REWARD:
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            new_transitions = []
            for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                # ajustamos la recompensa:
                if done:
                    if reward == -1:
                        new_reward = 0
                elif reward == -100:
                    new_reward = -10
                else:
                    new_reward = 0
        new_transitions.append((prob, next_s, new_reward, done))

env.unwrapped.P[s][a] = new_transitions
agent = ValueIterationAgent(env, gamma=0.9, max_iter = max_iterations)
start_time = time.time()
train(agent)
train_time = time.time() - start_time
steps_list = []
for n_ep in range(MEDIA):
    state, _ = env.reset()
    #print('Episode: ', n_ep)
    total_reward = 0
    steps = 0
    for i in range(T_MAX):
        action = agent.select_action(state)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        env.render()
        steps += 1
        if is_done:
            break
    steps_list.append(steps)
                        

print()
print()
print_policy(agent.policy())
print()
print(np.mean(steps_list))
avg_steps = np.mean(steps_list)
std_steps = np.std(steps_list)
'''

