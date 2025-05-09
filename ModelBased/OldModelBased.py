# Declaración de constantes

# Número máximo de pasos por episodio
T_MAX = 150

# Número de episodios para la prueba
# (repeticiones)
NUM_EPISODES = 30

#factor de descuento:
GAMMA = 0.975
#cada paso -1, caer por el agujero -100, llegar al final +0

#como de buena es la recompensa
#si la recompensa es mayor que este umbral, se considera que ha convergido 
REWARD_THRESHOLD = -70

import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import csv 

env = gym.make('CliffWalking-v0', is_slippery = True)

def test_episode(agent, env):
    state, _ = env.reset()
    is_done = False
    t = 0
    print("episode: ", t)
    while not is_done:
        action = agent.select_action(state)
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

#esta funcion comprueba todos los episodios (repeticiones) y para cada episodio el numero de iteraciones (t, que son los pasos)
def check_improvements(agent):
    reward_test = 0.0
    for i in range(NUM_EPISODES):
        total_reward = 0.0
        state, _ = env.reset()
        for j in range(T_MAX):
            #print("check_improvements: ", i)
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done: 
                break
            state = new_state
        reward_test += total_reward
    reward_avg = reward_test / NUM_EPISODES
    return reward_avg

def train(agent): 
    rewards = []
    max_diffs = []
    t = 0
    best_reward = None

    #itera hasta que converja
    while best_reward is None or best_reward < REWARD_THRESHOLD:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        print("After value iteration, max_diff = " + str(max_diff))
        t += 1
        reward_test = check_improvements(agent)
        rewards.append(reward_test)
               
        if best_reward is None or reward_test > best_reward:
            print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
            best_reward = reward_test
    
    return rewards, max_diffs

'''
def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))
'''
class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories):
        self.env = env
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories

    #esto es la exploracion, da un numero de pasos tomando acciones aleatorias
    #para cada acción que toma en un estado, se apunta su recompensa, y los diferentes
    #estados a los que le ha llevado esa acción con el numero de veces para cada una de estas
    #de manera que en rewards tenemos R(s,a,s') y en transits tenemos para cada (estado,accion)
    #un vector de acciones con el numero de veces que han aparecido
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset() 
            else: 
                self.state = new_state

    def calc_action_value(self, state, action):
        #estimamos la Q*(s,a). 
        #target count es cunatas veces hemos hecho esa accion en ese estado
        #total es el numero de repesticiones de esa accion en el estado
        #para todos los estados, se guarda la recompensa de llegar al estado ese
        #items() devuelve una lista de tuplas (key, value), donde key es el estado 
        #y value el numero de veces que ha aparecido
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for s_, count in target_counts.items():
            r = self.rewards[(state, action, s_)]
            prob = (count / total)
            action_value += prob*(r + self.gamma * self.V[s_])
        return action_value

    def select_action(self, state):
        #self.env.action_space.n: el environment tiene action_space y observation_space. Uno es el espacio de acciones
        #el otro el espacio de estados. (.n) solo da el tamaño
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # mira para cada estado que acción le dará la mejor utilidad
    def value_iteration(self):
        #num_trajectories es el número de pasos aleatorios que va a dar, 
        #de esta manera nos aseguramos de explorar bien el espacio
        self.play_n_random_steps(self.num_trajectories)

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
    #que devuelve max_diff
    #es el maximo de la diferencia entre el valor de la 
    #iteracion t y t-1 de entre todos los estados!
    
    def print_V(self):
        rows, cols = 4, 12 # CliffWalking grid dimensions
        policy_arrows = [V_value for V_value in self.V]
        try:
            print(np.array(policy_arrows).reshape(rows, cols))
        except ValueError as e:
            print(f"Error reshaping policy: {e}")
            print("Policy array:", policy_arrows)

    def policy(self):   
        print("Policy:")
        policy = np.zeros(env.observation_space.n) 
        #para cada uno de los estados, cojo la acción que me de mayor valor!
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))    
        print_policy(policy)    
        return policy

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

def inspect_policy(agent):
    is_done = False
    rewards = []
    for n_ep in range(NUM_EPISODES):
        print("episodio: ", n_ep)
        state, _ = env.reset()
        #print('Episode: ', n_ep)
        total_reward = 0
        is_done = False
        while not is_done:
            #print("inspect policy: ", i)
            action = agent.select_action(state)
            state, reward, is_done, truncated, _ = env.step(action)
            total_reward = total_reward + reward
            #env.render()
            if is_done:
                break
        rewards.append(total_reward)
    draw_rewards(rewards)


def print_learned_model(agent):
    print("\n=== LEARNED ENVIRONMENT MODEL ===")
    
    # Get all unique (state, action) pairs
    sa_pairs = list(agent.transits.keys())
    
    for state in range(agent.env.observation_space.n):
        print(f"\nState {state}:")
        for action in range(agent.env.action_space.n):
            if (state, action) not in agent.transits:
                continue
                
            print(f"  Action {action}:")
            total = sum(agent.transits[(state, action)].values())
            
            # Print transitions and rewards
            for s_next, count in agent.transits[(state, action)].items():
                prob = count / total
                reward = agent.rewards.get((state, action, s_next), 0)
                print(f"    → State {s_next}: Prob={prob:.2f}, Reward={reward}")
                


def main():

    #print(env.unwrapped.P)
    
    #for i in range(5):
    #    csv_file = f"cliffwalking_.csv"

    agent = DirectEstimationAgent(env, gamma=GAMMA, num_trajectories = 10000)
    
    train(agent)
    print_learned_model(agent)
    print("1")

    agent.policy()

    print("2")
    inspect_policy(agent)

    print("vamos a enseñar agente!")
    new_env = gym.make('CliffWalking-v0', render_mode = "human", is_slippery = True)

    state, _ = new_env.reset()
    new_env.render()
    is_done = False
    suma_rec = 0
    t = 0
    while not is_done:
        action = agent.select_action(state)
        state, reward, is_done, truncated, _ = new_env.step(action)
        suma_rec += reward
        new_env.render()
        t += 1

    print("Valor final: ", suma_rec)

    print("imprimpo las V*")
    agent.print_V()

if __name__ == "__main__":
    main()