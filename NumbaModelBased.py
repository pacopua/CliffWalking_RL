# Declaración de constantes
T_MAX = 150
NUM_EPISODES = 30
GAMMA = 0.975 # Default, will be overridden in main
REWARD_THRESHOLD = -70 # Not currently used in main logic

import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
import csv

from sklearn.model_selection import ParameterSampler
# from itertools import product # Not used if ParameterSampler is used

# Numba import
from numba import njit

# --- Numba-fied helper functions ---
# These will be called by the agent's methods.
# They are defined outside the class for Numba to compile them effectively.

@njit(cache=True)
def _calc_action_value_numba(state: int, action: int, V: np.ndarray,
                             R_sas: np.ndarray, N_sas: np.ndarray, N_sa: np.ndarray,
                             gamma: float, nS: int):
    """
    Calculates Q(s,a) using the learned model (R_sas, N_sas, N_sa).
    R_sas: Expected reward R(s,a,s') - for this agent, it's the direct reward observed.
    N_sas: Counts N(s,a,s')
    N_sa: Counts N(s,a)
    """
    if N_sa[state, action] == 0:
        return 0.0  # Or some other default, e.g., -np.inf if you want to avoid unvisited
    
    action_value = 0.0
    total_transitions_from_sa = N_sa[state, action]

    for next_s in range(nS):
        if N_sas[state, action, next_s] > 0:
            prob = N_sas[state, action, next_s] / total_transitions_from_sa
            reward = R_sas[state, action, next_s] # Assuming R_sas stores the sample reward
            action_value += prob * (reward + gamma * V[next_s])
    return action_value

@njit(cache=True)
def _select_action_numba(state: int, V: np.ndarray,
                         R_sas: np.ndarray, N_sas: np.ndarray, N_sa: np.ndarray,
                         gamma: float, nS: int, nA: int):
    best_action = 0 # Default to action 0
    best_value = -np.inf # Initialize with a very small number
    
    # Find the first valid action's value to initialize best_value
    # This handles cases where initial Q-values might all be 0 or negative
    first_valid_value_found = False
    for action_idx in range(nA):
        action_val = _calc_action_value_numba(state, action_idx, V, R_sas, N_sas, N_sa, gamma, nS)
        if not first_valid_value_found or action_val > best_value : # check if N_sa[state, action_idx] > 0 inside calc
            best_value = action_val
            best_action = action_idx
            first_valid_value_found = True # Only set if calc_action_value was meaningful
        # elif action_val > best_value: #This line was the bug
        #     best_value = action_val
        #     best_action = action_idx
            
    # If no action has been explored from this state, N_sa might be 0 for all actions
    # leading to best_value remaining -np.inf. In such a case, a random action might be better.
    # For now, it will return action 0.
    # A more robust way could be to check if all N_sa[state, :] are 0 and return a random action.
    # However, the current logic will pick the one that _calc_action_value_numba returns highest
    # (which will be 0 if N_sa is 0 for that action).
    return best_action


@njit(cache=True)
def _value_iteration_sweep_numba(V_old: np.ndarray, R_sas: np.ndarray,
                                 N_sas: np.ndarray, N_sa: np.ndarray,
                                 gamma: float, nS: int, nA: int):
    """
    Performs one sweep of value iteration.
    Returns new V and max_diff.
    """
    V_new = np.copy(V_old) # Important to work on a copy
    max_diff = 0.0
    for s in range(nS):
        if np.sum(N_sa[s, :]) == 0: # If state 's' was never visited as a starting point of a transition
            continue # V[s] remains unchanged (e.g. 0 if V was initialized to 0)

        state_action_values = np.empty(nA, dtype=np.float64)
        for a in range(nA):
            state_action_values[a] = _calc_action_value_numba(s, a, V_old, R_sas, N_sas, N_sa, gamma, nS)
        
        best_state_value = np.max(state_action_values)
        if not np.isfinite(best_state_value): # Handles cases where all actions lead to -np.inf
            best_state_value = 0.0 # Or some other default for unreachable/unexplored states

        diff = np.abs(best_state_value - V_old[s])
        if diff > max_diff:
            max_diff = diff
        V_new[s] = best_state_value
    return V_new, max_diff

# --- Original Functions (modified slightly if needed) ---
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

def check_improvements(agent, env, num_episodes_check, t_max_check):
    reward_test = 0.0
    for _ in range(num_episodes_check):
        total_reward = 0.0
        state, _ = env.reset()
        for _ in range(t_max_check):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done or truncated:
                break
            state = new_state
        reward_test += total_reward
    reward_avg = reward_test / num_episodes_check
    return reward_avg

def train(agent, t_max_val_iter=1000): # Added max iterations for value iteration
    rewards_log = []
    max_diffs_log = []
    best_reward = -np.inf # Initialize with a very small number
    
    for i in range(t_max_val_iter): # Iterate up to t_max_val_iter times for value iteration
        max_diff = agent.value_iteration_step() # Changed to a single step
        max_diffs_log.append(max_diff)
        
        if i % 20 == 0: # Print less frequently
            print(f"Value Iteration step {i}, max_diff = {max_diff:.6f}")
        
        # Check improvements less frequently to save time, e.g., every 5 VI steps
        if i % 5 == 0 or max_diff < agent.epsilon :
            reward_test = check_improvements(agent, agent.env, NUM_EPISODES // 2, T_MAX) # Use fewer episodes for intermediate checks
            rewards_log.append(reward_test)
            
            if reward_test > best_reward:
                best_reward = reward_test
                #print(f"Best reward updated {reward_test:.2f} at VI step {i}")
            
        if max_diff < agent.epsilon:
            print(f"Value Iteration converged after {i+1} steps. Max_diff = {max_diff:.6f}")
            # Final check with full episodes
            reward_test = check_improvements(agent, agent.env, NUM_EPISODES, T_MAX)
            rewards_log.append(reward_test)
            if reward_test > best_reward: best_reward = reward_test
            print(f"Final best reward after convergence: {best_reward:.2f}")
            break
    else: # If loop finishes without break
        print(f"Value Iteration reached max {t_max_val_iter} steps. Max_diff = {max_diff:.6f}")
        reward_test = check_improvements(agent, agent.env, NUM_EPISODES, T_MAX)
        rewards_log.append(reward_test)
        print(f"Final reward after max VI steps: {rewards_log[-1]:.2f}")


    return rewards_log, max_diffs_log

class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories_per_vi_step, epsilon):
        self.env = env
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        
        self.current_state, _ = self.env.reset() # Renamed from self.state to avoid confusion
        
        # Model: R(s,a,s'), N(s,a,s'), N(s,a)
        # For R_sas, we store the sum of rewards for (s,a,s') and average later if needed,
        # or simply use the last observed reward if play_n_random_steps updates it.
        # The provided code `self.rewards[(self.state, action, new_state)] = reward`
        # suggests storing a single (likely the latest) reward.
        # Let's stick to that: R_sas will store the reward observed for a transition s,a -> s'
        self.R_sas = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64)
        self.N_sas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int32) # Counts for (s,a,s')
        self.N_sa = np.zeros((self.nS, self.nA), dtype=np.int32)      # Counts for (s,a)
        
        self.V = np.zeros(self.nS, dtype=np.float64)
        self.gamma = gamma
        self.num_trajectories_per_vi_step = num_trajectories_per_vi_step # Renamed for clarity
        self.epsilon = epsilon

    def play_n_random_steps(self, count):
        # This function interacts with the environment, so it remains a Python method.
        # Updates to Numba-compatible arrays (R_sas, N_sas, N_sa) are fine.
        for _ in range(count):
            action = self.env.action_space.sample() # Exploration
            # Ensure self.current_state is valid. It should be after reset or previous step.
            if self.current_state is None or self.current_state >= self.nS: # Basic safety
                 self.current_state, _ = self.env.reset()

            new_state, reward, is_done, truncated, _ = self.env.step(action)
            
            # Update model
            # R_sas stores the reward for the transition (s, a, s').
            # If multiple rewards are seen for the exact same (s, a, s'), this overwrites.
            # This matches the original defaultdict behavior for self.rewards.
            self.R_sas[self.current_state, action, new_state] = reward
            self.N_sas[self.current_state, action, new_state] += 1
            self.N_sa[self.current_state, action] += 1
            
            if is_done or truncated:
                self.current_state, _ = self.env.reset()
            else:
                self.current_state = new_state

    def calc_action_value(self, state, action):
        # Wrapper to call the Numba-fied version
        return _calc_action_value_numba(state, action, self.V, self.R_sas, self.N_sas, self.N_sa,
                                        self.gamma, self.nS)

    def select_action(self, state):
        # Wrapper to call the Numba-fied version
        return _select_action_numba(state, self.V, self.R_sas, self.N_sas, self.N_sa,
                                    self.gamma, self.nS, self.nA)

    def value_iteration_step(self):
        # Perform exploration (model learning step)
        self.play_n_random_steps(self.num_trajectories_per_vi_step)

        # Perform value iteration sweep using the learned model
        # Pass copies or ensure Numba function doesn't modify V in place unexpectedly if V_old is needed
        V_new, max_diff = _value_iteration_sweep_numba(
            self.V, self.R_sas, self.N_sas, self.N_sa,
            self.gamma, self.nS, self.nA
        )
        self.V = V_new # Update the agent's value function
        return max_diff

    def get_policy_array(self):
        policy = np.zeros(self.nS, dtype=np.int32)
        for s in range(self.nS):
            policy[s] = self.select_action(s) # This uses the Numba-optimized select_action
        return policy

def print_policy(policy_array, rows=4, cols=12):
    """Prints the policy in a grid format."""
    # Action mapping for CliffWalking: 0: ^, 1: >, 2: v, 3: < (standard in some Gym versions)
    # My CliffWalking-v0 with is_slippery=True: 0: up, 1: right, 2: down, 3: left
    # Let's assume 0: Up (^) (towards row 0), 1: Right (>) (towards col max)
    # 2: Down (v) (towards row max), 3: Left (<) (towards col 0)
    visual_help = {0:'^', 1:'>', 2:'v', 3:'<', -1:'?'} # -1 for undefined if needed
    
    if len(policy_array) != rows * cols:
        print(f"Warning: Policy length ({len(policy_array)}) doesn't match grid dimensions ({rows}x{cols}).")
    
    policy_arrows = [visual_help.get(int(action), '?') for action in policy_array]
    try:
        print("\nLearned Policy (0:^, 1:>, 2:v, 3:<):")
        reshaped_policy = np.array(policy_arrows).reshape(rows, cols)
        # For CliffWalking, state 0 is top-left. Standard printout is fine.
        # If state 0 was bottom-left, you might want to print `np.flipud(reshaped_policy)`
        print(reshaped_policy)
    except ValueError as e:
        print(f"Error reshaping policy: {e}")
        print("Policy array:", policy_arrows)


def print_learned_model_summary(agent):
    print("\n=== LEARNED ENVIRONMENT MODEL (Summary) ===")
    print(f"Shape of R_sas (rewards s,a,s'): {agent.R_sas.shape}")
    print(f"Shape of N_sas (counts s,a,s'): {agent.N_sas.shape}")
    print(f"Shape of N_sa (counts s,a): {agent.N_sa.shape}")

    print(f"\nNon-zero entries in N_sa (visited state-action pairs): {np.count_nonzero(agent.N_sa)}")
    
    # Example: Print transitions for a few states if they have been visited
    for s_idx in range(min(agent.nS, 5)): # Print for first 5 states
        if np.sum(agent.N_sa[s_idx, :]) > 0: # If this state was ever a starting point
            print(f"\nState {s_idx}:")
            for a_idx in range(agent.nA):
                if agent.N_sa[s_idx, a_idx] > 0: # If this action was taken in this state
                    print(f"  Action {a_idx} (taken {agent.N_sa[s_idx, a_idx]} times):")
                    for next_s_idx in range(agent.nS):
                        if agent.N_sas[s_idx, a_idx, next_s_idx] > 0:
                            prob = agent.N_sas[s_idx, a_idx, next_s_idx] / agent.N_sa[s_idx, a_idx]
                            reward = agent.R_sas[s_idx, a_idx, next_s_idx]
                            print(f"    → State {next_s_idx}: Prob={prob:.2f}, Reward={reward:.2f} (seen {agent.N_sas[s_idx, a_idx, next_s_idx]} times)")


def main():
    params = {
        'gamma': [0.9, 0.95, 0.99],
        'num_trajectories_per_vi_step': [100, 500, 1000], # Reduced from 10k for speed
        'finish_reward': [0, 10], # Reduced options
        'fall_reward': [-100, -500], # Reduced options
        'step_reward': [-1, -5],     # Reduced options
        'epsilon': [0.1, 0.01, 0.001] # Numba is sensitive to very small epsilons for float comparisons
    }

    n_combinations = 20 # Reduced for faster testing
    param_combinations = list(ParameterSampler(params, n_iter=n_combinations, random_state=42))
    
    # For a single quick test:
    # param_combinations = [{
    #     'gamma': 0.99, 'num_trajectories_per_vi_step': 500, 
    #     'finish_reward': 10, 'fall_reward': -100, 'step_reward': -1, 'epsilon': 0.01
    # }]


    output_csv_file = 'fractional_results_numba.csv'
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['gamma', 'num_trajectories_per_vi_step', 'finish_reward',
                  'fall_reward', 'step_reward', 'epsilon',
                  'elapsed_time_train_s', 'avg_steps_eval']
        csv_writer.writerow(header)
        print(f"Running experiments. Results will be saved to {output_csv_file}")

        for i, combo in enumerate(param_combinations):
            print(f"\n--- Testing Combination {i+1}/{len(param_combinations)}: {combo} ---")
            
            # Average results over a few full runs for stability
            num_repetitions_per_combo = 2 # Reduced from 3
            combo_total_time = 0
            combo_total_avg_steps = 0

            for rep in range(num_repetitions_per_combo):
                print(f"  Repetition {rep+1}/{num_repetitions_per_combo}")
                # CliffWalking's P dictionary structure:
                # P[state][action] = list of (probability, next_state, reward, done) tuples
                # is_slippery=False makes it deterministic, which is not what P usually implies.
                # For CliffWalking, there's usually no stochasticity in transitions by default.
                # The `is_slippery` parameter is more common in FrozenLake.
                # Let's assume CliffWalking here is deterministic unless modified.
                env = gym.make('CliffWalking-v0') # Default CliffWalking

                # --- Modify rewards in the environment ---
                # This is a bit hacky and relies on the internal structure of Gym's discrete.py
                # env.unwrapped.P is the transition model
                # P[state][action] is a list of tuples (prob, next_state, reward, done)
                # For deterministic envs, this list has one tuple: (1.0, next_state, reward, done)
                
                # Identify cliff states and goal state (specific to 4x12 CliffWalking)
                # Goal state is 47 (bottom-right). States 37-46 are cliff states.
                # Start state is 36 (bottom-left).
                rows, cols = env.unwrapped.shape # Typically 4, 12
                goal_state = rows * cols - 1
                cliff_states = [s for s in range(cols * (rows - 1) + 1, cols * rows - 1)]

                new_P = {s: {a: [] for a in range(env.action_space.n)} for s in range(env.observation_space.n)}

                for s in range(env.observation_space.n):
                    for a in range(env.action_space.n):
                        transitions = env.unwrapped.P[s][a]
                        for prob, next_s, original_reward, done in transitions:
                            new_reward = original_reward # Start with original
                            if done and next_s == goal_state : # Reached goal
                                new_reward = combo['finish_reward']
                            elif next_s in cliff_states: # Fell off cliff (next_s is the cliff state itself)
                                # In CliffWalking, stepping on a cliff state means -100 and reset.
                                # The 'done' flag might not be true if you just land on the cliff,
                               # but the reward is -100. Let's use the original_reward check.
                                if original_reward == -100: # Standard cliff penalty
                                    new_reward = combo['fall_reward']
                                # If it's just a normal step, not goal, not cliff
                            elif not done and next_s != goal_state and original_reward != -100 :
                                new_reward = combo['step_reward']
                            
                            new_P[s][a].append((prob, next_s, new_reward, done))
                env.unwrapped.P = new_P
                # --- End of reward modification ---

                agent = DirectEstimationAgent(
                    env,
                    gamma=combo['gamma'],
                    num_trajectories_per_vi_step=combo['num_trajectories_per_vi_step'],
                    epsilon=combo['epsilon']
                )

                start_time = timeit.default_timer()
                # The train function now handles the value iteration loop
                train_rewards_log, train_max_diffs_log = train(agent, t_max_val_iter=500) # Max 500 VI steps
                end_time = timeit.default_timer()
                elapsed_time = end_time - start_time
                combo_total_time += elapsed_time

                # Evaluate final policy
                avg_steps_eval = 0
                num_eval_episodes = NUM_EPISODES # Use full NUM_EPISODES for final eval
                for _ in range(num_eval_episodes):
                    state, _ = env.reset()
                    steps = 0
                    terminated = False
                    truncated = False
                    while not (terminated or truncated):
                        action = agent.select_action(state) # Uses learned V
                        state, _, terminated, truncated, _ = env.step(action)
                        steps += 1
                        if steps >= T_MAX:
                            truncated = True # Ensure termination
                    avg_steps_eval += steps
                
                avg_steps_eval /= num_eval_episodes
                combo_total_avg_steps += avg_steps_eval
                
                print(f"  Rep {rep+1}: Train time: {elapsed_time:.2f}s, Avg eval steps: {avg_steps_eval:.2f}")
                # print_policy(agent.get_policy_array()) # Optional: print policy for inspection
                # print_learned_model_summary(agent) # Optional
                env.close()
            
            avg_combo_time = combo_total_time / num_repetitions_per_combo
            avg_combo_steps = combo_total_avg_steps / num_repetitions_per_combo

            csv_writer.writerow([
                combo['gamma'],
                combo['num_trajectories_per_vi_step'],
                combo['finish_reward'],
                combo['fall_reward'],
                combo['step_reward'],
                combo['epsilon'],
                avg_combo_time,
                avg_combo_steps
            ])
            csvfile.flush() # Ensure data is written to disk periodically
            print(f"Finished Combo {i+1}. Avg time: {avg_combo_time:.2f}s, Avg steps: {avg_combo_steps:.2f}")

    print(f"\nAll experiments complete. Results saved to {output_csv_file}")
    
    # Example of how to plot final results (if you want to visualize them)
    # try:
    #     df_results = pd.read_csv(output_csv_file)
    #     print("\nExperiment Results Summary:")
    #     print(df_results.head())
    #     # Example plot:
    #     # plt.figure()
    #     # sns.scatterplot(data=df_results, x='gamma', y='avg_steps_eval', hue='epsilon')
    #     # plt.title('Performance vs. Gamma and Epsilon')
    #     # plt.show()
    # except Exception as e:
    #     print(f"Could not read or plot results: {e}")


if __name__ == "__main__":
    # It's good practice to warm up Numba functions if precise timing of the first run matters
    # For this script, the overhead of the first compilation is part of the "setup" for each combo.
    # If you had one agent and ran it many times, you'd warm up outside the main loop.
    print("Starting Numba Direct Estimation Agent experiments...")
    main()