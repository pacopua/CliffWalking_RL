import numpy as np
import gymnasium as gym
import time
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from numba import njit
import sys
import json
import csv

# Add debug information about CPU detection
def get_cpu_info():
    """Get information about CPU detection for debugging"""
    info = {
        'os_cpu_count': os.cpu_count(),
        'multiprocessing_cpu_count': multiprocessing.cpu_count(),
        'available_processors': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'python_version': sys.version,
        'multiprocessing_start_method': multiprocessing.get_start_method(),
    }
    return info

# Numba-optimized core functions
@njit
def _select_action_numba(state, Q, n_actions, epsilon):
    if np.random.random() <= epsilon:
        return np.random.randint(0, n_actions)
    else:
        q_values_for_state = Q[state]
        max_q = -np.inf
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] > max_q:
                max_q = q_values_for_state[i]

        count = 0
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] == max_q:
                count += 1

        best_actions = np.empty(count, dtype=np.int64)
        idx = 0
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] == max_q:
                best_actions[idx] = i
                idx += 1

        if len(best_actions) == 0: # Should not happen if Q is initialized
             return np.random.randint(0, n_actions)
        return best_actions[np.random.randint(0, len(best_actions))]

@njit
def _update_q_numba(Q, state, action, reward, next_state, is_done, alpha, gamma):
    q_next_max = 0.0
    if not is_done:
        max_val = -np.inf
        for i in range(Q[next_state].shape[0]):
            if Q[next_state, i] > max_val:
                max_val = Q[next_state, i]
        q_next_max = max_val

    Q[state, action] += alpha * (reward + gamma * q_next_max - Q[state, action])
    return Q

@njit
def _modify_reward_numba(state, action, next_state, reward, is_done, finish_reward, fall_reward, step_reward, use_custom_rewards):
    if not use_custom_rewards:
        return reward
    GOAL_STATE_CLIFFWALKING = 47
    ORIGINAL_REWARD_FALL = -100
    ORIGINAL_REWARD_STEP = -1

    if reward == ORIGINAL_REWARD_FALL and fall_reward is not None:
        return fall_reward
    if is_done and next_state == GOAL_STATE_CLIFFWALKING and finish_reward is not None:
        return finish_reward
    if reward == ORIGINAL_REWARD_STEP and step_reward is not None:
        return step_reward
    return reward


class HybridQLearningAgent:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.5, epsilon_decay=0.001, epsilon_end=0.01,
                 finish_reward=None, fall_reward=None, step_reward=None, t_max=200):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = np.random.uniform(low=0, high=0.01, size=(self.n_states, self.n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon # Initial epsilon for training
        self.current_epsilon = epsilon # Current epsilon, decayed during training
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.finish_reward = finish_reward
        self.fall_reward = fall_reward
        self.step_reward = step_reward
        self.use_custom_rewards = any(x is not None for x in [finish_reward, fall_reward, step_reward])
        self.t_max = t_max
        self.training_steps_counter = 0 # Renamed to avoid conflict
        self.GOAL_STATE_CLIFFWALKING = 47

    def decay_epsilon(self):
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon * (1 - self.epsilon_decay))

    def select_action(self, state, training=True):
        # Use self.current_epsilon for training, 0 for evaluation
        epsilon_to_use = self.current_epsilon if training else 0.0
        return _select_action_numba(state, self.Q, self.n_actions, epsilon_to_use)

    def modify_reward(self, state, action, next_state, reward, is_done):
        return _modify_reward_numba(
            state, action, next_state, reward, is_done,
            self.finish_reward, self.fall_reward, self.step_reward,
            self.use_custom_rewards
        )

    def update_Q(self, state, action, reward, next_state, is_done):
        self.Q = _update_q_numba(
            self.Q, state, action, reward, next_state, is_done, self.alpha, self.gamma
        )

    def learn_from_episode(self): # This is for a single training episode
        state, _ = self.env.reset()
        # Reset current_epsilon at the start of a training session if needed, or ensure it's handled by train()
        # For now, assuming train() manages the overall epsilon decay schedule.

        for step in range(self.t_max):
            action = self.select_action(state, training=True) # Always explore during training episode
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            reward = self.modify_reward(state, action, next_state, env_reward, terminated)
            self.update_Q(state, action, reward, next_state, terminated) # Update Q-table
            state = next_state
            self.training_steps_counter += 1
            if terminated or truncated:
                break
        # Training metrics are not primary focus here, but could be collected if needed
        # For this structure, evaluation metrics are collected separately
        return # No explicit return needed if metrics are collected outside or via evaluate

    def train(self, num_episodes):
        # Reset epsilon at the start of a full training run
        self.current_epsilon = self.epsilon # Use the initial epsilon from params

        for i in range(num_episodes):
            self.learn_from_episode() # Run one training episode
            self.decay_epsilon()      # Decay epsilon after each episode

        return {'Q': self.Q, 'total_training_steps': self.training_steps_counter}

    def evaluate(self, num_eval_episodes):
        """Evaluates the agent's greedy policy."""
        eval_rewards = []
        eval_successes = []
        eval_steps = []

        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            is_successful = False
            for step in range(self.t_max):
                episode_steps += 1
                action = self.select_action(state, training=False) # Epsilon = 0 for evaluation
                next_state, env_reward, terminated, truncated, _ = self.env.step(action)
                # Use modified rewards for evaluation consistency if desired, or original env_reward
                reward = self.modify_reward(state, action, next_state, env_reward, terminated)
                episode_reward += reward
                state = next_state
                if terminated:
                    if next_state == self.GOAL_STATE_CLIFFWALKING:
                        is_successful = True
                    break
                if truncated:
                    break

            eval_rewards.append(episode_reward)
            eval_successes.append(is_successful)
            eval_steps.append(episode_steps)

        mean_eval_reward = np.mean(eval_rewards) if eval_rewards else np.nan
        eval_success_rate = np.mean(eval_successes) if eval_successes else np.nan

        successful_eval_steps = [s for i, s in enumerate(eval_steps) if eval_successes[i]]
        mean_eval_steps_if_successful = np.mean(successful_eval_steps) if successful_eval_steps else np.nan

        return {
            'eval_mean_reward': mean_eval_reward,
            'eval_success_rate': eval_success_rate,
            'eval_mean_steps_if_successful': mean_eval_steps_if_successful
        }

def worker_initializer():
    pid = os.getpid()
    seed = int(time.time() * 1000) % 10000 + pid
    np.random.seed(seed)

def train_worker(worker_args):
    worker_id, params, num_train_episodes, num_eval_episodes = worker_args # Added num_eval_episodes
    try:
        env = gym.make("CliffWalking-v0", render_mode=None, is_slippery=True)
        agent = HybridQLearningAgent(
            env=env,
            gamma=params.get('gamma', 0.99),
            alpha=params.get('alpha', 0.1),
            epsilon=params.get('epsilon', 0.5), # This is initial epsilon for training
            epsilon_decay=params.get('epsilon_decay', 0.001),
            epsilon_end=params.get('epsilon_end', 0.01),
            finish_reward=params.get('finish_reward'),
            fall_reward=params.get('fall_reward'),
            step_reward=params.get('step_reward'),
            t_max=params.get('t_max', 200)
        )

        start_time = time.time()
        training_details = agent.train(num_train_episodes) # Train the agent
        training_time = time.time() - start_time

        # After training, evaluate the learned policy
        eval_results = agent.evaluate(num_eval_episodes)

        results_to_return = {
            'sample_id': worker_id,
            'training_time': training_time,
            'Q_final': training_details['Q'], # Could be large, consider if needed
            'total_training_steps': training_details['total_training_steps'],
            **eval_results # Add all evaluation metrics
        }
        env.close()
        return results_to_return
    except Exception as e:
        import traceback
        print(f"Worker {worker_id} encountered error: {e}\n{traceback.format_exc()}")
        raise

def run_enhanced_parallel_training(params, samples=10, num_train_episodes=2000, num_eval_episodes=100, force_num_workers=None):
    cpu_info = get_cpu_info()
    if force_num_workers is not None:
        max_workers = force_num_workers
    else:
        max_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 4)
    max_workers = max(1, min(max_workers, samples))

    print(f"Running {samples} samples using {max_workers} workers.")
    print(f"  Training for {num_train_episodes} episodes each.")
    print(f"  Evaluating for {num_eval_episodes} episodes each after training.")

    worker_args_list = [(i, params, num_train_episodes, num_eval_episodes) for i in range(samples)]

    raw_results_list = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer) as executor:
        start_time_total = time.time()
        futures = [executor.submit(train_worker, args) for args in worker_args_list]
        for future in futures:
            try:
                raw_results_list.append(future.result())
            except Exception as e:
                print(f"Error in worker process: {e}")
        total_wall_clock_time = time.time() - start_time_total

    if not raw_results_list:
        print("No results returned from workers!")
        return None

    # Aggregate evaluation metrics across samples
    all_eval_mean_rewards = np.array([r['eval_mean_reward'] for r in raw_results_list if 'eval_mean_reward' in r])
    all_eval_success_rates = np.array([r['eval_success_rate'] for r in raw_results_list if 'eval_success_rate' in r])
    all_eval_mean_steps_if_successful = np.array([r['eval_mean_steps_if_successful'] for r in raw_results_list if 'eval_mean_steps_if_successful' in r])
    all_training_times = np.array([r['training_time'] for r in raw_results_list if 'training_time' in r])

    # Calculate mean and std for aggregated metrics
    final_eval_mean_reward = np.nanmean(all_eval_mean_rewards) if len(all_eval_mean_rewards) > 0 else np.nan
    final_eval_std_reward = np.nanstd(all_eval_mean_rewards) if len(all_eval_mean_rewards) > 0 else np.nan

    final_eval_mean_success_rate = np.nanmean(all_eval_success_rates) if len(all_eval_success_rates) > 0 else np.nan
    final_eval_std_success_rate = np.nanstd(all_eval_success_rates) if len(all_eval_success_rates) > 0 else np.nan

    # For steps, only average if there were successful episodes
    valid_steps = all_eval_mean_steps_if_successful[~np.isnan(all_eval_mean_steps_if_successful)]
    final_eval_mean_steps_if_successful = np.nanmean(valid_steps) if len(valid_steps) > 0 else np.nan
    final_eval_std_steps_if_successful = np.nanstd(valid_steps) if len(valid_steps) > 0 else np.nan

    mean_training_time_per_sample = np.nanmean(all_training_times) if len(all_training_times) > 0 else np.nan
    std_training_time_per_sample = np.nanstd(all_training_times) if len(all_training_times) > 0 else np.nan

    return {
        'eval_mean_reward': final_eval_mean_reward,
        'eval_std_reward': final_eval_std_reward,
        'eval_mean_success_rate': final_eval_mean_success_rate,
        'eval_std_success_rate': final_eval_std_success_rate,
        'eval_mean_steps_if_successful': final_eval_mean_steps_if_successful,
        'eval_std_steps_if_successful': final_eval_std_steps_if_successful,
        'mean_training_time_per_sample': mean_training_time_per_sample,
        'std_training_time_per_sample': std_training_time_per_sample,
        'total_wall_clock_time': total_wall_clock_time,
        'speedup_factor': sum(all_training_times) / total_wall_clock_time if total_wall_clock_time > 0 and len(all_training_times) > 0 else 0,
        'cpu_info': cpu_info,
        # 'all_Qs_final': [r['Q_final'] for r in raw_results_list] # Optional: if you need all Q tables
    }

if __name__ == "__main__":
    all_parameter_sets = []
    try:
        with open('results/random_search_trials.json', 'r') as f:
            all_parameter_sets.extend(json.load(f))
    except FileNotFoundError as e:
        print(f"Error: Could not find parameter file: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from parameter files. Check their format.")
        sys.exit(1)

    if not all_parameter_sets:
        print("No parameter sets found. Exiting.")
        sys.exit(0)

    # --- CONFIGURABLE SETTINGS ---
    default_num_train_episodes = 2000
    num_eval_episodes_after_training = 100 # Number of episodes for final evaluation
    samples_per_param_set = 5
    force_num_workers = None
    # --- END CONFIGURABLE SETTINGS ---

    csv_file_name = 'results/training_results_with_evaluation.csv'
    csv_header = [
        'gamma', 'alpha', 'epsilon', 'epsilon_decay', 'epsilon_end',
        'finish_reward', 'fall_reward', 'step_reward', 't_max',
        'num_train_episodes', 'num_eval_episodes',
        'samples_per_param_set',
        'eval_mean_reward', 'eval_std_reward',                           # EVAL
        'eval_mean_success_rate', 'eval_std_success_rate',               # EVAL
        'eval_mean_steps_if_successful', 'eval_std_steps_if_successful',# EVAL
        'mean_training_time_per_sample', 'std_training_time_per_sample',
        'total_wall_clock_time', 'speedup_factor',
        'os_cpu_count', 'available_processors'
    ]

    all_run_results_for_csv = []
    total_runs = len(all_parameter_sets)
    print(f"Starting parameter sweep for {total_runs} distinct parameter sets.")
    print(f"Each parameter set will be run {samples_per_param_set} times (samples).")
    print(f"Evaluation after training will use {num_eval_episodes_after_training} episodes.")

    for run_idx, current_params_from_json in enumerate(all_parameter_sets):
        print(f"\n--- Running parameter set {run_idx + 1}/{total_runs} ---")
        print(f"Parameters from JSON: {current_params_from_json}")

        num_train_episodes_for_this_run = current_params_from_json.get('num_episodes', default_num_train_episodes)
        # 'num_episodes' from JSON is now interpreted as num_train_episodes

        final_results_agg = run_enhanced_parallel_training(
            params=current_params_from_json, # Pass the full set from JSON
            samples=samples_per_param_set,
            num_train_episodes=num_train_episodes_for_this_run,
            num_eval_episodes=num_eval_episodes_after_training, # New argument
            force_num_workers=force_num_workers
        )

        if final_results_agg:
            row_data = {
                **current_params_from_json, # Input parameters
                'num_train_episodes': num_train_episodes_for_this_run,
                'num_eval_episodes': num_eval_episodes_after_training,
                'samples_per_param_set': samples_per_param_set,
                **final_results_agg # Aggregated results including eval metrics
            }
            row_data['os_cpu_count'] = final_results_agg['cpu_info'].get('os_cpu_count')
            row_data['available_processors'] = final_results_agg['cpu_info'].get('available_processors')
            if 'cpu_info' in row_data: del row_data['cpu_info']

            all_run_results_for_csv.append(row_data)

            print(f"Results for set {run_idx + 1}:")
            print(f"  Eval Mean Reward: {row_data.get('eval_mean_reward', 'N/A'):.2f} ± {row_data.get('eval_std_reward', 'N/A'):.2f}")
            print(f"  Eval Mean Success Rate: {row_data.get('eval_mean_success_rate', 'N/A'):.2%}")
            print(f"  Eval Mean Steps if Successful: {row_data.get('eval_mean_steps_if_successful', 'N/A'):.2f}")
            print(f"  Mean Training Time / sample: {row_data.get('mean_training_time_per_sample', 'N/A'):.2f}s")
        else:
            print(f"Training/Evaluation failed for set {run_idx + 1}, params: {current_params_from_json}")

    if all_run_results_for_csv:
        print(f"\nWriting {len(all_run_results_for_csv)} results to {csv_file_name}...")
        try:
            with open(csv_file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_run_results_for_csv)
            print(f"Successfully wrote results to {csv_file_name}")
        except IOError:
            print(f"Error writing to CSV file {csv_file_name}.")
    else:
        print("No results to write to CSV.")

    print("\nParameter sweep finished.")
