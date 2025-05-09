def main():
    # ... (keep your existing params and setup code) ...

    with open('fractional_results.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'gamma', 'num_trajectories', 'finish_reward', 
            'fall_reward', 'step_reward', 'epsilon', 
            'elapsed_time', 'std_training_time_per_sample',  # Added std for time
            'avg_steps', 'eval_std_steps_if_successful'
        ])

        for combo in param_combinations:
            print(f"Testing: {combo}")
            BIGstep_average = 0
            BIGtime_average = 0
            successful_steps = []
            training_times = []  # Track training times for each repetition

            for _ in range(3):  # 3 repetitions
                env = gym.make('CliffWalking-v0', is_slippery=True)
                
                # Modify rewards (unchanged)
                for s in range(env.observation_space.n):
                    for a in range(env.action_space.n):
                        new_transitions = []
                        for prob, next_s, reward, done in env.unwrapped.P[s][a]:
                            if done:
                                new_reward = combo['finish_reward']
                            elif reward == -100:
                                new_reward = combo['fall_reward']
                            else:
                                new_reward = combo['step_reward']
                            new_transitions.append((prob, next_s, new_reward, done))
                        env.unwrapped.P[s][a] = new_transitions

                agent = DirectEstimationAgent(
                    env, 
                    gamma=combo['gamma'],
                    num_trajectories=combo['num_trajectories'],
                    epsilon=combo['epsilon']
                )

                start = timeit.default_timer()
                train(agent)
                end = timeit.default_timer()
                training_time = end - start
                training_times.append(training_time)  # Store each repetition's time

                # Evaluation phase (unchanged)
                tot_steps = 0
                episode_steps = []
                successful_episode_steps = []

                for _ in range(NUM_EPISODES):
                    state, _ = env.reset()
                    steps = 0
                    is_successful = False

                    while True:
                        action = agent.select_action(state)
                        state, reward, done, _, _ = env.step(action)
                        steps += 1
                        if done:
                            is_successful = (reward == combo['finish_reward'])
                            break
                        if steps >= T_MAX:
                            break

                    episode_steps.append(steps)
                    if is_successful:
                        successful_episode_steps.append(steps)

                if successful_episode_steps:
                    BIGstep_average += np.mean(successful_episode_steps)
                    successful_steps.extend(successful_episode_steps)
                else:
                    BIGstep_average += 0

                BIGtime_average += training_time
                env.close()

            # Compute metrics
            mean_time = BIGtime_average / 3
            std_time = np.std(training_times) if len(training_times) > 1 else 0
            std_steps = np.std(successful_steps) if successful_steps else 0

            csv_writer.writerow([
                combo['gamma'],
                combo['num_trajectories'],
                combo['finish_reward'],
                combo['fall_reward'],
                combo['step_reward'],
                combo['epsilon'],
                mean_time,
                std_time,  # New: Standard deviation of training times
                BIGstep_average / 3,
                std_steps
            ])