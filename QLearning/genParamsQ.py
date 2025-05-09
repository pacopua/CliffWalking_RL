import json
import random

param_distributions = { 
    "gamma":         [0.90, 0.95, 0.99],
    "alpha":         [0.05, 0.1, 0.2, 0.3, 0.5],
    "epsilon":       [0.3, 0.5, 0.7, 1.0],
    "epsilon_end":   [0.01, 0.05, 0.1],
    "num_episodes":  [1000, 2000, 4000, 8000, 12000],
    "epsilon_decay": [0.001, 0.0005, 0.0001, 0.00005],
    "rewards_tuple": [(0, -100, -1), (10, -100, -1), (100, -100, -1), (0, -100, 0)], 
    "t_max":         [200]
}

num_random_trials = 500 # ~ 6h
trials = []
for _ in range(num_random_trials):
    trial = {}
    for param_name, values in param_distributions.items():
        chosen_value = random.choice(values)
        if param_name == "rewards_tuple": # Handle the reward tuple
            trial["finish_reward"] = chosen_value[0]
            trial["fall_reward"] = chosen_value[1]
            trial["step_reward"] = chosen_value[2]
        else:
            trial[param_name] = chosen_value
    trials.append(trial)

out_file = "results/random_search_trials.json"
with open(out_file, "w") as f:
    json.dump(trials, f, indent=2)
print(f"Saved {len(trials)} random trials to {out_file}")
# Your main script would then load this single JSON file