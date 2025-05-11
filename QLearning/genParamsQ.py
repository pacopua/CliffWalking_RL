import json
import random

param_distributions = {
    "gamma":         [0.90, 0.95, 0.99],
    "alpha":         [0.05, 0.1, 0.2, 0.3, 0.5],
    "epsilon":       [0.3, 0.5, 0.7, 1.0],
    "epsilon_end":   [0.01, 0.05, 0.1],
    "num_episodes":  [1000, 2000, 4000, 8000, 12000], # Max 12k as you mentioned
    "epsilon_decay": [0.001, 0.0005, 0.0001, 0.00005],
    "rewards_tuple": [(0, -100, -1), (10, -100, -1), (100, -100, -1), (0, -100, 0)],
    "t_max":         [200]
}

num_desired_unique_trials = 500 # Your target
generated_trials_set = set() # Use a set to store frozensets of trial items for uniqueness
trials_list = []
max_attempts = num_desired_unique_trials * 5 # Stop if it's too hard to find new unique trials (e.g., space is small)
attempts = 0

while len(trials_list) < num_desired_unique_trials and attempts < max_attempts:
    attempts += 1
    trial = {}
    current_trial_items = [] # For checking uniqueness

    for param_name, values in param_distributions.items():
        chosen_value = random.choice(values)
        if param_name == "rewards_tuple":
            trial["finish_reward"] = chosen_value[0]
            trial["fall_reward"] = chosen_value[1]
            trial["step_reward"] = chosen_value[2]
            # Add to items for uniqueness check
            current_trial_items.append(("finish_reward", chosen_value[0]))
            current_trial_items.append(("fall_reward", chosen_value[1]))
            current_trial_items.append(("step_reward", chosen_value[2]))
        else:
            trial[param_name] = chosen_value
            current_trial_items.append((param_name, chosen_value))

    # Create a frozenset of items to check for uniqueness because dicts are unhashable
    # and item order in a dict might vary across Python versions for frozenset of dict.items()
    # Sorting items by key ensures consistent representation for the set
    trial_signature = frozenset(sorted(current_trial_items))

    if trial_signature not in generated_trials_set:
        generated_trials_set.add(trial_signature)
        trials_list.append(trial)

if len(trials_list) < num_desired_unique_trials:
    print(f"Warning: Only generated {len(trials_list)} unique trials after {max_attempts} attempts. The parameter space might be smaller than desired or sampling is inefficient.")

out_file = "random_search_unique_trials.json"
with open(out_file, "w") as f:
    json.dump(trials_list, f, indent=2)
print(f"Saved {len(trials_list)} unique random trials to {out_file}")