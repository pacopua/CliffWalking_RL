import json
import itertools

# 1. Define all hyperparameters and their possible values
param_values = {
    "gamma":         [0.90, 0.95, 0.99],
    "alpha":         [0.1, 0.3, 0.5],
    "epsilon":       [0.1, 0.3, 0.5],
    "num_episodes":  [500, 2000, 4000],
    "epsilon_decay": [0.1, 0.001, 1e-6],
    "finish_reward": [0, 10, 100],
    "fall_reward":   [-10, -100, -1000],
    "step_reward":   [0.0, -1, -10]
}

# 2. Define groups of dependent parameters
groups = {
    "learning":       ["gamma", "alpha", "finish_reward", "fall_reward", "step_reward"],
    "exploration":    ["epsilon", "epsilon_decay", "num_episodes"]
}

# 3. Compute default (median) value for each parameter
defaults = {}
for k, vals in param_values.items():
    sorted_vals = sorted(vals)
    defaults[k] = sorted_vals[len(sorted_vals) // 2]

# 4. For each group, generate combinations of that group's params, fixing others
for group_name, keys in groups.items():
    trials = []
    # Build list of value-lists for the group
    vals_list = [param_values[k] for k in keys]
    # Iterate over all combinations in this group
    for combo in itertools.product(*vals_list):
        trial = {}
        # assign group values
        for k, v in zip(keys, combo):
            trial[k] = v
        # assign defaults for other parameters
        for other_k, default_v in defaults.items():
            if other_k not in trial:
                trial[other_k] = default_v
        trials.append(trial)

    # Save to JSON file
    out_file = f"{group_name}_group_trials.json"
    with open(out_file, "w") as f:
        json.dump(trials, f, indent=2)
    print(f"Saved {len(trials)} trials for group '{group_name}' to {out_file}")
