# generate_and_filter.py

import json
import itertools
import random
import argparse
import os

# 1. Definición de espacios de parámetros por algoritmo
param_spaces = {
    "q_learning": {
        "gamma":     [0.90, 0.95, 0.99],
        "alpha":     [0.1, 0.3, 0.5],
        "epsilon":   [0.1, 0.3, 0.5],
        "t_max":     [100, 200],
        "num_episodes": [500, 2000, 4000],
        "epsilon_decay": [0.1, 0.001, 1e-6],
        "finish_reward": [0, 10, 100],
        "fall_reward": [-10, -100, -1000],
        "step_reward": [0.0, -1, -10],
    },
    "value_iteration": {
        "gamma":    [0.90, 0.95, 0.99],
        "convergencia":  [1, 0.01, 1e-4, 1e-6],
        "finish_reward": [0, 10, 100],
        "fall_reward": [-10, -100, -1000],
        "step_reward": [0.0, -1, -10],
    },
    "model_based": {
        "gamma":         [0.90, 0.95, 0.99],
        "num_trayectorias": [1000, 10000, 100000],
        "finish_reward": [0, 10, 100],
        "fall_reward": [-10, -100, -1000],
        "step_reward": [0.0, -1, -10],
    }
}

SEED = 42
random.seed(SEED)

def generate_shared_trials():
    """Genera la lista completa de trials, cada uno con su clave 'algorithm'."""
    trials = []
    for algo, space in param_spaces.items():
        keys = list(space.keys())
        for values in itertools.product(*(space[k] for k in keys)):
            trial = {"algorithm": algo}
            trial.update(dict(zip(keys, values)))
            trials.append(trial)
    random.shuffle(trials)
    return trials

def filter_by_algorithm(trials, algo_name):
    """Devuelve sólo los trials cuya clave 'algorithm' coincide con algo_name."""
    return [t for t in trials if t.get("algorithm") == algo_name]

def main():
    p = argparse.ArgumentParser(
        description="Genera 'shared_trials.json' y/o filtra por algoritmo"
    )
    p.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=list(param_spaces.keys()) + ["shared"],
        help="Nombre de algoritmo (q_learning, value_iteration, model_based) o 'shared' para exportar todo"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida (por defecto: '<algo>_trials.json' o 'shared_trials.json')"
    )
    args = p.parse_args()

    # Generar todas las combinaciones
    trials = generate_shared_trials()

    # Determinar qué exportar
    if args.algo is None or args.algo == "shared":
        export_trials = trials
        default_name = "shared_trials.json"
    else:
        export_trials = filter_by_algorithm(trials, args.algo)
        default_name = f"{args.algo}_trials.json"

    out_path = args.output or default_name

    # Escribir JSON
    with open(out_path, "w") as f:
        json.dump(export_trials, f, indent=2)
    print(f"Saved {len(export_trials)} trials to '{out_path}'")

if __name__ == "__main__":
    main()
