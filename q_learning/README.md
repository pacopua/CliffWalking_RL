## 3. Q-Learning: Estudio de Hiperparámetros

Esta sección detalla los pasos para reproducir el estudio y análisis del algoritmo Q-Learning en el entorno CliffWalking-v0 con la opción `is_slippery=True`.

### 3.1. Visión General del Estudio de Q-Learning

El objetivo es analizar cómo diferentes configuraciones de hiperparámetros fijos afectan el rendimiento de Q-Learning. Esto incluye la generación de un conjunto diverso de combinaciones de hiperparámetros, la ejecución de experimentos de entrenamiento y evaluación, y el posterior análisis de los resultados para identificar tendencias e influencias.

### 3.2. Prerrequisitos Específicos

Además de los prerrequisitos generales del proyecto, asegúrese de tener:
*   Python 3.9+ (e.g., 3.10.x)
*   Paquetes: `numpy`, `gymnasium` (e.g., 0.28.x), `numba`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
    ```bash
    pip install numpy gymnasium numba pandas matplotlib seaborn scikit-learn
    ```

### 3.3. Estructura de Directorios

```
.
├── qlearning.py           # Agente Q-Learning básico (una sola configuración)
├── genParamsQ.py          # Genera combinaciones de HPs para Q-Learning
├── qlearningstudy.py      # Ejecuta los experimentos (paralelo, Numba)
├── analysisPipeline.py    # Analiza los resultados y genera los gráficos
├── results/
│   ├── random_search_trials.json              # HPs muestreados
│   ├── training_results_with_evaluation.csv   # Resultados de los experimentos
│   ├── best.csv / best.json                   # Mejores configuraciones
│   └── plot_*.png                             # Gráficos del análisis
└── README.md
```

*(`genParamsQ.py` escribe `random_search_unique_trials.json` en el directorio actual; muévalo a `results/random_search_trials.json`, que es la ruta que lee `qlearningstudy.py`).*

### 3.4. Pasos para la Reproducción

**Paso 1: Generación de Combinaciones de Hiperparámetros**

El script `genParamsQ.py` crea un archivo JSON con combinaciones de hiperparámetros muestreadas aleatoriamente.

1.  **Configurar `genParamsQ.py`:**
    *   Defina los rangos y valores para cada hiperparámetro en `param_distributions`.
    *   Establezca `num_desired_unique_trials` (e.g., 500).
    *   Asegúrese de que `out_file` (e.g., `results/random_search_trials.json`) sea correcto.
2.  **Ejecutar:**
    ```bash
    python genParamsQ.py
    ```

**Paso 2: Ejecución de los Experimentos de Q-Learning**

El script `qlearningstudy.py` utiliza el JSON generado para entrenar y evaluar el agente Q-Learning.

1.  **Configurar `qlearningstudy.py`:**
    *   Verifique que carga el archivo JSON correcto (generado en el Paso 1).
    *   Ajuste `samples_per_param_set` (e.g., 5), `num_eval_episodes_after_training` (e.g., 100).
    *   Confirme el nombre del archivo CSV de salida (e.g., `results/training_results_with_evaluation.csv`).
2.  **Ejecutar (se recomienda redirigir la salida para ejecuciones largas):**
    ```bash
    python qlearningstudy.py > results/qlearning_run_log.txt 2>&1
    ```

**Paso 3: Análisis de Resultados**

El script `analysisPipeline.py` procesa el CSV de resultados para generar gráficos y análisis de importancia de características.

1.  **Configurar `analysisPipeline.py`:**
    *   Establezca `CSV_FILEPATH` al archivo CSV generado en el Paso 2.
    *   Defina `RESULTS_DIR` para los gráficos (e.g., `results/`).
    *   Elija `TARGET_METRIC_FOR_ANALYSIS` (e.g., `'eval_mean_steps_if_successful'`).
    *   Ajuste `MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS` si es necesario.
2.  **Ejecutar:**
    ```bash
    python analysisPipeline.py
    ```

### 3.5. Salidas Esperadas (para Q-Learning)

*   Un archivo JSON (`results/random_search_trials.json`) con las configuraciones de hiperparámetros.
*   Un archivo CSV (`results/training_results_with_evaluation.csv`) con los resultados agregados de los experimentos.
*   Una serie de gráficos PNG en `results/` mostrando:
    *   Efecto de hiperparámetros individuales sobre la métrica de rendimiento.
    *   Impacto de las estructuras de recompensa.
    *   Importancia de características según Random Forest.
    *   Las N mejores combinaciones de hiperparámetros.
*   Salida en consola del script de análisis (e.g., OOB score, importancias impresas).

Estos artefactos permitirán replicar el análisis y comprender las conclusiones extraídas sobre el rendimiento de Q-Learning bajo diferentes configuraciones.