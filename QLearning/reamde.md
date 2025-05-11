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

### 3.3. Estructura de Directorios Sugerida
Use code with caution.
Markdown
.
├── genParamsQ.py # Script para generar combinaciones de HPs para Q-Learning
├── qlearningstudy.py # Script principal para ejecutar experimentos de Q-Learning
├── analysisPipeline.py # Script para analizar resultados de Q-Learning
├── results/
│ ├── qlearning_random_trials.json # HPs generados para Q-Learning (ejemplo)
│ ├── qlearning_results.csv # Resultados de experimentos Q-Learning (ejemplo)
│ ├── qlearning_plots/ # Subdirectorio para gráficos de Q-Learning
│ │ ├── plot_*.png
│ └── qlearning_run_log.txt # Log de ejecución (opcional)
└── README.md
*(Ajuste los nombres de archivo y directorios en los scripts si difieren de esta estructura).*

### 3.4. Pasos para la Reproducción

**Paso 1: Generación de Combinaciones de Hiperparámetros**

El script `genParamsQ.py` crea un archivo JSON con combinaciones de hiperparámetros muestreadas aleatoriamente.

1.  **Configurar `genParamsQ.py`:**
    *   Defina los rangos y valores para cada hiperparámetro en `param_distributions`.
    *   Establezca `num_desired_unique_trials` (e.g., 500).
    *   Asegúrese de que `out_file` (e.g., `results/qlearning_random_trials.json`) sea correcto.
2.  **Ejecutar:**
    ```bash
    python genParamsQ.py
    ```

**Paso 2: Ejecución de los Experimentos de Q-Learning**

El script `qlearningstudy.py` utiliza el JSON generado para entrenar y evaluar el agente Q-Learning.

1.  **Configurar `qlearningstudy.py`:**
    *   Verifique que carga el archivo JSON correcto (generado en el Paso 1).
    *   Ajuste `samples_per_param_set` (e.g., 5), `num_eval_episodes_after_training` (e.g., 100).
    *   Confirme el nombre del archivo CSV de salida (e.g., `results/qlearning_results.csv`).
2.  **Ejecutar (se recomienda redirigir la salida para ejecuciones largas):**
    ```bash
    python qlearningstudy.py > results/qlearning_run_log.txt 2>&1
    ```

**Paso 3: Análisis de Resultados**

El script `analysisPipeline.py` procesa el CSV de resultados para generar gráficos y análisis de importancia de características.

1.  **Configurar `analysisPipeline.py`:**
    *   Establezca `CSV_FILEPATH` al archivo CSV generado en el Paso 2.
    *   Defina `RESULTS_DIR` para los gráficos (e.g., `results/qlearning_plots/`).
    *   Elija `TARGET_METRIC_FOR_ANALYSIS` (e.g., `'eval_mean_steps_if_successful'`).
    *   Ajuste `MIN_SUCCESS_RATE_FOR_STEPS_ANALYSIS` si es necesario.
2.  **Ejecutar:**
    ```bash
    python analysisPipeline.py
    ```

### 3.5. Salidas Esperadas (para Q-Learning)

*   Un archivo JSON (`results/qlearning_random_trials.json`) con las configuraciones de hiperparámetros.
*   Un archivo CSV (`results/qlearning_results.csv`) con los resultados agregados de los experimentos.
*   Una serie de gráficos PNG en `results/qlearning_plots/` mostrando:
    *   Efecto de hiperparámetros individuales sobre la métrica de rendimiento.
    *   Impacto de las estructuras de recompensa.
    *   Importancia de características según Random Forest.
    *   Las N mejores combinaciones de hiperparámetros.
*   Salida en consola del script de análisis (e.g., OOB score, importancias impresas).

Estos artefactos permitirán replicar el análisis y comprender las conclusiones extraídas sobre el rendimiento de Q-Learning bajo diferentes configuraciones.