"""
utils.py — Funciones auxiliares para el proyecto del Perceptrón de Rosenblatt.

Este módulo proporciona herramientas para:
  - Generación de datasets de prueba (lineales y no lineales).
  - Métricas de evaluación (Accuracy, Matriz de Confusión).
  - Visualización de la frontera de decisión (opcional).

Todas las implementaciones usan exclusivamente NumPy.
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Generación de Datasets
# =============================================================================

def generate_and_gate_data(n_samples: int = 200, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset para la compuerta lógica AND (linealmente separable).

    La compuerta AND devuelve 1 solo si ambas entradas son 1:
        AND(0,0)=0, AND(0,1)=0, AND(1,0)=0, AND(1,1)=1

    Se generan muestras aleatorias alrededor de los 4 vértices del espacio [0,1]².

    Parámetros:
    -----------
    n_samples : int
        Número total de muestras a generar. Debe ser par y múltiplo de 4.
    seed : int | None
        Semilla para el generador de números aleatorios (reproducibilidad).

    Retorna:
    --------
    X : np.ndarray de forma (n_samples, 2)
        Características (x1, x2).
    y : np.ndarray de forma (n_samples,)
        Etiquetas binarias (0 o 1).
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 4

    # Clase 1: ambas entradas altas → AND = 1
    class_1 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    y_1 = np.ones(n_per_class, dtype=int)

    # Clase 0: al menos una entrada baja → AND = 0
    class_0_1 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    class_0_2 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    class_0_3 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    y_0 = np.zeros(n_per_class * 3, dtype=int)

    X = np.vstack([class_1, class_0_1, class_0_2, class_0_3])
    y = np.concatenate([y_1, y_0])

    # Mezcla aleatoria
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def generate_or_gate_data(n_samples: int = 200, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset para la compuerta lógica OR (linealmente separable).

    La compuerta OR devuelve 1 si al menos una entrada es 1:
        OR(0,0)=0, OR(0,1)=1, OR(1,0)=1, OR(1,1)=1

    Parámetros:
    -----------
    n_samples : int
        Número total de muestras. Debe ser múltiplo de 4.
    seed : int | None
        Semilla para reproducibilidad.

    Retorna:
    --------
    X : np.ndarray de forma (n_samples, 2)
    y : np.ndarray de forma (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 4

    # Clase 1: al menos una entrada alta → OR = 1
    class_1_1 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    class_1_2 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    class_1_3 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    y_1 = np.ones(n_per_class * 3, dtype=int)

    # Clase 0: ambas bajas → OR = 0
    class_0 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    y_0 = np.zeros(n_per_class, dtype=int)

    X = np.vstack([class_1_1, class_1_2, class_1_3, class_0])
    y = np.concatenate([y_1, y_0])

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def generate_xor_data(n_samples: int = 200, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset para la compuerta lógica XOR (NO linealmente separable).

    La compuerta XOR devuelve 1 si las entradas son diferentes:
        XOR(0,0)=0, XOR(0,1)=1, XOR(1,0)=1, XOR(1,1)=0

    NOTA: El perceptrón simple NO puede separar linealmente este dataset.
    Esto demuestra la limitación del perceptrón de capa única.

    Parámetros:
    -----------
    n_samples : int
        Número total de muestras. Debe ser múltiplo de 4.
    seed : int | None
        Semilla para reproducibilidad.

    Retorna:
    --------
    X : np.ndarray de forma (n_samples, 2)
    y : np.ndarray de forma (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 4

    # Clase 1: entradas diferentes → XOR = 1
    class_1_1 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    class_1_2 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    y_1 = np.ones(n_per_class * 2, dtype=int)

    # Clase 0: entradas iguales → XOR = 0
    class_0_1 = np.column_stack([
        rng.uniform(0.0, 0.3, n_per_class),
        rng.uniform(0.0, 0.3, n_per_class)
    ])
    class_0_2 = np.column_stack([
        rng.uniform(0.7, 1.0, n_per_class),
        rng.uniform(0.7, 1.0, n_per_class)
    ])
    y_0 = np.zeros(n_per_class * 2, dtype=int)

    X = np.vstack([class_1_1, class_1_2, class_0_1, class_0_2])
    y = np.concatenate([y_1, y_0])

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def generate_spiral_data(n_samples: int = 300, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un dataset sintético espiral (no linealmente separable).

    Utilizado para demostrar las limitaciones del perceptrón de capa única
    frente a problemas no lineales.

    Parámetros:
    -----------
    n_samples : int
        Total de muestras divididas equitativamente entre las 2 clases.
    seed : int | None
        Semilla para reproducibilidad.

    Retorna:
    --------
    X : np.ndarray de forma (n_samples, 2)
    y : np.ndarray de forma (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    # Clase 0: espiral sentido horario
    theta_0 = np.linspace(0, 4 * np.pi, n_per_class)
    r_0 = np.linspace(0.1, 1.0, n_per_class)
    X_0 = np.column_stack([
        r_0 * np.cos(theta_0) + rng.normal(0, 0.05, n_per_class),
        r_0 * np.sin(theta_0) + rng.normal(0, 0.05, n_per_class)
    ])
    y_0 = np.zeros(n_per_class, dtype=int)

    # Clase 1: espiral sentido antihorario (desfasada)
    theta_1 = np.linspace(np.pi, 5 * np.pi, n_per_class)
    r_1 = np.linspace(0.1, 1.0, n_per_class)
    X_1 = np.column_stack([
        r_1 * np.cos(theta_1) + rng.normal(0, 0.05, n_per_class),
        r_1 * np.sin(theta_1) + rng.normal(0, 0.05, n_per_class)
    ])
    y_1 = np.ones(n_per_class, dtype=int)

    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


# =============================================================================
# Métricas de Evaluación
# =============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la precisión (accuracy) de las predicciones.

    Definición matemática:
        Accuracy = (Número de predicciones correctas) / (Total de muestras)
                 = (1/N) * Σ I(y_true_i == y_pred_i)

    Donde I(·) es la función indicadora que vale 1 si la condición es verdadera,
    y 0 en caso contrario.

    Parámetros:
    -----------
    y_true : np.ndarray de forma (n_samples,)
        Etiquetas verdaderas (0 o 1).
    y_pred : np.ndarray de forma (n_samples,)
        Etiquetas predichas por el modelo (0 o 1).

    Retorna:
    --------
    float
        Proporción de predicciones correctas en el rango [0.0, 1.0].

    Ejemplo:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy_score(y_true, y_pred)
    0.75
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return float(correct / total) if total > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de confusión para clasificación binaria.

    Estructura de la matriz 2x2:
               Predicho
               |  0  |  1  |
    Real  0 | TN  | FP  |
          1 | FN  | TP  |

    Donde:
    - TN (True Negative):  Clase 0 predicha como 0
    - FP (False Positive): Clase 0 predicha como 1
    - FN (False Negative): Clase 1 predicha como 0
    - TP (True Positive):  Clase 1 predicha como 1

    Parámetros:
    -----------
    y_true : np.ndarray de forma (n_samples,)
        Etiquetas verdaderas.
    y_pred : np.ndarray de forma (n_samples,)
        Etiquetas predichas.

    Retorna:
    --------
    np.ndarray de forma (2, 2)
        Matriz de confusión con la estructura descrita arriba.

    Ejemplo:
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0],
           [1, 2]])
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                      [fn, tp]], dtype=int)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Genera un reporte de clasificación con métricas detalladas por clase.

    Métricas calculadas por clase c:
    - Precisión (Precision):  TP_c / (TP_c + FP_c)
    - Recall (Sensibilidad):  TP_c / (TP_c + FN_c)
    - F1-Score:              2 * (Precisión * Recall) / (Precisión + Recall)
    - Soporte:               Número de muestras reales de la clase

    Parámetros:
    -----------
    y_true : np.ndarray de forma (n_samples,)
    y_pred : np.ndarray de forma (n_samples,)

    Retorna:
    --------
    str
        Cadena de texto con el reporte formateado.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Métricas para clase 0
    precision_0 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_0 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    # Métricas para clase 1
    precision_1 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall_1 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    accuracy = accuracy_score(y_true, y_pred)

    report = (
        f"               precision    recall  f1-score   support\n"
        f"\n"
        f"           0       {precision_0:.4f}    {recall_0:.4f}    {f1_0:.4f}        {tp+fn}\n"
        f"           1       {precision_1:.4f}    {recall_1:.4f}    {f1_1:.4f}        {tn+fp}\n"
        f"\n"
        f"    accuracy                           {accuracy:.4f}        {len(y_true)}\n"
        f"   macro avg       {(precision_0+precision_1)/2:.4f}    {(recall_0+recall_1)/2:.4f}    {(f1_0+f1_1)/2:.4f}        {len(y_true)}\n"
    )
    return report


# =============================================================================
# Visualización (opcional, requiere Matplotlib)
# =============================================================================

def plot_decision_boundary(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Perceptrón — Frontera de Decisión",
    show_plot: bool = True
) -> Optional[None]:
    """
    (Opcional) Visualiza la frontera de decisión del perceptrón en 2D.

    Dibuja:
    1. Los puntos de datos coloreados por clase.
    2. La recta de decisión w·x + b = 0.
    3. Las regiones de decisión sombreadas.

    Parámetros:
    -----------
    model : object
        Objeto con atributos `weights_` y `bias_` (resultado de `fit()`).
    X : np.ndarray de forma (n_samples, 2)
        Datos de entrada bidimensionales.
    y : np.ndarray de forma (n_samples,)
        Etiquetas verdaderas.
    title : str
        Título de la gráfica.
    show_plot : bool
        Si True, muestra la gráfica. Si False, la suprime.

    Retorna:
    --------
    None

    Notas:
    ------
    Esta función requiere Matplotlib. Si no está disponible, imprime un
    mensaje indicando que la visualización no se puede realizar.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Matplotlib no está instalado. Instálalo con: pip install matplotlib")
        return None

    weights = model.weights_
    bias = model.bias_

    # Crear malla de puntos
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Predecir en cada punto de la malla
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # Graficar regiones de decisión
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=0.5)

    # Graficar puntos de datos
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='k', linewidths=0.5, s=60)

    plt.xlabel("X₁", fontsize=12)
    plt.ylabel("X₂", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(*scatter.legend_elements(), title="Clase")
    plt.colorbar(scatter, label="Etiqueta")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.close()
    print("📊 Gráfica de frontera de decisión guardada como 'decision_boundary.png'")

    return None


def plot_error_history(errors: list, title: str = "Historial de Convergencia") -> Optional[None]:
    """
    (Opcional) Grafica el historial de errores por época durante el entrenamiento.

    Parámetros:
    -----------
    errors : list[int]
        Lista con el número de errores por época.
    title : str
        Título de la gráfica.

    Retorna:
    --------
    None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Matplotlib no está instalado.")
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='b')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel("Errores de Clasificación", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("error_history.png", dpi=150)
    plt.close()
    print("📊 Gráfica de historial de errores guardada como 'error_history.png'")
    return None
