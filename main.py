"""
main.py — Script de ejecución del proyecto Perceptrón de Rosenblatt.

Este script demuestra el ciclo completo de:
  1. Generación de datos de prueba (compuertas lógicas AND/OR/XOR).
  2. Entrenamiento del Perceptrón.
  3. Evaluación del rendimiento (accuracy, matriz de confusión, reporte).
  4. Visualización de la frontera de decisión e historial de convergencia.

Restricción: Se utiliza ÚNICAMENTe NumPy para toda la lógica del modelo.
"""

import numpy as np
from perceptron import Perceptron
from utils import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    generate_and_gate_data,
    generate_or_gate_data,
    generate_xor_data,
    generate_spiral_data,
    plot_decision_boundary,
    plot_error_history,
)


def run_experiment(data_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   learning_rate: float = 0.1, n_iterations: int = 50) -> None:
    """
    Ejecuta un experimento completo de entrenamiento y evaluación del Perceptrón.

    Pasos:
    ------
    1. Instanciar y entrenar el Perceptrón con los datos de entrenamiento.
    2. Realizar predicciones sobre el conjunto de prueba.
    3. Calcular métricas de evaluación.
    4. Imprimir resultados por consola.
    5. Generar visualizaciones (si Matplotlib está disponible).

    Parámetros:
    -----------
    data_name : str
        Nombre descriptivo del dataset (para mostrar en los reportes).
    X_train : np.ndarray de forma (n_train_samples, n_features)
        Datos de entrenamiento.
    y_train : np.ndarray de forma (n_train_samples,)
        Etiquetas de entrenamiento.
    X_test : np.ndarray de forma (n_test_samples, n_features)
        Datos de prueba.
    y_test : np.ndarray de forma (n_test_samples,)
        Etiquetas de prueba.
    learning_rate : float
        Tasa de aprendizaje para el entrenamiento.
    n_iterations : int
        Número máximo de iteraciones de entrenamiento.
    """
    print("=" * 70)
    print(f"  🧠 EXPERIMENTO: Perceptrón de Rosenblatt — {data_name.upper()}")
    print("=" * 70)

    # --- Entrenamiento ---
    print(f"\n📊 Datos de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f"📊 Datos de prueba:       {X_test.shape[0]} muestras")
    print(f"⚙️  Hiperparámetros: learning_rate={learning_rate}, n_iterations={n_iterations}")

    perceptron = Perceptron(learning_rate=learning_rate, n_iterations=n_iterations)
    perceptron.fit(X_train, y_train)

    # --- Pesos y sesgo aprendidos ---
    print(f"\n{'─' * 50}")
    print("📐 PESOS Y SESGO APRENDIDOS:")
    print(f"{'─' * 50}")
    print(f"  Pesos (w): {perceptron.weights_}")
    print(f"  Sesgo (b): {perceptron.bias_:.6f}")
    print(f"  Ecuación de decisión: {perceptron.weights_[0]:.4f}·x₁ + {perceptron.weights_[1]:.4f}·x₂ + {perceptron.bias_:.4f} = 0")

    # --- Predicciones ---
    y_train_pred = perceptron.predict(X_train)
    y_test_pred = perceptron.predict(X_test)

    # --- Métricas en Entrenamiento ---
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    print(f"\n{'─' * 50}")
    print(f"📈 RESULTADOS EN ENTRENAMIENTO:")
    print(f"{'─' * 50}")
    print(f"  Precisión (Accuracy):  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"\n  Matriz de Confusión:")
    print(f"    {train_cm[0][0]:4d}  {train_cm[0][1]:4d}")
    print(f"    {train_cm[1][0]:4d}  {train_cm[1][1]:4d}")

    # --- Métricas en Prueba ---
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n{'─' * 50}")
    print(f"📈 RESULTADOS EN PRUEBA (TEST):")
    print(f"{'─' * 50}")
    print(f"  Precisión (Accuracy):  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\n  Matriz de Confusión:")
    print(f"    {test_cm[0][0]:4d}  {test_cm[0][1]:4d}")
    print(f"    {test_cm[1][0]:4d}  {test_cm[1][1]:4d}")
    print(f"\n  Reporte de Clasificación:\n{classification_report(y_test, y_test_pred)}")

    # --- Historial de convergencia ---
    print(f"\n{'─' * 50}")
    print(f"📉 HISTORIAL DE CONVERGENCIA (Errores por Época):")
    print(f"{'─' * 50}")
    for epoch, err in enumerate(perceptron.errors_, 1):
        status = "✅ Convergido" if err == 0 else "⚠️  En entrenamiento"
        print(f"  Época {epoch:3d}: {err:3d} errores  —  {status}")

    # --- Visualizaciones ---
    try:
        plot_decision_boundary(
            perceptron, np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]),
            title=f"Perceptrón — {data_name} (frontera de decisión)"
        )
        plot_error_history(perceptron.errors_, title=f"Convergencia — {data_name}")
    except Exception as e:
        print(f"\n⚠️  No se pudieron generar las visualizaciones: {e}")

    print(f"\n{'=' * 70}")
    print(f"  ✅ EXPERIMENTO {data_name.upper()} COMPLETADO")
    print(f"{'=' * 70}\n")


def main() -> None:
    """
    Punto de entrada principal del programa.

    Ejecuta tres experimentos:
    1. Compuerta AND (linealmente separable → el perceptrón converge).
    2. Compuerta OR (linealmente separable → el perceptrón converge).
    3. Compuerta XOR (NO linealmente separable → el perceptrón NO converge).

    También genera un dataset espiral para demostrar limitaciones adicionales.
    """
    print("\n" + "=" * 70)
    print("  🧠 PERCEPTRÓN DE ROSENBLATT — IMPLEMENTACIÓN COMPLETA")
    print("  📐 Solo NumPy — Sin frameworks de IA externos")
    print("=" * 70 + "\n")

    # =====================================================================
    # EXPERIMENTO 1: Compuerta AND
    # =====================================================================
    print("\n🔬 Experimento 1: Compuerta Lógica AND (linealmente separable)")
    X_and, y_and = generate_and_gate_data(n_samples=200, seed=42)

    # División train/test (80/20)
    split_idx = int(0.8 * len(X_and))
    X_and_train, X_and_test = X_and[:split_idx], X_and[split_idx:]
    y_and_train, y_and_test = y_and[:split_idx], y_and[split_idx:]

    run_experiment(
        data_name="AND Gate",
        X_train=X_and_train, y_train=y_and_train,
        X_test=X_and_test, y_test=y_and_test,
        learning_rate=0.1, n_iterations=50
    )

    # =====================================================================
    # EXPERIMENTO 2: Compuerta OR
    # =====================================================================
    print("\n🔬 Experimento 2: Compuerta Lógica OR (linealmente separable)")
    X_or, y_or = generate_or_gate_data(n_samples=200, seed=42)

    split_idx = int(0.8 * len(X_or))
    X_or_train, X_or_test = X_or[:split_idx], X_or[split_idx:]
    y_or_train, y_or_test = y_or[:split_idx], y_or[split_idx:]

    run_experiment(
        data_name="OR Gate",
        X_train=X_or_train, y_train=y_or_train,
        X_test=X_or_test, y_test=y_or_test,
        learning_rate=0.1, n_iterations=50
    )

    # =====================================================================
    # EXPERIMENTO 3: Compuerta XOR
    # =====================================================================
    print("\n🔬 Experimento 3: Compuerta Lógica XOR (NO linealmente separable)")
    print("⚠️  NOTA: El perceptrón de capa simple NO puede resolver XOR.")
    print("      Se espera que NO converja y que muestre errores persistentes.\n")
    X_xor, y_xor = generate_xor_data(n_samples=200, seed=42)

    split_idx = int(0.8 * len(X_xor))
    X_xor_train, X_xor_test = X_xor[:split_idx], X_xor[split_idx:]
    y_xor_train, y_xor_test = y_xor[:split_idx], y_xor[split_idx:]

    run_experiment(
        data_name="XOR Gate",
        X_train=X_xor_train, y_train=y_xor_train,
        X_test=X_xor_test, y_test=y_xor_test,
        learning_rate=0.01, n_iterations=100
    )

    # =====================================================================
    # EXPERIMENTO 4: Dataset Espiral
    # =====================================================================
    print("\n🔬 Experimento 4: Dataset Espiral (no linealmente separable)")
    print("⚠️  NOTA: Demuestra la limitación del perceptrón con datos no lineales.\n")
    X_spiral, y_spiral = generate_spiral_data(n_samples=300, seed=42)

    split_idx = int(0.8 * len(X_spiral))
    X_spiral_train, X_spiral_test = X_spiral[:split_idx], X_spiral[split_idx:]
    y_spiral_train, y_spiral_test = y_spiral[:split_idx], y_spiral[split_idx:]

    run_experiment(
        data_name="Spiral Dataset",
        X_train=X_spiral_train, y_train=y_spiral_train,
        X_test=X_spiral_test, y_test=y_spiral_test,
        learning_rate=0.01, n_iterations=100
    )

    print("\n🎯 RESUMEN FINAL")
    print("─" * 70)
    print("El perceptrón de Rosenblatt demuestra:")
    print("  ✅ Convergencia garantizada con datos linealmente separables (AND, OR)")
    print("  ❌ Incapacidad para datos no linealmente separables (XOR, Espiral)")
    print("  📐 Toda la implementación usa exclusivamente álgebra lineal NumPy")
    print("─" * 70)


if __name__ == "__main__":
    main()
