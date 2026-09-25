"""
perceptron.py — Implementación del Perceptrón de Rosenblatt desde cero con NumPy.

El Perceptrón de Rosenblatt es un clasificador lineal binario que aprende
pesos (w) y un sesgo (b) mediante actualizaciones iterativas basadas en el
error de clasificación. Es el precursor de las redes neuronales modernas.

Ecuación fundamental de la decisión:
    z = w · x + b
    ŷ = 1 si z >= 0, de lo contrario 0

Regla de actualización (para cada muestra i con etiqueta y_i):
    Si y_i != ŷ_i:
        w  <- w  + η * (y_i - ŷ_i) * x_i
        b  <- b  + η * (y_i - ŷ_i)

Donde:
    η  (eta) = tasa de aprendizaje (learning rate)
    x_i      = vector de características de la muestra i
    y_i      = etiqueta real (0 o 1)
    ŷ_i      = predicción del modelo
"""

import numpy as np


class Perceptron:
    """Clase que implementa el Perceptrón de Rosenblatt usando álgebra lineal matricial con NumPy."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """
        Inicializa el Perceptrón con los hiperparámetros configurados.

        Parámetros:
        -----------
        learning_rate : float
            Tasa de aprendizaje (η). Controla el tamaño del paso en la
            actualización de pesos. Valores típicos: 0.01 – 0.1.
            Ecuación de actualización: Δw = η · (y_i - ŷ_i) · x_i

        n_iterations : int
            Número máximo de épocas (iteraciones completas sobre el conjunto
            de entrenamiento) durante el entrenamiento.

        Atributos internos:
        -------------------
        weights_ : np.ndarray de forma (n_features,)
            Vector de pesos aprendido durante el entrenamiento.
        bias_ : float
            Sesgo (b) aprendido durante el entrenamiento.
        errors_ : list[int]
            Historial del número de errores de clasificación por época.
            Sirve para monitorear la convergencia del modelo.
        """
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.errors_: list[int] = []

    def _unit_step_function(self, z: np.ndarray) -> np.ndarray:
        """
        Función de escalón unitario (step function) usada como función de activación.

        Matemáticamente:
            f(z) = 1  si z >= 0
            f(z) = 0  si z < 0

        Parámetros:
        -----------
        z : np.ndarray
            Valor o vector de valores de entrada (producto punto w·x + b).

        Retorna:
        --------
        np.ndarray
            1 si z >= 0, 0 en caso contrario. Misma forma que z.
        """
        return np.where(z >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Entrena el Perceptrón utilizando el algoritmo de aprendizaje iterativo.

        Algoritmo:
        ----------
        Para cada época t = 1, ..., n_iterations:
            1. Calcular la predicción: z = X · w + b
            2. Aplicar la función de escalón: ŷ = f(z)
            3. Calcular el error: errors = y - ŷ
            4. Si errors == 0 para todas las muestras, converger → romper.
            5. Actualizar pesos y sesgo para cada muestra con error:
                 w  <- w  + η · (y_i - ŷ_i) · x_i
                 b  <- b  + η · (y_i - ŷ_i)
            6. Registrar el número total de errores en la época.

        Parámetros:
        -----------
        X : np.ndarray de forma (n_samples, n_features)
            Matriz de datos de entrenamiento. Cada fila es una muestra.
        y : np.ndarray de forma (n_samples,)
            Vector de etiquetas binarias (0 o 1).

        Retorna:
        --------
        self : Perceptron
            Retorna la instancia del modelo entrenado para encadenamiento.

        Notas:
        ------
        - El método opera en modo online (actualización muestra a muestra).
        - Si los datos son linealmente separables, el algoritmo converge
          garantizadamente (Teorema de convergencia de Rosenblatt).
        """
        n_samples, n_features = X.shape

        # Inicialización de pesos y sesgo a cero
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        self.errors_ = []

        for epoch in range(self.n_iterations):
            total_errors = 0
            for i in range(n_samples):
                # Calcular la salida neta (producto punto + sesgo)
                z_i: float = np.dot(X[i], self.weights_) + self.bias_
                # Predicción usando función de escalón
                y_pred_i: int = int(self._unit_step_function(np.array([z_i]))[0])

                # Calcular el error para esta muestra
                error: int = int(y[i]) - y_pred_i

                if error != 0:
                    # Actualización de pesos: w <- w + η * error * x_i
                    self.weights_ += self.learning_rate * error * X[i]
                    # Actualización del sesgo: b <- b + η * error
                    self.bias_ += self.learning_rate * error
                    total_errors += 1

            self.errors_.append(total_errors)

            # Si no hubo errores en toda la época, el modelo ha convergido
            if total_errors == 0:
                print(f"✅ Perceptrón convergió en la época {epoch + 1}.")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones para nuevos datos usando el modelo entrenado.

        Proceso:
        --------
        1. Calcular la salida neta: z = X · w + b
        2. Aplicar la función de escalón unitario: ŷ = f(z)

        Parámetros:
        -----------
        X : np.ndarray de forma (n_samples, n_features)
            Matriz de datos de entrada.

        Retorna:
        --------
        np.ndarray de forma (n_samples,)
            Vector de predicciones binarias (0 o 1).

        Lanza:
        ------
        ValueError
            Si el modelo no ha sido entrenado aún (pesos no inicializados).
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError(
                "El modelo no ha sido entrenado. Llama a 'fit()' antes de 'predict()'."
            )

        z: np.ndarray = np.dot(X, self.weights_) + self.bias_
        return self._unit_step_function(z)
