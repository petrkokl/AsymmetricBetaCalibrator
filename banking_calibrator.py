import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class AsymmetricBetaCalibrator:
    def __init__(self):
        """
        Асимметричная бета-калибровка
        """
        self.a_ = None
        self.b_ = None
        self.c_ = None

    def fit(self, probs, y_true):
        """
        Обучение  с использованием функции потерь с штрафом для недооценки.

        Параметры:
        - probs: Массив предсказанных вероятностей базовой модели.
        - y_true: Метки (0 или 1).
        """
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        initial_params = np.array([0.0, 1.0, -1.0])  # Начальные значения a, b, c
        bounds = [
            (None, None),  # a: без ограничений
            (1e-5, None),  # b > 0
            (None, -1e-5),  # c < 0
        ]

        # Определение асимметричной функции потерь
        def asymmetric_log_loss(params):
            a, b, c = params
            calibrated_probs = expit(a + b * np.log(probs) + c * np.log(1 - probs))
            calibrated_probs = np.clip(calibrated_probs, epsilon, 1 - epsilon)

            # Штраф для недооценки положительных случаев
            penalty = 1 + y_true * np.sqrt(1 - calibrated_probs)
            loss = -(
                y_true * penalty * np.log(calibrated_probs)
                + (1 - y_true) * np.log(1 - calibrated_probs)
            )
            return np.mean(loss)

        # Оптимизация параметров
        result = minimize(
            asymmetric_log_loss, initial_params, method="L-BFGS-B", bounds=bounds
        )

        # Сохранение оптимизированных параметров
        self.a_, self.b_, self.c_ = result.x
        return self

    def predict_proba(self, probs):
        """
        Предсказание калиброванных вероятностей.

        Параметры:
        - probs: Массив предсказанных вероятностей базовой модели.

        Возвращает:
        - Массив откалиброванных вероятностей.
        """
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        calibrated_probs = expit(
            self.a_ + self.b_ * np.log(probs) + self.c_ * np.log(1 - probs)
        )
        return np.clip(calibrated_probs, 0.001, 1)
