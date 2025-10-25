from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class ClassicSVM:
    """
    Small wrapper for classical SVM to scale data before fitting/predicting.
    """
    def __init__(self, svc: SVC, scaler: Optional[StandardScaler] = None):
        self._svc = svc
        self._scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, data: np.ndarray, labels: np.ndarray) -> "ClassicSVM":
        data = np.asarray(data, dtype=float)
        y = np.asarray(labels)
        self._scaler = self._scaler.fit(data)
        x_train = self._scaler.transform(data)
        self._svc.fit(x_train, y)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        x_test = self._scaler.transform(np.asarray(data, dtype=float))
        return self._svc.predict(x_test)

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        x_test = self._scaler.transform(np.asarray(data, dtype=float))
        return self._svc.decision_function(x_test)
