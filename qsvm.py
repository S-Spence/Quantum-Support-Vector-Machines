import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, Optional

from sympy import ZZ

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics.pairwise import rbf_kernel


class FidelityKernel:
    """
    Implements a fidelity based quantum kernel
    K(x,y)=|<phi(x)|phi(y)>|^2 via Qiskit FidelityQuantumKernel."""
    def __init__(self, feature_map: QuantumCircuit):
        self.feature_map = feature_map
        self._kernel = None

    def fit(self, data: np.ndarray):
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        self._kernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=fidelity)
        return self._kernel.evaluate(data, data)

    def test(self, data: np.ndarray, train_data: np.ndarray):
        # fit data if self._kernel is None
        if self._kernel is None:
            self.fit(train_data)
        return self._kernel.evaluate(data, train_data)

class ProjectedKernel:
    """
    Implements a Projected Quantum Kernel that projects the quantum state to a classical state
    before passing to an RBF SVM.
    """
    def __init__(self, feature_map: QuantumCircuit, gamma: float = 1.0):
        self.feature_map = feature_map
        self.gamma = gamma
        self._z_train: Optional[np.ndarray] = None

    @staticmethod
    def _pauli_z_expectations(probs: np.ndarray, n_qubits: int):
        """
        Compute single- and pairwise Pauli-Z expectations from a probability vector.
        """
        idx = np.arange(2**n_qubits)
        # single-qubit <Z_i>
        z = np.zeros(n_qubits)
        signs = []
        for i in range(n_qubits):
            mask = 1 << (n_qubits - 1 - i)
            s_i = np.where((idx & mask) == 0, 1.0, -1.0) # +1 on |0>, -1 on |1>
            signs.append(s_i)
            z[i] = np.dot(s_i, probs)
        # pairwise <Z_i Z_j>
        zz = []
        for i in range(n_qubits):
            s_i = signs[i]
            for j in range(i + 1, n_qubits):
                zz.append(np.dot(s_i * signs[j], probs))
        zz = np.asarray(zz, dtype=float)
        return z, zz

    def _features(self, data: np.ndarray) -> np.ndarray:
        n_qubits = self.feature_map.num_qubits
        out = []
        for x in data:
            circuit = self.feature_map.assign_parameters(x, inplace=False)
            state_vector = Statevector.from_instruction(circuit)
            probs = np.abs(state_vector.data)**2
            z, zz = self._pauli_z_expectations(probs, n_qubits)
            out.append(np.concatenate([z, zz]))
        return np.vstack(out)

    def fit(self, data: np.ndarray):
        self._z_train = self._features(data)
        return rbf_kernel(self._z_train, self._z_train, gamma=self.gamma)

    def test(self, data: np.ndarray, train_data: np.ndarray):
        # fit data if self._z_train is None
        if self._z_train is None:
            self.fit(train_data)
        z_test = self._features(data)
        return rbf_kernel(z_test, self._z_train, gamma=self.gamma)
    
class QuantumSVM:
    """
    A Quantum SVM classifier using a fidelity-based quantum kernel and SVC.

    Params:
      - feature_map: (QuantumCircuit), ex: ZZFeatureMap.
      - C: Regularization parameter for SVC.
      - scale_range: Tuple for MinMaxScaler feature scaling.
      - seed: Random seed for SVC.
    """

    def __init__(self, kernel: Callable, C: float = 1.0, scale_range=(0.0, 1.0), seed: int | None = 42):
        self.kernel = kernel
        self.C = C
        self.scale_range = scale_range
        self.seed = seed

        self._scaler = None
        self._clf = None
        self._x_train = None

    def fit(self, data: np.ndarray, labels: np.ndarray):
        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels)

        # confirm feature map and data dimensions match for fidelity kernel
        if hasattr(self.kernel, "feature_map") and hasattr(self.kernel.feature_map, "num_qubits"):
            if self.kernel.feature_map.num_qubits != data.shape[1]:
                raise ValueError("feature_map qubits != data features")

        self._scaler = MinMaxScaler(feature_range=self.scale_range).fit(data)
        x_train = self._scaler.transform(data)

        k_train = self.kernel.fit(x_train)
        self._clf = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self._clf.fit(k_train, labels)
        self._x_train = x_train
        return self

    def predict(self, data: np.ndarray, batch_size:int = 1024) -> np.ndarray:
        if self._clf is None:
            raise ValueError("QuantumSVM must be fit() before predict().")
        x_test = self._scaler.transform(np.asarray(data, dtype=float))
        out = np.empty(len(x_test), dtype=int)
        for start in range(0, len(x_test), batch_size):
            stop = min(start + batch_size, len(x_test))
            k = self.kernel.test(x_test[start:stop], self._x_train)  # (b x N)
            out[start:stop] = self._clf.predict(k)
        return out

    def decision_function(self, data: np.ndarray, batch_size:int = 1024) -> np.ndarray:
        if self._clf is None:
            raise ValueError("QuantumSVM must be fit() before decision_function().")
        data = self._scaler.transform(np.asarray(data, dtype=float))
        scores = np.empty(len(data), dtype=float)
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            k = self.kernel.test(data[start:stop], self._x_train)
            scores[start:stop] = self._clf.decision_function(k)
        return scores
