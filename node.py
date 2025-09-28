from typing import Optional, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import copy

class FederatedNode:
    """
    A single federated learning client node that holds private data
    and trains a local Logistic Regression model.
    """

    def __init__(self, node_id: int, X_local_scaled: np.ndarray, y_local: np.ndarray, max_iter: int = 1000):
        self.node_id = node_id
        self.X_local_scaled = X_local_scaled
        self.y_local = y_local
        self.model = LogisticRegression(random_state=42, max_iter=max_iter)
        self.is_active = True
        self.local_accuracy = 0.0

    def train_local_model(self, global_weights: Optional[Dict] = None) -> Optional[Dict]:
        """
        Train local model on node's private (already scaled) data.
        Optionally receives global_weights (a dict with 'coef_', 'intercept_', 'classes_').

        Returns:
            dict with 'coef_', 'intercept_', 'classes_', 'data_size' or None if inactive.
        """
        if not self.is_active:
            return None

        # If global_weights provided, we set attributes on the local model.
        # Note: sklearn LogisticRegression.fit() will overwrite coef_ when called.
        if global_weights is not None:
            try:
                self.model.coef_ = global_weights['coef_'].copy()
                self.model.intercept_ = global_weights['intercept_'].copy()
                self.model.classes_ = global_weights['classes_'].copy()
            except Exception:
                # In case shapes mismatch or attributes missing, ignore and proceed
                pass

        # Train from scratch on local data (acceptable for this simple simulation)
        self.model.fit(self.X_local_scaled, self.y_local)

        # Evaluate on local data (training accuracy)
        preds = self.model.predict(self.X_local_scaled)
        self.local_accuracy = float(accuracy_score(self.y_local, preds))

        return {
            'coef_': copy.deepcopy(self.model.coef_),
            'intercept_': copy.deepcopy(self.model.intercept_),
            'classes_': copy.deepcopy(self.model.classes_),
            'data_size': int(len(self.y_local))
        }

    def set_active(self, active: bool):
        """Mark node active or inactive to simulate availability."""
        self.is_active = active
