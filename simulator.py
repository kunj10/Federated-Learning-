from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from node import FederatedNode
from utils import set_seed, setup_logger
import config

logger = setup_logger()

class FederatedLearningSimulator:
    """
    Orchestrates the federated learning simulation:
    - prepares data (IID or non-IID)
    - runs rounds with local training + FedAvg aggregation
    - simulates node failures and recoveries
    - evaluates aggregated global model against a held-out test set
    """

    def __init__(self, n_nodes: int = config.N_NODES, failure_rate: float = config.FAILURE_RATE, non_iid: bool = config.NON_IID):
        set_seed(config.RANDOM_SEED)
        self.n_nodes = n_nodes
        self.failure_rate = failure_rate
        self.nodes: List[FederatedNode] = []
        self.global_weights: Optional[Dict] = None
        self.round_accuracies: List[float] = []
        self.round_logs: List[Dict] = []
        self.non_iid = non_iid

        # placeholders for data
        self.X_test = None
        self.y_test = None
        self.X_full_train = None
        self.y_full_train = None
        self.global_scaler: Optional[StandardScaler] = None

        # Prepare dataset and nodes
        self._prepare_data()

    def _prepare_data(self):
        """Load Iris dataset, split into train/test, scale globally, and partition among nodes."""
        logger.info("Preparing dataset (Iris) and splitting among nodes")

        iris = load_iris()
        X, y = iris.data, iris.target

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
        )

        # Fit a global scaler on the training set (consistent preprocessing)
        self.global_scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.global_scaler.transform(X_train)
        X_test_scaled = self.global_scaler.transform(X_test)

        self.X_test = X_test_scaled
        self.y_test = y_test
        self.X_full_train = X_train_scaled
        self.y_full_train = y_train

        logger.info(f"Train samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")
        logger.info(f"Distributing training data among {self.n_nodes} nodes (non_iid={self.non_iid})")

        # Partition training data indices among nodes
        if self.non_iid:
            # Non-IID: group by label then split so nodes get contiguous label blocks
            sorted_idx = np.argsort(y_train)
            node_partitions = np.array_split(sorted_idx, self.n_nodes)
        else:
            # IID: shuffle then split
            perm = np.random.permutation(len(X_train_scaled))
            node_partitions = np.array_split(perm, self.n_nodes)

        # Create FederatedNode objects with scaled local data
        self.nodes = []
        for i in range(self.n_nodes):
            idx = node_partitions[i]
            X_local = X_train_scaled[idx]
            y_local = y_train[idx]
            node = FederatedNode(i, X_local, y_local, max_iter=config.MAX_ITER)
            self.nodes.append(node)
            logger.info(f"  Node {i}: {len(X_local)} samples, classes={np.unique(y_local)}")

    def federated_averaging(self, local_weights: List[Dict]) -> Optional[Dict]:
        """Compute FedAvg (weighted average by data size) for coef_ and intercept_."""
        if not local_weights:
            return None

        total = sum(w['data_size'] for w in local_weights)
        avg_coef = np.zeros_like(local_weights[0]['coef_'])
        avg_intercept = np.zeros_like(local_weights[0]['intercept_'])

        for w in local_weights:
            ratio = w['data_size'] / total
            avg_coef += w['coef_'] * ratio
            avg_intercept += w['intercept_'] * ratio

        return {
            'coef_': avg_coef,
            'intercept_': avg_intercept,
            'classes_': local_weights[0]['classes_']
        }

    def simulate_node_failures(self) -> List[str]:
        """
        Simulate node failures/recoveries.
        Multiple nodes may change state in a single round.
        Returns list of event strings (e.g., ["Node 1 FAILED", "Node 3 RECOVERED"])
        """
        events = []
        for node in self.nodes:
            if node.is_active and np.random.rand() < self.failure_rate:
                node.set_active(False)
                events.append(f"Node {node.node_id} FAILED")
            elif not node.is_active and np.random.rand() >= self.failure_rate:
                node.set_active(True)
                events.append(f"Node {node.node_id} RECOVERED")
        return events

    def evaluate_global_model(self, global_weights: Dict) -> float:
        """Evaluate aggregated weights on the global test set. Returns accuracy."""
        if global_weights is None:
            return 0.0

        # Create a temp LogisticRegression and set required attributes
        temp = LogisticRegression(random_state=42, max_iter=config.MAX_ITER)
       # Initialize model attributes by fitting on a subset with at least 2 classes
        unique_classes = np.unique(self.y_full_train)
        if len(unique_classes) >= 2:
            mask = np.isin(self.y_full_train, unique_classes[:2])
            temp.fit(self.X_full_train[mask], self.y_full_train[mask])

        # Overwrite coefficients with aggregated ones
        temp.coef_ = global_weights['coef_'].copy()
        temp.intercept_ = global_weights['intercept_'].copy()
        temp.classes_ = global_weights['classes_'].copy()

        # Predict on the globally scaled test set
        preds = temp.predict(self.X_test)
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(self.y_test, preds))

    def run_simulation(self, n_rounds: int = config.N_ROUNDS) -> Optional[Dict]:
        """Run the federated training simulation for n_rounds."""
        logger.info("Starting federated training simulation")
        self.global_weights = None
        self.round_accuracies = []
        self.round_logs = []

        for r in range(1, n_rounds + 1):
            logger.info(f"--- ROUND {r} ---")
            events = self.simulate_node_failures()
            if events:
                for ev in events:
                    logger.info(f"  {ev}")

            active_nodes = [n for n in self.nodes if n.is_active]
            inactive_nodes = [n for n in self.nodes if not n.is_active]
            logger.info(f"  Active nodes: {[n.node_id for n in active_nodes]}")
            if inactive_nodes:
                logger.info(f"  Inactive nodes: {[n.node_id for n in inactive_nodes]}")

            if not active_nodes:
                logger.warning("No active nodes this round. Skipping aggregation.")
                self.round_logs.append({'round': r, 'active_nodes': [], 'inactive_nodes': [n.node_id for n in inactive_nodes], 'failure_events': events, 'global_accuracy': None})
                continue

            # Local training
            local_weights = []
            for node in active_nodes:
                w = node.train_local_model(self.global_weights)
                if w:
                    local_weights.append(w)
                    logger.info(f"  Node {node.node_id}: local_accuracy={node.local_accuracy:.4f}")

            if not local_weights:
                logger.warning("No local weights collected this round. Skipping aggregation.")
                self.round_logs.append({'round': r, 'active_nodes': [n.node_id for n in active_nodes], 'inactive_nodes': [n.node_id for n in inactive_nodes], 'failure_events': events, 'global_accuracy': None})
                continue

            # Aggregation
            self.global_weights = self.federated_averaging(local_weights)
            logger.info(f"  Aggregated weights from {len(local_weights)} nodes")

            # Optional: broadcast aggregated weights to all nodes (so evaluation uses same base)
            for node in self.nodes:
                # Overwrite node.model weights with global (optional)
                try:
                    node.model.coef_ = self.global_weights['coef_'].copy()
                    node.model.intercept_ = self.global_weights['intercept_'].copy()
                    node.model.classes_ = self.global_weights['classes_'].copy()
                except Exception:
                    pass

            # Evaluate global aggregated model
            acc = self.evaluate_global_model(self.global_weights)
            self.round_accuracies.append(acc)
            logger.info(f"  Global model accuracy: {acc:.4f}")

            # Log round
            self.round_logs.append({
                'round': r,
                'active_nodes': [n.node_id for n in active_nodes],
                'inactive_nodes': [n.node_id for n in inactive_nodes],
                'failure_events': events,
                'global_accuracy': acc
            })

        logger.info("Federated training simulation finished")
        return self.global_weights

    def train_centralized_model(self) -> float:
        """Train a centralized LogisticRegression on the full training set and return test accuracy."""
        logger.info("Training centralized baseline on full data")
        scaler = self.global_scaler
        X_train = self.X_full_train
        X_test = self.X_test
        y_train = self.y_full_train
        y_test = self.y_test

        clf = LogisticRegression(random_state=42, max_iter=config.MAX_ITER)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        from sklearn.metrics import accuracy_score
        acc = float(accuracy_score(y_test, preds))
        logger.info(f"Centralized model accuracy: {acc:.4f}")
        return acc

    def compare_results(self, centralized_accuracy: float):
        """Compare federated vs centralized results and print summary."""
        logger.info("Comparing results")
        if not self.round_accuracies:
            logger.warning("No federated training results to compare.")
            return

        final_fed = self.round_accuracies[-1]
        best_fed = max(self.round_accuracies)
        logger.info(f"Centralized Accuracy: {centralized_accuracy:.4f}")
        logger.info(f"Final Federated Accuracy: {final_fed:.4f}")
        logger.info(f"Best Federated Accuracy: {best_fed:.4f}")
        logger.info(f"Difference (Final - Centralized): {final_fed - centralized_accuracy:+.4f}")
        logger.info(f"Difference (Best - Centralized): {best_fed - centralized_accuracy:+.4f}")

        if final_fed >= 0.95 * centralized_accuracy:
            logger.info("✅ Federated learning achieved comparable performance.")
        else:
            logger.info("⚠️ Federated learning shows reduced performance (expected).")

        # Summary stats
        node_failure_events = sum(bool(l.get('failure_events')) for l in self.round_logs)
        successful_rounds = sum(1 for l in self.round_logs if l.get('global_accuracy') is not None)
        logger.info(f"Training completed over {len(self.round_logs)} rounds")
        logger.info(f"Rounds with any failure/recovery events: {node_failure_events}")
        logger.info(f"Successful training rounds: {successful_rounds}")

    def plot_training_progress(self):
        """Convenience wrapper to plot results using utils.plot_accuracies."""
        try:
            from utils import plot_accuracies
            plot_accuracies(self.round_accuracies, self.round_logs, centralized_acc=None)
        except Exception as e:
            logger.error(f"Failed to plot results: {e}")
