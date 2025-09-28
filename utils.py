from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import logging
import random

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import numpy as _np
    _np.random.seed(seed)
    random.seed(seed)


def plot_accuracies(round_accuracies: List[float], round_logs: List[Dict], centralized_acc: float = None):
    """Plot global model accuracy over rounds and mark failure/recovery events."""

    if not round_accuracies:
        print("No training data to plot")
        return

    rounds = list(range(1, len(round_accuracies) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, round_accuracies, marker='o', linewidth=2, label='Federated Global Accuracy')

    if centralized_acc is not None:
        plt.hlines(centralized_acc, 1, len(round_accuracies), linestyles='dashed', label='Centralized Baseline')

    # Mark failure/recovery events
    first_event_label = True
    for log in round_logs:
        events = log.get('failure_events', [])
        for ev in events:
            if 'FAILED' in ev:
                label = 'Node Failure' if first_event_label else None
                plt.scatter(log['round'], round_accuracies[log['round'] - 1], color='red', marker='x', s=100, label=label, zorder=5)
                first_event_label = False
            elif 'RECOVERED' in ev:
                label = 'Node Recovery' if first_event_label else None
                plt.scatter(log['round'], round_accuracies[log['round'] - 1], color='green', marker='+', s=100, label=label, zorder=5)
                first_event_label = False

    plt.xlabel('Training Round')
    plt.ylabel('Global Model Accuracy')
    plt.title('Federated Learning: Global Accuracy per Round')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def setup_logger(name: str = "fl_sim", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a simple logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-6s | %(message)s", datefmt="%H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger
