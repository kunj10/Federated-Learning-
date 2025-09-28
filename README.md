# 🌐 Federated Learning Simulator — Logistic Regression on Iris

A compact, hands-on demo of **Federated Learning (FL)**, showing how multiple nodes train local models, aggregate them via **Federated Averaging (FedAvg)**, and handle real-world challenges like node failures and recoveries.

---

## 🚀 Project Overview

This project is a minimal yet complete simulation of Federated Learning built on the classic **Iris dataset**. It demonstrates the core principles of FL in a friendly, modular way:

* Train local `LogisticRegression` models on private data at multiple simulated clients (nodes).
* Use **FedAvg** to aggregate parameters across nodes.
* Simulate node failures and recoveries to reflect real-world unreliability.
* Compare the federated model with a centralized baseline trained on the full dataset.
* Optionally create non-IID splits to mimic heterogeneous client distributions.
* Visualize accuracy trends and failure events with intuitive plots.

This repository is structured to feel like a real portfolio project, ready for sharing with teammates or using in presentations.

---

## ✨ Key Features

* **Clean FL simulation:** Clearly separates clients and server logic.
* **FedAvg aggregation:** Weighted averaging using local dataset sizes.
* **Node failure & recovery:** Random per-round events to test robustness.
* **Centralized baseline:** Train on the full dataset for comparison.
* **Global preprocessing:** Ensures consistent scaling across nodes.
* **Non-IID splits:** Simulate realistic heterogeneous distributions.
* **Accuracy visualization:** Plots global performance across rounds with annotations for node failures.
* **Modular architecture:** Easy to extend and experiment with.

---

## 🏗 Technical Architecture

### Project Structure

```
federated_learning/
│
├─ main.py             # Entry point to run the simulation
├─ config.py           # Set parameters: nodes, rounds, failure rate, non-IID toggle
├─ node.py             # Defines FederatedNode class (local model + training)
├─ simulator.py        # Core simulator: data prep, FedAvg, failure logic, evaluation
├─ utils.py            # Helper functions: logging, plotting, seeding
├─ requirements.txt    # Minimal dependencies
```

### Data Flow Overview

1. **Data Preparation**

   * Load Iris dataset → split train/test → fit global scaler → distribute training set across nodes (IID or non-IID).

2. **Federated Training (per round)**

   * Simulate node failures and recoveries.
   * Active nodes train local models and send back weights + local dataset size.
   * Server performs FedAvg to compute global weights.
   * Evaluate global model on the scaled test set.
   * Log round stats (accuracy, active nodes, failures).

3. **Centralized Comparison**

   * Train a single logistic regression on the full dataset.
   * Compare federated vs centralized accuracy and visualize trends.

**Why FedAvg?**
FedAvg computes a weighted average of client parameters using the number of samples per client as weights. It is the standard baseline aggregation technique in Federated Learning and is especially effective for small-scale simulations like this.

---

## 🛠 Installation

### Option A — Using `uv` (if installed)

```bash
pip install uv

cd federated_learning
uv venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\Activate.ps1     # Windows
uv pip install -r requirements.txt
uv run main.py
```

### Option B — Standard `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
python main.py
```

### Dependencies

* numpy
* scikit-learn
* matplotlib

---

## 🎯 Usage Guide

Run the simulation:

```bash
python main.py
```

### Customize via `config.py`:

| Parameter    | Description                                |
| ------------ | ------------------------------------------ |
| N_NODES      | Number of nodes/clients                    |
| N_ROUNDS     | Number of FL rounds                        |
| FAILURE_RATE | Node failure probability per round         |
| NON_IID      | Use non-IID data distribution (True/False) |
| PLOT_RESULTS | Display accuracy plots                     |

**Example: Stronger failures over 20 rounds**

```python
# config.py
N_NODES = 3
N_ROUNDS = 20
FAILURE_RATE = 0.4
```

**Output:**

* Logs showing active/inactive nodes per round
* Local accuracies and FedAvg aggregation events
* Global accuracy trends
* Optional plot annotated with failures

---

## 📊 Visualization

The project includes **matplotlib plots** to visualize:

* Global accuracy across rounds
* Failed node events
* Comparison between federated and centralized models

**Example plot idea:**
A line for global accuracy with red dots marking failed nodes.

---

## 🤝 Contribution Guidelines

We welcome contributions!

1. Fork & branch: `feature/your-idea`
2. Maintain consistent code style & add docstrings
3. Add tests & update README if needed
4. Open a pull request with clear description & expected output

**Ideas:**

* Implement `SGDClassifier` + `partial_fit` for warm-start updates
* Add CSV logging for experiments
* Add CLI (argparse) for runtime configuration

---

## 📄 License

MIT License — free for educational or personal use. Add a LICENSE file if sharing publicly.

---

## 💡 Practical Use Cases

* Teach federated learning basics to students
* Explore impact of client heterogeneity on global performance
* Test robustness of aggregation strategies under node failures

---

## ✨ Why This Project?

This project provides a controlled, hands-on environment to explore real FL questions:

* How to combine models trained on private data
* Effects of offline clients
* Performance comparison: federated vs centralized models

It’s simple, modular, and perfect for experimentation or teaching.

---

