from simulator import FederatedLearningSimulator
import config
from utils import setup_logger

logger = setup_logger("main")

def main():
    logger.info("Starting Federated Learning Simulation (multi-file project)")

    sim = FederatedLearningSimulator(n_nodes=config.N_NODES, failure_rate=config.FAILURE_RATE, non_iid=config.NON_IID)
    sim.run_simulation(n_rounds=config.N_ROUNDS)

    centralized_acc = sim.train_centralized_model()
    sim.compare_results(centralized_acc)

    if config.PLOT_RESULTS:
        sim.plot_training_progress()

    logger.info("Simulation complete. Check logs and plots for details.")

if __name__ == "__main__":
    main()
