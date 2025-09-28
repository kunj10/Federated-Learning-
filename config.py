# Simple configuration for the federated learning simulation

RANDOM_SEED = 42

# Simulation parameters
N_NODES = 4
N_ROUNDS = 12
FAILURE_RATE = 0.25     # probability that a node fails in a round (per-node)
TEST_SIZE = 0.30        # fraction of dataset reserved for testing

# Behavior toggles
NON_IID = False         # Set True to use non-IID split (nodes mostly have one class)
PLOT_RESULTS = True     # Whether to show/save the accuracy plot

# LogisticRegression training
MAX_ITER = 1000
