# UAE-optimizer

Machine learning–guided optimization of ultrasound-assisted extraction parameters for maximizing bioactive compound yield.

## Workflow:

1. Use [Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) (LHS) to design experiments, ensure representative data coverage.
    1. 100 Latin Hypercube samples were generated to cover the 4D parameter space.
	2. A diverse subset of 30 points was selected using a MaxMin ([greedy space-filling heuristic](https://www.sciencedirect.com/topics/computer-science/greedy-heuristic)) algorithm for initial experiments (Set = 1), with the remaining 70 reserved for follow-up (Set = 2).
        1. MaxMin: at each step, selecting the point that maximizes the minimum distance to the already chosen points.
2. Use [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (k-NN) regression to model and optimize the 4D parameter space of enzyme concentration, ultrasound temperature, time, and power.(work in progress)

## Dependencies

- `numpy` 
- `pandas`
- `scikit-learn` 
- `pyDOE2` 

```bash
pip install -r requirements.txt
