# UAE-optimizer

Machine learning–guided optimization of ultrasound-assisted extraction parameters for maximizing bioactive compound yield.

## Workflow:

1. Use [Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) (LHS) to design experiments, ensure representative data coverage.(done)
2. use [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)(k-NN) regression to model and optimize the 4D parameter space of enzyme concentration, ultrasound temperature, time, and power.()

## Dependencies

- `numpy` 
- `pandas`
- `scikit-learn` 
- `pyDOE2` 

```bash
pip install -r requirements.txt
