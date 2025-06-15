# UAE-optimizer

Machine learning–guided optimization of ultrasound-assisted extraction parameters for maximizing bioactive compound yield.

## Workflow

1. Use [Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) (LHS) to design experiments, ensure representative data coverage. The generation is fully data driven and parameterized through `config.json`.
    1. List each variable with its `[min, max]` range in `config.json`.
    2. A number of `total_samples` Latin Hypercube samples were generated to cover the parameter space.
    3. Generate all $2^n$ boundary combinations.
    4. Construct a Latin Hypercube Sampling (LHS) design in `n` dimensions.
	5. A diverse subset with a number of `pretest_size` points was selected using a MaxMin ([greedy space-filling heuristic](https://www.sciencedirect.com/topics/computer-science/greedy-heuristic)) algorithm for initial experiments (Set = 1), with the remaining reserved for follow-up (Set = 2).
        1. MaxMin: at each step, selecting the point that maximizes the minimum distance to the already chosen points.
        2. Boundary combinations are always included in Set 1.
    6. Outcome visualized with [seaborn.pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html).
2. Use [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (k-NN) regression to model and optimize the 4D parameter space of enzyme concentration, ultrasound temperature, time, and power.(work in progress)

## Dependencies

 - `numpy`
 - `pandas`
 - `pyDOE`
 - `scikit-learn`
 - `matplotlib`
 - `seaborn`

```bash
pip install -r requirements.txt
```
