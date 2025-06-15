import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from itertools import product

# load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path) as f:
    cfg = json.load(f)
parameters = cfg["parameters"]
bounds = np.array(list(parameters.values()))
column_names = list(parameters.keys())
num_total = cfg["total_samples"]
pretest_size = cfg["pretest_size"]

# boundary points
boundary_raw = np.array(list(product(*bounds)))

# lhs generate rest points
num_vars = bounds.shape[0]
num_lhs_needed = num_total - len(boundary_raw)
lhs_raw = lhs(num_vars, samples=num_lhs_needed)

lhs_scaled = lhs_raw * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

lhs_scaled[:, 1:] = np.round(lhs_scaled[:, 1:])
lhs_scaled[:, 0] = lhs_scaled[:, 0].round(1)
boundary_raw[:, 1:] = np.round(boundary_raw[:, 1:])
boundary_raw[:, 0] = boundary_raw[:, 0].round(1)
scaled = np.vstack([boundary_raw, lhs_scaled])

# use MaxMin to select rest points from lhs only
def select_maxmin_subset(points, n_subset, init_points=None):
    if not init_points: 
        selected = [0]
    else:
        selected = list(init_points)
    while len(selected) < n_subset:
        remaining = list(set(range(len(points))) - set(selected))
        dists = pairwise_distances(points[remaining], points[selected])
        min_dists = dists.min(axis=1)
        next_point = remaining[np.argmax(min_dists)]
        selected.append(next_point)
    return selected

lhs_df = pd.DataFrame(lhs_scaled, columns=column_names)
lhs_points = lhs_df.values
selected_count = pretest_size - len(boundary_raw)
selected_lhs_ids = select_maxmin_subset(lhs_points, selected_count, [])

pretest = np.vstack([boundary_raw, lhs_scaled[selected_lhs_ids]])
followup_ids = list(set(range(len(lhs_scaled))) - set(selected_lhs_ids))
followup = lhs_scaled[followup_ids]

df_pretest = pd.DataFrame(pretest, columns=column_names)
df_pretest['Set'] = 1
df_followup = pd.DataFrame(followup, columns=column_names)
df_followup['Set'] = 2
df_all = pd.concat([df_pretest, df_followup], ignore_index=True)

# pairplot visualize
df_all.to_csv("lhs_labeled_100.csv", index=False)
g = sns.pairplot(df_all, vars=column_names, hue='Set')
g.savefig("lhs_pairplot.png", dpi=300, bbox_inches="tight")
plt.show()
