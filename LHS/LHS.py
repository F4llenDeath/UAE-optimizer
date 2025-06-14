import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

bounds = np.array([
    [2, 6],       # Enzyme (%)
    [35, 70],     # Temp (°C)
    [20, 60],     # Time (min)
    [150, 350]    # Power (W)
])

# 16 boundary points
boundary_raw = np.array(list(product(*bounds)))

# lhs generate rest 84 points
num_total = 100
num_vars = 4
num_lhs_needed = num_total - len(boundary_raw)
lhs_raw = lhs(num_vars, samples=num_lhs_needed)

lhs_scaled = lhs_raw * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

lhs_scaled[:, 1:] = np.round(lhs_scaled[:, 1:])
lhs_scaled[:, 0] = lhs_scaled[:, 0].round(1)
boundary_raw[:, 1:] = np.round(boundary_raw[:, 1:])
boundary_raw[:, 0] = boundary_raw[:, 0].round(1)
scaled = np.vstack([boundary_raw, lhs_scaled])

# use MaxMin to select 14 more points from lhs only
def select_maxmin_subset(points, n_subset, init_points=None):
    if not init_points:  # covers None or empty list
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

lhs_df = pd.DataFrame(lhs_scaled, columns=['Enzyme (%)', 'Temp (°C)', 'Time (min)', 'Power (W)'])
lhs_points = lhs_df.values
selected_lhs_ids = select_maxmin_subset(lhs_points, 14, [])

pretest = np.vstack([boundary_raw, lhs_scaled[selected_lhs_ids]])
followup_ids = list(set(range(len(lhs_scaled))) - set(selected_lhs_ids))
followup = lhs_scaled[followup_ids]

df_pretest = pd.DataFrame(pretest, columns=['Enzyme (%)', 'Temp (°C)', 'Time (min)', 'Power (W)'])
df_pretest['Set'] = 1
df_followup = pd.DataFrame(followup, columns=['Enzyme (%)', 'Temp (°C)', 'Time (min)', 'Power (W)'])
df_followup['Set'] = 2
df_all = pd.concat([df_pretest, df_followup], ignore_index=True)

# pairplot visualize
df_all.to_csv("lhs_labeled_100.csv", index=False)
g = sns.pairplot(df_all, vars=['Enzyme (%)', 'Temp (°C)', 'Time (min)', 'Power (W)'], hue='Set')
g.savefig("lhs_pairplot.png", dpi=300, bbox_inches="tight")
plt.show()