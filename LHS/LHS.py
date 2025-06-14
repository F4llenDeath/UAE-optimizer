import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.metrics import pairwise_distances

num_total = 100
num_vars = 4
full_raw = lhs(num_vars, samples=num_total)

bounds = np.array([
    [2, 6],       # Enzyme (%)
    [35, 70],     # Temp (°C)
    [20, 60],     # Time (min)
    [150, 350]    # Power (W)
])
scaled = full_raw * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

scaled[:, 1:] = np.round(scaled[:, 1:])
enzyme = scaled[:, 0].round(1)
scaled[:, 0] = enzyme

# Select 30 most spread-out points (MaxMin subset)
def select_maxmin_subset(points, n_subset):
    selected = [0]  # start with first point
    for _ in range(1, n_subset):
        remaining = list(set(range(len(points))) - set(selected))
        dists = pairwise_distances(points[remaining], points[selected])
        min_dists = dists.min(axis=1)
        next_point = remaining[np.argmax(min_dists)]
        selected.append(next_point)
    return selected

selected_ids = select_maxmin_subset(scaled, 30)
df_all = pd.DataFrame(scaled, columns=['Enzyme (%)', 'Temp (°C)', 'Time (min)', 'Power (W)'])
df_pretest = df_all.iloc[selected_ids]
df_followup = df_all.drop(index=selected_ids)

labels = np.full(len(df_all), 2)  
labels[selected_ids] = 1          
df_all['Set'] = labels
df_all.to_csv("lhs_labeled_100.csv", index=False)