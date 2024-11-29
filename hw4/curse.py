import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
sample_size = np.random.randint(100, 1000, 4)
for idx, num_samples in enumerate(sample_size):
    # TODO: Fill in a valid feature range
    feature_range = range(1,101)
    ratios = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        num_info = max(1, num_features // 2 ) #flooring to get integer, n_informative on takes integers
        numOfClasses = min(10, max(2, num_features // 10))
        X,Y  = make_classification(
            n_samples= num_samples,
            n_features= num_features,
            n_informative= num_info,
            n_classes= numOfClasses,
            n_clusters_per_class=1,
            n_redundant=0,
            n_repeated=0,
            random_state=0
        )
        #print(X)
        # TODO: Choose random query point from X
        query_pointIdx = np.random.randint(0, X.shape[0])
        query_point = X[query_pointIdx]
        
        # TODO: remove query pt from X so it isn't used in distance calculations
        querypoint = np.delete(X, query_pointIdx, axis=0)

        # TODO: Calculate distances
        distances = np.linalg.norm(querypoint - query_point, axis=1)
        if np.min(distances) == 0:
            ratio = float('inf')
        else:
            ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)
    
    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')
    
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
