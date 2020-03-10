# K-Means
K-Means algorithm with OpenMP

Compiled on windows 10: -std=c++14 -O3 -fopenmp

## Important changes

* upgrade gcc version (currently 4.9.2)
* use latest OpenMP features after gcc upgrade (currently OpenMP 4.0)
* make PRNG thread safe

## Less important

* stopping criterion
* mini-batch implementation

## Performance

* ~2.5x faster than Python sklearn's k_means (single threaded)

tested with:

* data size: 3x100,000
* #iter: 35

| n_clusters | sklearn (s)  | c++(serial)(s) | c++(parallel)(s)| perf. gain |
| :---       |:---:         |:---:           |:---:            |---:        |
|     2      |     -        |     -          |     -           |     -      |
|     4      |     -        |     -          |     -           |     -      |
|     8      |     -        |     -          |     -           |     -      |
|     16     |     -        |     -          |     -           |     -      |
|     32     |     -        |     -          |     -           |     -      |
|     64     |     -        |     -          |     -           |     -      |


**sklearn tested the following way with IPython Jupyter notebook**
```python
import numpy as np
from sklearn.cluster import k_means

DATASET = np.random.rand(100000, 3)
%timeit centroid, label, inertia, best_n_iter = k_means(DATASET, init='random', \
                                                precompute_distances=False, n_init=1, \
                                                n_clusters=64, n_jobs=1, \
                                                max_iter=35, tol=-1, algorithm="full", \
                                                return_n_iter=True)
```
