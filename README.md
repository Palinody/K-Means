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

tested against Python's sklearn library with the following data and hyperparameters:

* data shape: 3x100,000
* num. iter: 35
* time computed by taking the average over 7 runs
* parallel tests performed with 4 threads

| n_clusters | sklearn(ms)   | c++(ser.)(ms)  |  perf. gain(sklearn/c++) |c++(par.)(ms)| perf. gain (ser./par.)|
| :---       |:---:          |:---:           |---:                      |:---:        |---:                   |
|     2      | 214 ± 1.3     |  54 ± 3.39     |3.96| 232 ± 6.32|0.23|-|
|     4      | 348 ± 3.26    |  91 ± 2.84     |3.82| 266 ± 5.07|0.34|-|
|     8      | 556 ± 7.57    | 165 ± 3.41     |3.36| 296 ± 7.59|0.56|-|
|     16     | 898 ± 19.5    | 309 ± 5.7      |2.91| 363 ± 6.64|0.85|-|
|     32     |1530 ± 16.7    | 586 ± 8.59     |2.61| 430 ± 8.15|1.36|-|
|     64     |2730 ± 15.7    |1143 ± 6.34     |2.39| 602 ±18.17|1.90|-|


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
