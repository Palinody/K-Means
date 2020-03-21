# K-Means
K-Means algorithm with OpenMP

windows: g++ -std=c++17 -O3 -fopenmp
* mingw-w64: 7.0.0
* GCC 9.2.0

linux: g++-9 -std=c++17 -O3 -fopenmp

[OpenMP 5.0](https://www.openmp.org/wp-content/uploads/OpenMPRef-5.0-111802-web.pdf)

## TODO

* stopping criterion :heavy_check_mark:
* mini-batch implementation
* streams
* [algorithms](https://www.cplusplus.com/reference/algorithm/) 
* thread safe prng :heavy_check_mark:

# Performance

## Computation

tested against Python's sklearn library with the following data and hyperparameters:

* data shape: 3x100,000
* num. iter: 35
* time computed by taking the average over 7 runs
* parallel tests performed with 4 threads

| n_clusters | sklearn<br />time (ms)| c++(ser.)<br />time (ms)|  c++/sklearn<br />time ratio (%)|c++(par.)<br />time (ms)| par./ser.<br />time ratio (%)|
| :---       |:---:          |:---:           |---:                      |:---:        |---:                   |
|     2      | 214 ± 1.3     |  51 ±7.6      |23.83| 203 ±9.02|398.04|-|
|     4      | 348 ± 3.26    |  78 ±7.3      |22.41| 228 ±8.35|292.31|-|
|     8      | 556 ± 7.57    | 127 ±5.89     |22.84| 237 ±5.90|186.61|-|
|     16     | 898 ± 19.5    | 234 ±9.02     |26.06| 294 ±6.05|125.64|-|
|     32     |1530 ± 16.7    | 444 ±8.25     |29.02| 344 ±9.01|77.48|-|
|     64     |2730 ± 15.7    | 837 ±8.35     |30.66| 463 ±7.36|55.32|-|

:exclamation: faster computation doesn't imply faster convergence. No tests have been performed in this regard yet.

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

![k_means](https://user-images.githubusercontent.com/32341154/76665071-cb5b4200-6586-11ea-8810-be73367fd6c7.png)

## Convergence

...
