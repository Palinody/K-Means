# K-Means
K-Means algorithm with [OpenMP 5.0](https://www.openmp.org/wp-content/uploads/OpenMPRef-5.0-111802-web.pdf)

Compiled with:

**windows**: g++ -std=c++17 -O3 -fopenmp
* mingw-w64: 7.0.0
* GCC 9.2.0

**linux**: g++-9 -std=c++17 -O3 -fopenmp

## TODO

* stopping criterion :heavy_check_mark:
* mini-batch implementation
* centroid initialization
* streams (.txt :heavy_check_mark:, .csv, .bin)
* [algorithms](https://www.cplusplus.com/reference/algorithm/) 
* thread safe prng :heavy_check_mark:

# Performance

tested against Python's sklearn library with the following data and hyperparameters:

* data shape: 3x100,000
* num. iter: [2, 36]
* time computed by taking the average over 7 runs
* parallel tests performed with 6 threads

## Computation time

![k-means-comp-time](https://user-images.githubusercontent.com/32341154/78424774-0e817000-7670-11ea-89fc-155254f3bd08.png)

## N. iterations w.r.t. n. cluster

![k-means-n-iters](https://user-images.githubusercontent.com/32341154/78424779-104b3380-7670-11ea-8393-a9d71d1bb6f5.png)

## Error comparison: sklearn | c++ ser. | c++ par.

![k-means-error](https://user-images.githubusercontent.com/32341154/78424776-0fb29d00-7670-11ea-9286-214f5ec720da.png)

:exclamation: c++ generally converges with **more** iteration in a **shorter** amount of time than sklearn for a similar result. The speed at which it converges can be improved with a different initialization technique and/or a better stopping criterion. The current approach is to naively initialize the centroids to random points generated from a uniform distribution bounded by the min-max values of the training set w.r.t. each dimension. :exclamation:

**sklearn tested the following way with IPython Jupyter notebook**
```python
import numpy as np
from sklearn.cluster import k_means

%timeit centroid, label, inertia, best_n_iter = k_means(DATASET, init='random', \
                                                precompute_distances=False, n_init=1, \
                                                n_clusters=n, n_jobs=1, \
                                                max_iter=400, tol=0.01, algorithm="full", \
                                                return_n_iter=True)
```
To test `k_means` with a fixed number of iterations, just set `tol = -1`. `DATASET` is a `3x100000` float datapoints generated from a c++ script. The data distribution can be found in the next section.

# Visualization

## Ground truth data

![k-means-ground-truth](https://user-images.githubusercontent.com/32341154/77862438-a10eb300-721b-11ea-9ef6-d5ae325ad15f.png)

## sklearn-cpp

sklearn             |  c++
:-------------------------:|:-------------------------:
![k-means-sklearn-2d](https://user-images.githubusercontent.com/32341154/77862427-93f1c400-721b-11ea-93d9-4e864a2cc6d7.png)  |  ![k-means-cpp-2d](https://user-images.githubusercontent.com/32341154/77862434-9a803b80-721b-11ea-9d67-454ce925d31f.png)
