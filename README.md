# K-Means
K-Means algorithm with OpenMP

Compiled on windows 10: -std=c++14 -O3 -fopenmp

## Important changes

* upgrade gcc version (currently 4.9.2)
* use latest OpenMP features after gcc upgrade (currently OpenMP 4.0)

## Less important

* stopping criterion
* mini-batch implementation

## Performance

* ~2.4x slower than Python scikitlearn's k_means (single threaded)
test: 100,000 3D data samples, 31 iter, 8 clusters 
