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

* ~1.02x slower than Python scikitlearn's k_means (single threaded)
test: 100,000 3D data samples, 31 iter, 4 clusters 
