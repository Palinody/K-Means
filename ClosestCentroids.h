#pragma once

#include "headers/Matrix.h"

template<typename T>
class ClosestCentroids : public Matrix<int>{
public:

    /**
      * with 2 rows we need to init the matrix with the same data on both lines
      * The first one is computed w.r.t. data and cluster. 2nd one is a copy.
      * The reason behind that is that EVERY closestCentroid is GUARANTED to
      * be modified during initialization but is NOT during the next iteration.
      * Some data that do not satisfy the if(read_val > abs_sum) criterion may
      * remain with the initialized centroids, which are wrong.
      */
	ClosestCentroids(int rows, int cols, Matrix<T> data, Matrix<T> cluster, int value = 0, int num_threads = 1) : 
        Matrix<int>(rows, cols, value, num_threads),
        _toggle{ 1 } { 
        
        initDistBuffer();
        //getClosest(data, cluster);
        //#pragma omp parallel for simd num_threads(_n_threads)
        //for(int i = 0; i < _cols; ++i){
        //    _matrix[i+_cols] = _matrix[i]; // copying row_idx 0 data into row_idx 1
        //}
    }

    ClosestCentroids(int rows, int cols, int value = 0, int num_threads = 1) : 
        Matrix<int>(rows, cols, value, num_threads){ 
        
        initDistBuffer();
    }

    ~ClosestCentroids(){ free(_distBuffer); }

    /**
     * Gets closest cluster index w.r.t. each sample
     * 
     * Dimensions:
     *      rows: 1 (2 if toggle feature activated)
     *      cols: n_samples
    */
    ClosestCentroids& getClosest(const Matrix<T>& data, const Matrix<T>& cluster){
        int n_dims = data.getRows();
        int n_clusters = cluster.getCols();

        #pragma omp parallel for collapse(1) num_threads(_n_threads)
        for(int i = 0; i < _cols; ++i){

            //Matrix<T> min_buff(1, n_clusters, -1);
            T abs_sum = 0;
            for(int d = 0; d < n_dims; ++d){
                abs_sum += std::abs(data(d, i) - cluster(d, 0));
            }
            _matrix[i+_current_row*_cols] = 0;
            (*_distBuffer)(0, i) = abs_sum;
            for(int c = 1; c < n_clusters; ++c){
                abs_sum = 0;
                for(int d = 0; d < n_dims; ++d){
                    abs_sum += std::abs(data(d, i) - cluster(d, c));
                }
                if(abs_sum < (*_distBuffer)(0, i)){
                    _matrix[i+_current_row*_cols] = c;
                    (*_distBuffer)(0, i) = abs_sum;
                }
            }
        }
        _current_row = _toggled_row;
        _toggled_row ^= _toggle;
        return *this;
    }

    /**
     * Checks whether the stopping criterion is satisfied or not.
     * If 2 consecutive closest centroids computation's modification
     * rate that is below the given threshold, we consider that 
     * KMeans has converged 
    */
    float getModifRate(){
        // stopping criterion never satisfied if we dont keep track of assigned centroids modifications
        if(_rows < 2) return 0.0f;
        //assert(_rows == 2);
        int counter = 0;
        #pragma omp parallel for num_threads(_n_threads)
        for(int i = 0; i < _cols; ++i){
            const int& a = _matrix[i];
            const int& b = _matrix[i+_cols];
            if(!(a ^ b)) ++counter;
        }
        return 1.0f - static_cast<float>(counter) / _cols;
        
    }

    int& operator()(const int& col) {
	    return _matrix[col+_current_row*_cols];
    }

    const int& operator()(const int& col) const {
	    return _matrix[col+_current_row*_cols];
    }

private:
    void initDistBuffer(){
        _distBuffer = new Matrix<T>(1, _cols, std::numeric_limits<T>::max(), _n_threads);
    }

    Matrix<T> *_distBuffer;
    // we want to store the previous state of mapped centroids, toggle: 1 to switch between rows
    int _toggle = 0;
    // we store the toggled row
    int _toggled_row = 0;
    int _current_row = 0;
};