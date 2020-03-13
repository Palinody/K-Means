#pragma once

#include "containers/OpenMP/matrix/headers/Matrix.h"

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
        getClosest(data, cluster);
        #pragma omp parallel for simd num_threads(_n_threads)
        for(int i = 0; i < _cols; ++i){
            _matrix[i+_cols] = _matrix[i]; // copying row_idx 0 data into row_idx 1
        }
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

        #pragma omp parallel for collapse(2) num_threads(_n_threads)
        for(int c = 0; c < n_clusters; ++c){
            for(int i = 0; i < this->_cols; ++i){
                T abs_sum = 0;
                for(int d = 0; d < n_dims; ++d){
                    abs_sum += std::abs(data(d, i) - cluster(d, c));
                }
                #pragma omp read
                const T& read_val = (*this->_distBuffer)(0, i);
                
                if(read_val > abs_sum){
                    #pragma omp atomic write
                    (*this->_distBuffer)(0, i) = abs_sum;
                    #pragma omp atomic write
                    this->_matrix[i+_toggled_row*_cols] = c; // i: iterating over features
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
        assert(_rows == 2);
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
	    return this->_matrix[col+_current_row*_cols];
    }

    const int& operator()(const int& col) const {
	    return this->_matrix[col+_current_row*_cols];
    }

private:
    Matrix<T> *_distBuffer;
    // we want to store the previous state of mapped centroids, toggle: 1 to switch between rows
    int _toggle = 0;
    // we store the toggled row
    int _toggled_row = 0;
    int _current_row = 0;

    void initDistBuffer(){
        _distBuffer = new Matrix<T>(1, _cols, std::numeric_limits<float>::max(), _n_threads);
    }
};

template<typename T>
class KMeans{
public:
    KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion=true, int n_threads=1);
    ~KMeans(){ free(_dataset_to_centroids); }

    Matrix<T> getCentroid();
    Matrix<int> getDataToCentroid();

    T computeDist(const Matrix<T>& first, const Matrix<T>& second);
    void mapSampleToCentroid();
    void updateCentroids();
    void run(int max_iter, float threashold=-1);

    void print();


private:
    bool _stop_crit;
    int _n_threads;
    /**
     * number of features of the dataset (x0, x1, ..., xn)
    */
    int _dims;
    /**
     * number of training samples
    */
    int _samples;
    /**
    * desired number of clusters
    */
    int _n_clusters; 
    /**
     * K cluster centroids µ1, ..., µK. NxK matrix
     * where:
     *      N: number of dimensions
     *      K: number of classes/clusters
    */
    Matrix<T> _centroids;
    /**
     * By convention, _training_set is a NxM matrix
     * where:
     *      N: number of dimensions
     *      M: number of training samples
     * 
    */
    Matrix<T> _training_set;
    /**
     * M centroids indices mapping each training sample to a
     * corresponding cluster. 1xM matrix
     * where:
     *      M: number of samples
     *      element: index of cluster mapping 
     *               taining_index -> cluster_index
     * note: M may increase as we add new samples
     * The class doesn't support that yet
     * 
     * stopping criterion idea:
     *      keep _dataset_cluster from step t-1
     *      and check for changes.
     *      If almost no change -> stop algorithm
    */
    // Matrix<int> _dataset_to_centroids;
    ClosestCentroids<T> *_dataset_to_centroids;
};

template<typename T>
KMeans<T>::KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion, int n_threads) : 
        _training_set{ dataset },
        _n_clusters{ n_clusters },
        _stop_crit{ stop_criterion },
        _n_threads{ n_threads } {
        
    _training_set = dataset;
    _dims = dataset.getRows();
    _samples = dataset.getCols();
    _training_set.setThreads(_n_threads);
       
    Matrix<T> vMinValues = _training_set.vMin();
    Matrix<T> vMaxValues = _training_set.vMax();

    Matrix<T> new_centroids(_dims, n_clusters, UNIFORM, vMinValues, vMaxValues);
    _centroids = new_centroids;
    _n_clusters = n_clusters;
    _centroids.setThreads(_n_threads);

    if(stop_criterion){ _dataset_to_centroids = new ClosestCentroids<T>(2, _samples, _training_set, _centroids, 0, _n_threads); }
    else{ _dataset_to_centroids = new ClosestCentroids<T>(1, _samples, 0, _n_threads); }
}

template<typename T>
Matrix<T> KMeans<T>::getCentroid(){
    return _centroids;
}

template<typename T>
Matrix<int> KMeans<T>::getDataToCentroid(){
    return *static_cast<Matrix<int>* >(_dataset_to_centroids);
}

template<typename T>
void norm_1(T& elem){
    elem = abs(elem);
}

template<typename T>
void KMeans<T>::mapSampleToCentroid(){
    /**
     * TODO: extend matrix 
    */
   /* OLD METHOD
    for(int c = 0; c < _n_clusters; ++c){
        Matrix<T> curr_cluster = _centroids.getSlice(0, _dims, c, c+1);
        Matrix<T> training_set_cpy = _training_set;
        training_set_cpy.hBroadcast(curr_cluster, SUB);
        training_set_cpy.applyFunc(norm_1);
        _distBuff.insert(training_set_cpy.vSum(), c, c+1, 0, _samples);
    }
    _dataset_to_centroids = _distBuff.hMinIndex();
    */
   _dataset_to_centroids->getClosest(_training_set, _centroids);
}

template<typename T>
void KMeans<T>::updateCentroids(){
    Matrix<T> new_centroids(_dims, _n_clusters, 0, _n_threads);
    _centroids = new_centroids;
    // number of points assigned to a cluster
    Matrix<int> occurences(1, _n_clusters, 0, _n_threads);
    occurences.setThreads(1);
    _centroids.setThreads(1);

    #pragma omp parallel for num_threads(_n_threads)
    for(int i = 0; i < _samples; ++i){
        const int& k_index = (*_dataset_to_centroids)(i);
        assert(k_index != -1);
        #pragma omp simd
        for(int d = 0; d < _dims; ++d){
            _centroids(d, k_index) += _training_set(d, i);
        }
        #pragma omp atomic
        ++occurences(0, k_index);
    }

    #pragma omp parallel for num_threads(_n_threads)
    for(int c = 0; c < _n_clusters; ++c){
        if(!occurences(0, c)) continue;
        for(int d = 0; d < _dims; ++d){
            _centroids(d, c) /= occurences(0, c);
        }
    }
}

template<typename T>
void KMeans<T>::run(int max_iter, float threashold){
    //while(!_dataset_to_centroids->isEqual())

    this->mapSampleToCentroid();
    this->updateCentroids();
    if(max_iter == 1) return;
    int epoch = 1;
    float modif_rate_prev;
    float modif_rate_curr = 1;
    float inertia;
    do {
        this->mapSampleToCentroid();
        this->updateCentroids();
        modif_rate_prev = modif_rate_curr;
        modif_rate_curr = _dataset_to_centroids->getModifRate();
        inertia = modif_rate_prev - modif_rate_curr;
        //printf("%.3f %.3f\n", modif_rate_curr, (modif_rate_prev - modif_rate_curr));
        ++epoch;
    //} while(epoch < max_iter && modif_rate_curr > threashold && inertia != 0);
    } while(epoch < max_iter);
    //printf("iter number: %d\n", epoch);
}

template<typename T>
void KMeans<T>::print() {
    for(int d = 0; d < _dims; ++d){
        std::cout << "centroids_x = [";
        std::cout << _centroids.row(d) << "]" << std::endl;
    }
}
