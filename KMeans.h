#pragma once

#include "containers/OpenMP/matrix/headers/Matrix.h"

template<typename T>
class ClosestCentroids : public Matrix<int>{
public:
    ClosestCentroids(int num_threads = 1) : 
        Matrix<int>(num_threads){
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
     *      rows: 1
     *      cols: n_samples
    */
    ClosestCentroids& getDistance(const Matrix<T>& data, const Matrix<T>& cluster){
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
                    this->_matrix[i] = c;
                }
            }
        }
        return *this;
    }
private:
    Matrix<T> *_distBuffer;

    void initDistBuffer(){
        _distBuffer = new Matrix<T>(1, this->_cols, std::numeric_limits<float>::max(), this->_n_threads);
    }
};

template<typename T>
class KMeans{
public:
    KMeans(const Matrix<T>& dataset, int n_clusters, int n_threads=1);
    ~KMeans(){ free(_dataset_to_centroids); }

    Matrix<T> getCentroid();
    Matrix<int> getDataToCentroidMap();

    T computeDist(const Matrix<T>& first, const Matrix<T>& second);
    void mapSampleToCentroid();
    void updateCentroids();
    void run(int max_iter);

    void print();


private:
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
    //Matrix<int> _dataset_to_centroids;
    ClosestCentroids<T> *_dataset_to_centroids;
};

template<typename T>
KMeans<T>::KMeans(const Matrix<T>& dataset, int n_clusters, int n_threads) : 
        _training_set{ dataset },
        _n_clusters{ n_clusters },
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

    _dataset_to_centroids = new ClosestCentroids<T>(1, _samples, -1, _n_threads);
}

template<typename T>
Matrix<T> KMeans<T>::getCentroid(){
    return _centroids;
}

template<typename T>
Matrix<int> KMeans<T>::getDataToCentroidMap(){
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
   /*
    for(int c = 0; c < _n_clusters; ++c){
        Matrix<T> curr_cluster = _centroids.getSlice(0, _dims, c, c+1);
        Matrix<T> training_set_cpy = _training_set;
        training_set_cpy.hBroadcast(curr_cluster, SUB);
        training_set_cpy.applyFunc(norm_1);
        _distBuff.insert(training_set_cpy.vSum(), c, c+1, 0, _samples);
    }
    _dataset_to_centroids = _distBuff.hMinIndex();
    */
   _dataset_to_centroids->getDistance(_training_set, _centroids);
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
        int k_index = (*_dataset_to_centroids)(0, i);
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
void KMeans<T>::run(int max_iter){
    for(int epoch = 0; epoch < max_iter; ++epoch){
        this->mapSampleToCentroid();
        this->updateCentroids();
        //std::printf("%4d\r", epoch);
    }
}

template<typename T>
void KMeans<T>::print() {
    for(int d = 0; d < _dims; ++d){
        std::cout << "centroids_x = [";
        std::cout << _centroids.row(d) << "]" << std::endl;
    }
}
