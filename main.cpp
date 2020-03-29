#include <chrono>
#include <cmath>

#include "KMeans.h"

#include "headers/CSVMatrix.h"
#include "headers/Matrix.h"
#include "headers/Utils.h"
#include "headers/DataParser.h"
#include <string>

Matrix<float> dataGenerator(int n){
    // 4 classes and 2 dimensions (x, y) -> 8 values
    // (µ_x_0, µ_y_0), (..., ...), (µ_x_c, µ_y_c)
    float mu_vect[] = {-1, 2, \
                        2, 1, \
                        4, -2, \
                        3, -1};
    float sd_vect[] = {0.5, 0.6, \
                       0.8, 0.7, \
                       1.2, 0.3, \
                       0.3, 0.5};
    Matrix<float> database_generator(8, n, GAUSS, mu_vect, sd_vect, 1);
    return database_generator;
}

template<typename T>
Matrix<T> dataGenerator3D(int distr_samples){
    // #data_points = #distr * distr_samples
    // 4 classes and 3 dimensions (x, y, z) -> 12 values
    // (µ_x_0, µ_y_0, µ_z_0), (..., ..., ...), (µ_x_c, µ_y_c,µ_z_c)
    T mu_vect[] = {1, 2, 3, \
                        2, 1, 1, \
                        4, 2, 2, \
                        3, 1, 4};
    T sd_vect[] = {0.5, 0.6, 0.4, \
                       0.8, 0.7, 0.5, \
                       1.2, 0.3, 0.8, \
                       0.3, 0.5, 0.4};
    Matrix<T> database_generator(12, distr_samples, GAUSS, mu_vect, sd_vect, 1);

    Matrix<T> dataX = database_generator.row(0); 
    dataX.hStack(database_generator.row(3));
    dataX.hStack(database_generator.row(6));
    dataX.hStack(database_generator.row(9));

    Matrix<T> dataY = database_generator.row(1);
    dataY.hStack(database_generator.row(4));
    dataY.hStack(database_generator.row(7));
    dataY.hStack(database_generator.row(10));

    Matrix<T> dataZ = database_generator.row(2);
    dataZ.hStack(database_generator.row(5));
    dataZ.hStack(database_generator.row(8));
    dataZ.hStack(database_generator.row(11));

    Matrix<T> DATABASE(dataX);
    DATABASE.vStack(dataY);
    DATABASE.vStack(dataZ);
    return DATABASE;
}

void dataParsing(){
    std::string from_path = "containers/OpenMP/matrix/headers/files/input/mnist_train.csv";
    CSVMatrix<int> CSV(from_path);
    CSV.fillData(4);
    std::cout << "Numb. of digits loaded: " << CSV.getRows() << std::endl;
    
    for(int s = 0; s < 5; ++s){
        Matrix<int> digit = CSV.row(s);
        for(int i = 0; i < 28; ++i){
            for(int j = 0; j < 28; ++j){
                int val = digit(0, j+i*28);
                if(val > 0) std::cout << "o";
                else std::cout << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
   std::cout << CSV << std::endl;
   CSV.putFile("containers/OpenMP/matrix/headers/files/output/example.csv", 0, CSV.getRows());
}

template<typename T>
void sig(T& val){
    val = 1 / (1 + exp(-val));
}

double mean(const std::vector<double>& vect){
	double mean = 0;
	for(const auto& elem : vect) mean += elem;
	return mean /= vect.size();
}

double stand_dev(const std::vector<double>& vect, const double& mu){
	double res = 0;
	for(const auto& elem : vect) res += std::pow(elem-mu, 2);
	return std::sqrt(1.0/(vect.size()-1)*res);
}

double mean_mat(const Matrix<int>& vect){
	double mean = 0;
	for(int i = 0; i < vect.getCols(); ++i) mean += static_cast<double>(vect(0, i));
	return mean /= vect.getCols();
}

double stand_dev_mat(const Matrix<int>& vect, const double& mu){
	double res = 0;
	for(int i = 0; i < vect.getCols(); ++i) res += std::pow(static_cast<double>(vect(0, i))-mu, 2);
	return std::sqrt(1.0/(vect.getCols()-1)*res);
}

void KMeansBenckmark(const Matrix<float>& DATABASE, int n_centroids, 
		double *time_arr, double *sd_time_arr, 
		double *iters_arr, double *sd_iters_arr,
		double *errors_arr, double *sd_errors_arr,
		int iter){
    std::vector<double> time_buff;
    std::vector<double> iters_buff; // iters will be averaged -> double 
    std::vector<double> errors_buff; 
    
    static Timer<nano_t> timer_inner;

    for(int i = 0; i < 7; ++i){
        KMeans<float> KM(DATABASE, n_centroids, true, 1);

        timer_inner.reset();    
        KM.run(50, 0.02);
        time_buff.push_back(timer_inner.elapsed() * 1e-9);
	iters_buff.push_back(KM.getNIters());
	
	Matrix<int> predictions = KM.getDataToCentroid();
	Matrix<int> slice0 = predictions.getSlice(0, 1, 0,     25000);
	Matrix<int> slice1 = predictions.getSlice(0, 1, 25000, 50000);
	Matrix<int> slice2 = predictions.getSlice(0, 1, 50000, 75000);
	Matrix<int> slice3 = predictions.getSlice(0, 1, 75000, 100000);
	
	double sd_slice0 = stand_dev_mat(slice0, mean_mat(slice0));
	double sd_slice1 = stand_dev_mat(slice1, mean_mat(slice1));
	double sd_slice2 = stand_dev_mat(slice2, mean_mat(slice2));
	double sd_slice3 = stand_dev_mat(slice3, mean_mat(slice3));

	errors_buff.push_back((sd_slice0+sd_slice1+sd_slice2+sd_slice3)/4.0);
    }
    
    // sd = sqrt(1/(m-1)sum((x_i - x_bar)**2))
	double mu_time = mean(time_buff);
	double sd_time = stand_dev(time_buff, mu_time);

	double mu_iters = mean(iters_buff);
	double sd_iters = stand_dev(iters_buff, mu_iters);

	double mu_errors = mean(errors_buff);
	double sd_errors = stand_dev(errors_buff, mu_errors);
    printf("%d\r", iter);

    time_arr[iter] = mu_time;
    sd_time_arr[iter] = sd_time;

    iters_arr[iter] = mu_iters;
    sd_iters_arr[iter] = sd_iters;

    errors_arr[iter] = mu_errors;
    sd_errors_arr[iter] = sd_errors;
}

int main(int argc, char** argv){
	Timer<nano_t> timer;
	//uint64_t time_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    /////////////////////////////
    
   	//Matrix<float> DATABASE(dataX, 1, 400, 4);
    	//DATABASE.vStack(dataY, 1, 400);
    	//DATABASE.vStack(dataZ, 1, 400);
    Matrix<double> DATABASE = dataGenerator3D<double>(25000);
	// save database that has been used
	std::string path = "data/output/k-means-database.txt";
	TXTParser<double> txt1(path);
	txt1.putData(DATABASE.begin(), DATABASE.end(), DATABASE.getRows(), DATABASE.getCols(), path, false, ',');
    	
	timer.reset();
    KMeans<double> KM(DATABASE, 4, true, 4);
	KM.run(50, 0.00001);
	Matrix<double> centroid = KM.getCentroid();
	Matrix<int> dataToCentroid = KM.getDataToCentroid();
	std::cout << "KMeans time: " << (timer.elapsed()*1e-9) << std::endl;
	std::cout << "KMeans iters: " << KM.getNIters() << std::endl;
	//std::cout << "Centroids: \n" << centroid << std::endl;
	KM.print();
	//std::cout << "data to centroids: \n" << dataToCentroid << std::endl;
	std::string to_path = "data/output/k-means-dataToCentroid.txt";
	TXTParser<int> txt(to_path);
	txt.putData(dataToCentroid.begin(), dataToCentroid.end(), dataToCentroid.getRows(), dataToCentroid.getCols(), to_path, false, ','); 
	

	// Benchmarking Convergence
	/*
    	// [2, 100]  		
    	Matrix<float> DATABASE = dataGenerator3D(25000);
	// save database that has been used
	std::string path = "data/output/k-means-database.txt";
	TXTParser<float> txt(path);
	Matrix<float> DATABASE_T = DATABASE.transpose();
	txt.putData(DATABASE_T.begin(), DATABASE_T.end(), 
			DATABASE_T.getRows(), DATABASE_T.getCols(), 
			path, false, ',');
	

    	const int size = 35;
    	double time_arr[size];
    	double sd_time_arr[size];
	
	double iters_arr[size];
	double sd_iters_arr[size];

	double errors_arr[size]; 
	double sd_errors_arr[size];

    	for(int n = 0; n < size; ++n){
    	    KMeansBenckmark(DATABASE, 2+2, time_arr, sd_time_arr, iters_arr, sd_iters_arr, errors_arr, sd_errors_arr, n);
    	}
    	std::cout << "\ntime_arr = [";
    	for(int n = 0; n < size; ++n){
    	    std::cout << time_arr[n] << ", ";
    	    //sd_time_arr[n] = 0;
    	}
    	std::cout << "]";
    	std::cout << "\ntime_SD = [";
    	for(int n = 0; n < size; ++n){
       	 	std::cout << sd_time_arr[n] << ", ";
    	}
   	std::cout << "]" << std::endl;
    	std::cout << "\niters_arr = [";
    	for(int n = 0; n < size; ++n){
       	 	std::cout << iters_arr[n] << ", ";
    	}
   	std::cout << "]";
	std::cout << "\niters_SD = [";
    	for(int n = 0; n < size; ++n){
       	 	std::cout << sd_iters_arr[n] << ", ";
    	}
   	std::cout << "]" << std::endl;
	std::cout << "\nerrors_arr = [";
    	for(int n = 0; n < size; ++n){
       	 	std::cout << errors_arr[n] << ", ";
    	}
   	std::cout << "]" << std::endl;
	std::cout << "\nerrors_SD = [";
    	for(int n = 0; n < size; ++n){
       	 	std::cout << sd_errors_arr[n] << ", ";
    	}
   	std::cout << "]" << std::endl;
	*/
    	/////////////////////////////
    	uint64_t total_time = timer.elapsed();
    	printf("total time: %.5f\n", (total_time*1e-9));
    	return 0;
}
