#include<iostream>
#include<fstream>
#include<string>
#include<algorithm>
#include<sstream>
#include<cctype> // std::ispunct
#include<vector>
#include<limits> // std::numeric_limits<std::streamsize>::max()
#include<iterator>

template<typename T>
class TXTParser{
public:
	TXTParser(std::string path) : 
		_path{ path }{

		_rows = getRows();
		_cols = getCols();
	}

	size_t getRows(){
		size_t rows = 0;
		std::ifstream read(_path);
		for(std::string line; std::getline(read, line); ++rows){ }
		read.close();
		return rows;
	}
	
	size_t getCols(){
		size_t cols = 0;
		std::ifstream read(_path);
		std::string line;
		std::getline(read, line);
		std::replace_if(std::begin(line), std::end(line), 
			[](unsigned char c){ return std::ispunct(c); },
			' ');
		// inserting line in stream to parse content
		std::stringstream ss(line);	
		T val;
		while(ss){ ss >> val; ++cols; }	
		read.close();
		return cols-1;
	}

	void getData(T *container){
		std::ifstream read(_path);

		for(std::string line; std::getline(read, line);){
			//removing punctuation
			std::replace_if(std::begin(line), std::end(line), 
				[](unsigned char c){ return std::ispunct(c); },
				' ');
			// parse content of line
			std::stringstream ss(line);
			for(size_t j = 0; j < _cols; ++j){ ss >> container[j]; }
		}
		read.close();
	}
	
	std::fstream& gotoRow(std::fstream& file, size_t row){
		file.seekg(std::ios::beg);
		for(size_t i = 0; i < row; ++i){
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		return file;
	}

	/**
	 * Place the row cursur in the file
	 * */
	template<typename Iterator>
	void getDataIt(Iterator Begin, Iterator End, size_t row = 0){
		std::fstream file(_path);
		gotoRow(file, row);
		for(std::string line; std::getline(file, line), Begin != End; ){
			//removing punctuation
			std::replace_if(std::begin(line), std::end(line), 
				[](unsigned char c){ return std::ispunct(c); },
				' ');
			// parse content of line
			std::stringstream ss(line);
			for(size_t j{ 0 }; j < _cols; ++j, ++Begin){ ss >> *Begin; }
		}
		file.close();
	}
	
	template<typename Iterator>
	void putData(Iterator Begin, Iterator End, int rows, int cols, const std::string& to_path, bool append=true, const char separator=','){
		std::ofstream file;
		if(append) file.open(to_path, std::ios::app);
		else file.open(to_path);
		
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols-1; ++j, ++Begin){
				file << *Begin << separator;
			}
			// add last element + newline carriage
			file << *Begin << '\n';
			++Begin;
		}
		file.close();
	}

private:
	// absolute path
	std::string _path; 
	size_t _rows = 0;
	size_t _cols = 0;
};


