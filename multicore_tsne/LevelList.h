#pragma once
#include <ctime>
#include "Level.h"
#include "VpTree.h"
#include<algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>

class LevelList {
	Level *head, *tail;
	int perplexity, walksNum, walksLength, threhold, endLevelSize, length;
	
public:
	LevelList(int perp, int randWalksNum, int randWalksLength, int randWalksThrehold, int endSize) {
		perplexity = perp;
		walksNum = randWalksNum;
		walksLength = randWalksLength;
		threhold = randWalksThrehold;
		endLevelSize = endSize;
		head = NULL;
		tail = NULL;
		length = 0;
		srand(unsigned(time(0)));
	};
	void initData(double* data, int rowNum, int dim);
	void getTopLevelIndexes(int **_indexes);
	void computeLevelList(int** _levelSizes, int** _indexes, int** _levelInfluenceSizes, 
		int** _pointInfluenceSizes, int** _influenceIndexes, double** _influenceValues);
private:
	vector<int> computeNextLevelIndexes();
	SparseMatrix<double> computeNextLevelInfluences(vector<int> indexes);
	int computeNextLevel();
	void standardizeData(double* data, int rowNum, int dim);
};

class DataPoint
{
    int _D;
    int _ind;
    double* _x;

public:
    DataPoint() {
        _D = 1;
        _ind = -1;
        _x = NULL;
    }
    DataPoint(int D, int ind, double* x) {
        _D = D;
        _ind = ind;
        _x = (double*) malloc(_D * sizeof(double));
        for (int d = 0; d < _D; d++) _x[d] = x[d];
    }
    DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
        if (this != &other) {
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));
            for (int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
    }
    ~DataPoint() { if (_x != NULL) free(_x); }
    DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
        if (this != &other) {
            if (_x != NULL) free(_x);
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));
            for (int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
        return *this;
    }
    int index() const { return _ind; }
    int dimensionality() const { return _D; }
    double x(int d) const { return _x[d]; }
};


double euclidean_distance(const DataPoint &t1, const DataPoint &t2);

void getInfluenceIndexes(int* levelSizes, int* indexes, int* levelInfluenceSizes,
    int* pointInfluenceSizes, int* influenceIndexes, double* influenceValues,
    int *indexSet, int size, int** _resultSet);