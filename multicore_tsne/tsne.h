/*
*  tsne.h
*  Header file for t-SNE.
*
*  Created by Laurens van der Maaten.
*  Copyright 2012, Delft University of Technology. All rights reserved.
*
*  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
*/


#pragma once
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <time.h>
#include <omp.h>
#include <ctime>
#include <iostream>
#include <fstream>

#include "quadtree.h"
#include "vptree.h"
#include "LevelList.h"

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class TSNE
{
public:
	void run(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int numThreads, int maxIter, int randomState, int oldNum);
	void symmetrizeMatrix(int** rowP, int** colP, double** valP, int N);
private:
	int numThreads;
	clock_t initTree, edge, nonedge, total;
	void computeGradient(int* inpRowP, int* inpColP, double* inpValP, double* Y, int N, int D, double* dC, double theta, int oldNum);
	double evaluateError(int* rowP, int* colP, double* valP, double* Y, int N, double theta);
	void zeroMean(double* X, int N, int D);
	void computeGaussianPerplexity(double* X, int N, int D, int** _rowP, int** _colP, double** _valP, double perplexity, int K);
	double randn();
};