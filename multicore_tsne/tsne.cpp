/*/*
*  tsne.cpp
*  Implementation of both standard and Barnes-Hut-SNE.
*
*  Created by Laurens van der Maaten.
*  Copyright 2012, Delft University of Technology. All rights reserved.
*
*  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
*/

#include "tsne.h"

using namespace std;

static const int QT_NO_DIMS = 2;

// Perform t-SNE
// X -- double matrix of size [N, D]
// D -- input dimentionality
// Y -- array to fill with the result of size [N, noDims]
// noDims -- target dimentionality
void TSNE::run(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int oldNum) {

	if (N - 1 < 3 * perplexity) {
		printf("Perplexity too large for the number of data points!\n");
		exit(1);
	}
	edge = 0;
	nonedge = 0;
	total = 0;
	initTree = 0;
	numThreads = _numThreads;
	omp_set_num_threads(numThreads);

	printf("Using noDims = %d, perplexity = %f, and theta = %f\n", noDims, perplexity, theta);

	// Set learning parameters
	float totalTime = .0;
	time_t start, end;
	clock_t xsxStart, xsxEnd, xsx1, xsx2, xsxTotal;
	int stopLyingIter = 250, momSwitchIter = 250;
	double momentum = .5, finalMomentum = .8;
	double eta = 200.0;

	// Allocate some memory
	double* dY = (double*)calloc(N * noDims, sizeof(double));
	double* uY = (double*)calloc(N * noDims, sizeof(double));
	double* gains = (double*)malloc(N * noDims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * noDims; i++) {
		gains[i] = 1.0;
	}

	// Normalize input data (to prevent numerical problems)
	printf("Computing input similarities...\n");
	start = time(0);
	zeroMean(X, N, D);
	double maxX = .0;
	for (int i = 0; i < N * D; i++) {
		if (X[i] > maxX) maxX = X[i];
	}
	for (int i = 0; i < N * D; i++) {
		X[i] /= maxX;
	}

	// Compute input similarities
	int* rowP; int* colP; double* valP;

	// Compute asymmetric pairwise input similarities
	xsxStart = clock();
	computeGaussianPerplexity(X, N, D, &rowP, &colP, &valP, perplexity, (int)(3 * perplexity));
	xsxEnd = clock();
	cout << "xsx computing input similarities: " << (xsxEnd - xsxStart) << endl;

	// Symmetrize input similarities
	symmetrizeMatrix(&rowP, &colP, &valP, N);
	double sumP = .0;
	for (int i = 0; i < rowP[N]; i++) {
		sumP += valP[i];
	}
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] /= sumP;
	}

	end = time(0);
	printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start), (double)rowP[N] / ((double)N * (double)N));

	// Step 2
	// Lie about the P-values
	for (int i = 0; i < rowP[N]; i++) {
		valP[i] *= 12.0;
	}

	// Initialize solution
	// Build ball tree on old data set and update new points position
	if (oldNum > 0) {
		xsxStart = clock();
		VpTree<DataPoint, euclideanDistance>* oldTree = new VpTree<DataPoint, euclideanDistance>();
		std::vector<DataPoint> oldObjX(oldNum, DataPoint(D, -1, X));
#pragma omp parallel for
		for (int n = 0; n < oldNum; n++) {
			oldObjX[n] = DataPoint(D, n, X + n * D);
		}
		oldTree->create(oldObjX);
		xsxEnd = clock();
		cout << "xsx build ball tree on old data set: " << (xsxEnd - xsxStart) << endl;

		xsxStart = clock();
#pragma omp parallel for
		for (int i = oldNum * noDims; i < N * noDims; i += noDims) {
			// Find nearest neighbors
			std::vector<DataPoint> indices;
			std::vector<double> distances;
			int n = i / noDims, K = oldNum < 300? 1: min(5, (oldNum / 300));
			oldTree->search(DataPoint(D, n, X + n * D), K, &indices, &distances);
			for(int j = 0;j < noDims; j++) {
				Y[i + j] = .0;
			}
			for (int k = 0; k < K; k++) {
				for(int j = 0;j < noDims; j++) {
					Y[i + j] += Y[indices[k].index() * noDims + j];
				}
			}
			for(int j = 0;j < noDims; j++) {
				Y[i + j] /= double(K);
			}
		}
		delete oldTree;
		oldObjX.clear();
		xsxEnd = clock();
		cout << "xsx init new data points' position: " << (xsxEnd - xsxStart) << endl;
	}
	else {
		xsxStart = clock();
		if (randomState != -1) {
			srand(randomState);
		}
		for (int i = oldNum * noDims; i < N * noDims; i++) {
			Y[i] = randn() * .0001;
		}
		xsxEnd = clock();
		cout << "xsx init all data points' position: " << (xsxEnd - xsxStart) << endl;
	}

	// Perform main training loop
	start = time(0);
	xsxStart = clock();
	xsx1 = 0;
	xsx2 = 0;
	xsxTotal = 0;
	for (int iter = 0; iter < maxIter; iter++) {

		// Compute approximate gradient
		xsx1 = xsx1 - clock();
		computeGradient(rowP, colP, valP, Y, N, noDims, dY, theta, oldNum);
		xsx1 = xsx1 + clock();
		xsx2 = xsx2 - clock();
#pragma omp parallel for
		for (int i = oldNum * noDims; i < N * noDims; i++) {

			// Update gains
			gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
			if (gains[i] < .01) {
				gains[i] = .01;
			}

			// Perform gradient update (with momentum and gains)
			uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
			Y[i] = Y[i] + uY[i];
		}

		// Make solution zero-mean
		zeroMean(Y, N, noDims);
		xsx2 = xsx2 + clock();

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stopLyingIter) {
			for (int i = 0; i < rowP[N]; i++) {
				valP[i] /= 12.0;
			}
		}
		if (iter == momSwitchIter) {
			momentum = finalMomentum;
		}
		xsxTotal = xsxTotal - clock();

		// Print out progress
		if ((iter > 0 && iter % 50 == 0) || (iter == maxIter - 1)) {
			end = time(0);
			double C = .0;

			C = evaluateError(rowP, colP, valP, Y, N, theta);  // doing approximate computation here!

			if (iter == 0)
				printf("Iteration %d: error is %f\n", iter + 1, C);
			else {
				totalTime += (float)(end - start);
				printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start));
			}
			start = time(0);
		}
		xsxTotal = xsxTotal + clock();
	}

	xsxEnd = clock();
	// Print out time cost in process
	cout << "xsx main train loop: " << (xsxEnd - xsxStart) << endl;
	cout << "xsx main train loop, compute gradient: " << xsx1 << endl;
	cout << "xsx main train loop, update Y: " << xsx2 << endl;
	cout << "xsx main train loop, print process: " << xsxTotal << endl;
	cout << "xsx compute gradient, init tree: " << initTree << endl;
	cout << "xsx compute gradient, edge: " << edge << endl;
	cout << "xsx compute gradient, nonedge: " << nonedge << endl;
	cout << "xsx compute gradient, total: " << total << endl;
	end = time(0); totalTime += (float)(end - start);

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);

	free(rowP); rowP = NULL;
	free(colP); colP = NULL;
	free(valP); valP = NULL;

	printf("Fitting performed in %4.2f seconds.\n", totalTime);
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(int* inpRowP, int* inpColP, double* inpValP, double* Y, int N, int D, double* dC, double theta, int oldNum)
{
	initTree -= clock();

	// Construct quadtree on current map
	QuadTree* tree = new QuadTree(Y, N);
	initTree += clock();

	// Compute all terms required for t-SNE gradient
	double sumQ = .0;
	double* posF = (double*)calloc(N * D, sizeof(double));
	double* negF = (double*)calloc(N * D, sizeof(double));
	if (posF == NULL || negF == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	edge -= clock();
	tree->computeEdgeForces(inpRowP, inpColP, inpValP, N, posF, oldNum);
	edge += clock();

	nonedge -= clock();
#pragma omp parallel for reduction(+:sumQ)
	for (int n = 0; n < N; n++) {
		double buff[QT_NO_DIMS];
		double thisQ = .0;
		tree->computeNonEdgeForces(n, theta, negF + n * D, &thisQ, &buff[0]);
		sumQ += thisQ;
	}
	nonedge += clock();

	total -= clock();

	// Compute final t-SNE gradient
	for (int i = oldNum * D; i < N * D; i++) {
		dC[i] = posF[i] - (negF[i] / sumQ);
	}
	total += clock();
	free(posF);
	free(negF);
	delete tree;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(int* rowP, int* colP, double* valP, double* Y, int N, double theta)
{

	// Get estimate of normalization term
	//const int QT_NO_DIMS = 2;
	QuadTree* tree = new QuadTree(Y, N);
	double buff[QT_NO_DIMS] = { .0, .0 };
	double sumQ = .0;
	for (int n = 0; n < N; n++) {
		double buff1[QT_NO_DIMS];
		tree->computeNonEdgeForces(n, theta, buff, &sumQ, &buff1[0]);
	}

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * QT_NO_DIMS;
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {
			Q = .0;
			ind2 = colP[i] * QT_NO_DIMS;
			for (int d = 0; d < QT_NO_DIMS; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < QT_NO_DIMS; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < QT_NO_DIMS; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sumQ;
			C += valP[i] * log((valP[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}
	return C;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, int** _rowP, int** _colP, double** _valP, double perplexity, int K) {

	if (perplexity > K) printf("Perplexity should be lower than K!\n");

	// Allocate the memory we need
	*_rowP = (int*)malloc((N + 1) * sizeof(int));
	*_colP = (int*)calloc(N * K, sizeof(int));
	*_valP = (double*)calloc(N * K, sizeof(double));
	if (*_rowP == NULL || *_colP == NULL || *_valP == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	int* rowP = *_rowP;
	int* colP = *_colP;
	double* valP = *_valP;
	clock_t xsxStart, xsxEnd;

	rowP[0] = 0;
	for (int n = 0; n < N; n++) {
		rowP[n + 1] = rowP[n] + K;
	}

	// Build ball tree on data set
	VpTree<DataPoint, euclideanDistance>* tree = new VpTree<DataPoint, euclideanDistance>();
	std::vector<DataPoint> objX(N, DataPoint(D, -1, X));
#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		objX[n] = DataPoint(D, n, X + n * D);
	}
	tree->create(objX);

	// Loop over all points to find nearest neighbors
	printf("Building tree...\n");

	int stepsCompleted = 0;
#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		std::vector<double> curP(K);
		std::vector<DataPoint> indices;
		std::vector<double> distances;

		// Find nearest neighbors
		tree->search(objX[n], K + 1, &indices, &distances);

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double minBeta = -DBL_MAX;
		double maxBeta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sumP;
		while (!found && iter < 200) {

			// Compute Gaussian kernel row
			for (int m = 0; m < K; m++) {
				curP[m] = exp(-beta * distances[m + 1]);
			}

			// Compute entropy of current row
			sumP = DBL_MIN;
			for (int m = 0; m < K; m++) {
				sumP += curP[m];
			}
			double H = .0;
			for (int m = 0; m < K; m++) {
				H += beta * (distances[m + 1] * curP[m]);
			}
			H = (H / sumP) + log(sumP);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					minBeta = beta;
					if (maxBeta == DBL_MAX || maxBeta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + maxBeta) / 2.0;
				}
				else {
					maxBeta = beta;
					if (minBeta == -DBL_MAX || minBeta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + minBeta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		// Row-normalize current row of P and store in matrix
		for (int m = 0; m < K; m++) {
			curP[m] /= sumP;
		}
		for (int m = 0; m < K; m++) {
			colP[rowP[n] + m] = indices[m + 1].index();
			valP[rowP[n] + m] = curP[m];
		}

		// Print progress
#pragma omp atomic
		++stepsCompleted;

		if (stepsCompleted % 10000 == 0)
		{
#pragma omp critical
			printf(" - point %d of %d\n", stepsCompleted, N);
		}
	}

	// Clean up memory
	objX.clear();
	delete tree;
}


void TSNE::symmetrizeMatrix(int** _rowP, int** _colP, double** _valP, int N) {

	// Get sparse matrix
	int* rowP = *_rowP;
	int* colP = *_colP;
	double* valP = *_valP;

	// Count number of elements and row counts of symmetric matrix
	int* rowCounts = (int*)calloc(N, sizeof(int));
	if (rowCounts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {

			// Check whether element (colP[i], n) is present
			bool present = false;
			for (int m = rowP[colP[i]]; m < rowP[colP[i] + 1]; m++) {
				if (colP[m] == n) present = true;
			}
			if (present) rowCounts[n]++;
			else {
				rowCounts[n]++;
				rowCounts[colP[i]]++;
			}
		}
	}
	int noElem = 0;
	for (int n = 0; n < N; n++) noElem += rowCounts[n];

	// Allocate memory for symmetrized matrix
	int*    symRowP = (int*)malloc((N + 1) * sizeof(int));
	int*    symColP = (int*)malloc(noElem * sizeof(int));
	double* symValP = (double*)malloc(noElem * sizeof(double));
	if (symRowP == NULL || symColP == NULL || symValP == NULL) { printf("Memory allocation failed!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	symRowP[0] = 0;
	for (int n = 0; n < N; n++) symRowP[n + 1] = symRowP[n] + rowCounts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {                                 // considering element(n, colP[i])

																					  // Check whether element (col_P[i], n) is present
			bool present = false;
			for (int m = rowP[colP[i]]; m < rowP[colP[i] + 1]; m++) {
				if (colP[m] == n) {
					present = true;
					if (n <= colP[i]) {                                                // make sure we do not add elements twice
						symColP[symRowP[n] + offset[n]] = colP[i];
						symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
						symValP[symRowP[n] + offset[n]] = valP[i] + valP[m];
						symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i] + valP[m];
					}
				}
			}

			// If (colP[i], n) is not present, there is no addition involved
			if (!present) {
				symColP[symRowP[n] + offset[n]] = colP[i];
				symColP[symRowP[colP[i]] + offset[colP[i]]] = n;
				symValP[symRowP[n] + offset[n]] = valP[i];
				symValP[symRowP[colP[i]] + offset[colP[i]]] = valP[i];
			}

			// Update offsets
			if (!present || (present && n <= colP[i])) {
				offset[n]++;
				if (colP[i] != n) offset[colP[i]]++;
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < noElem; i++) symValP[i] /= 2.0;

	// Return symmetrized matrices
	free(*_rowP); *_rowP = symRowP;
	free(*_colP); *_colP = symColP;
	free(*_valP); *_valP = symValP;

	// Free up some memery
	free(offset); offset = NULL;
	free(rowCounts); rowCounts = NULL;
}

// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[n * D + d];
		}
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[n * D + d] -= mean[d];
		}
	}
	free(mean); mean = NULL;
}

// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

extern "C"
{
	extern void tsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState)
	{
		printf("Performing t-SNE using %d cores.\n", _numThreads);
		TSNE tsne;
		tsne.run(X, N, D, Y, noDims, perplexity, theta, _numThreads, maxIter, randomState, 0);
	}
	extern void incrementalTsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int oldNum)
	{
		printf("Performing incremental t-SNE using %d cores.\n", _numThreads);
		TSNE tsne;
		tsne.run(X, N, D, Y, noDims, perplexity, theta, _numThreads, maxIter, randomState, oldNum);
	}
	extern void landMarkSampling(int threads, int perp, int randWalksNum, int randWalksLength, int randWalksThrehold,
		int endSize, double* data, int rowNum, int dim, int** _levelSizes, int** _indexes, int** _levelInfluenceSizes,
		int** _pointInfluenceSizes, int** _influenceIndexes, int** _topLevelIndexes, double** _influenceValues)
	{
		printf("Performing landmark sampling using %d cores.\n", threads);
		omp_set_num_threads(threads);
		LevelList levelList = LevelList(perp, randWalksNum, randWalksLength, randWalksThrehold, endSize);
		printf("Initing data...\n");
		clock_t t = clock();
		levelList.initData(data, rowNum, dim);
		printf("Init data cost %f\n", float(clock() - t));
		printf("Computing level list...\n");
		t = clock();
		levelList.computeLevelList(_levelSizes, _indexes, _levelInfluenceSizes, _pointInfluenceSizes, _influenceIndexes, _influenceValues);
		printf("Compute level list cost %f\n", float(clock() - t));
		levelList.getTopLevelIndexes(_topLevelIndexes);
	}
	extern void getInfluenceIndexes(int threads, int* levelSizes, int* indexes, int* levelInfluenceSizes, int* pointInfluenceSizes,
		int* influenceIndexes, double* influenceValues, int *indexSet, int size, double minInfluenceValue, int** _resultSet)
	{
		printf("Get influenced indexes.\n");
		omp_set_num_threads(threads);
		getInfluenceIndexes(levelSizes, indexes, levelInfluenceSizes, pointInfluenceSizes, influenceIndexes, influenceValues, indexSet, size, minInfluenceValue, _resultSet);
	}
}
