#include "LevelList.h"
using namespace std;

void LevelList::standardizeData(double* data, int rowNum, int dim) {
    //compute mean and maximum, and set them to zero and one.
	int i, j, k = 0;
	double maxVal = 0, temp;
	double* mean = (double*) calloc(dim, sizeof(double));
	for (i = 0; i < rowNum; i++) {
		for (j = 0; j < dim; j++) {
			temp = data[i * dim + j];
			if (temp > maxVal) {
				maxVal = temp;
				k = j;
			}
			mean[j] += temp;
		}
	}
	for (j = 0; j < dim; j++) {
		mean[j] /= double(rowNum);
	}
	maxVal -= mean[k];
	for (i = 0; i < rowNum; i++) {
		for (j = 0; j < dim; j++) {
			data[i * dim + j] = (data[i * dim + j]  - mean[j]) / maxVal;
		}
	}
	free(mean); mean  = NULL;
}

void LevelList::initData(double* data, int rowNum, int dim) {
	//make value between -1 to 1, and mean zero.
	standardizeData(data, rowNum, dim);

	//define variables
	head = new Level(0, rowNum, 0);
	tail = head;
	SparseMatrix<double> transition(rowNum, rowNum), weight(1,rowNum), influence(0, 0);
	vector<Eigen::Triplet<double> > tTransition, tWeight;
	vector<int> indexes;

	//compute indexes, weight and transition matrix of first level
	// Build ball tree on data set
	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
	vector<DataPoint> obj_X(rowNum, DataPoint(dim, -1, data));
	#pragma omp parallel for
	for (int n = 0; n < rowNum; n++) {
		obj_X[n] = DataPoint(dim, n, data + n * dim);
	}
	tree->create(obj_X);
	int i;
	for (i = 0; i < rowNum; i++) {
		indexes.push_back(i);
		tWeight.push_back(Eigen::Triplet<double>(0, i, 1.0));
		std::vector<double> cur_P(perplexity * 3);
		std::vector<DataPoint> indices;
		std::vector<double> distances;
		// Find nearest neighbors
		tree->search(obj_X[i], 3 * perplexity + 1, &indices, &distances);

		// Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while (!found && iter < 200) {

			// Compute Gaussian kernel row
			#pragma omp parallel for
			for (int m = 0; m < 3 * perplexity; m++) {
				cur_P[m] = exp(-beta * distances[m + 1]);
			}

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < 3 * perplexity; m++) {
				sum_P += cur_P[m];
			}
			double H = .0;
			for (int m = 0; m < 3 * perplexity; m++) {
				H += beta * (distances[m + 1] * cur_P[m]);
			}
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}
		// Compute transition matrix
		beta = sqrt((1.0 / beta) / 2);
		#pragma omp parallel for
		for (int m = 0; m < 3 * perplexity; m++) {
			cur_P[m] = exp(distances[m + 1] / beta);
		}
		sum_P = DBL_MIN;
		for (int m = 0; m < 3 * perplexity; m++) {
			sum_P += cur_P[m];
		}
		for (int m = 0; m < 3 * perplexity; m++) {
			tTransition.push_back(Eigen::Triplet<double>(i, indices[m + 1].index(), cur_P[m] / sum_P));
		}
	}
	transition.setFromTriplets(tTransition.begin(), tTransition.end());
	weight.setFromTriplets(tWeight.begin(), tWeight.end());
	head->initData(NULL, transition, weight, influence, indexes);
	length++;
	delete tree;
}

void getInfluenceIndexes(int* levelSizes, int* indexes, int* levelInfluenceSizes,
	int* pointInfluenceSizes, int* influenceIndexes, double* influenceValues, int *indexSet, 
	int size, int** _resultSet) {
	if (size < 0) {
		return;
	}
	int levelID = levelSizes[0] - 2;
	int i, j, k, beginIndex = levelSizes[levelSizes[0] - 1];
	vector<int> levelIndexes;
	for (i = 0; i < size; i++) {
		j = indexes[beginIndex + indexSet[i]];
		if (j < levelID) {
			levelID = j;
			levelIndexes.clear();
		}
		levelIndexes.push_back(indexSet[i]);
	}
	if (levelID == 0 || pointInfluenceSizes[0] == 1) {
		*_resultSet = new int[size];
		int* resultSet = *_resultSet;
		#pragma omp parallel for
		for (i = 0; i < size; i++) {
			resultSet[i] = indexSet[i];
		}
		return;
	}
	sort(levelIndexes.begin(), levelIndexes.end());
	i = levelSizes[levelSizes[0] - levelID - 1];
	beginIndex = levelSizes[levelSizes[0] - levelID];
	j = 0;
	int levelIndexSize = levelIndexes.size();
	int* influenceSource = new int[levelIndexSize];
	for (k = 0; k + i < beginIndex; k++) {
		if (indexes[k + i] == levelIndexes[j]) {
			influenceSource[j] = k;
			j++;
			if (j == levelIndexSize) {
				break;
			}
		}
	}
	k = levelInfluenceSizes[0] - levelID;
	int l, r, temp;
	i = levelInfluenceSizes[k];
	l = 1;
	r = pointInfluenceSizes[0];
	while (l <= r) {
		temp = pointInfluenceSizes[(l + r) / 2];
		if (temp == i) {
			i = (l + r) / 2;
			break;
		}
		else if (temp < i) {
			l = (l + r) / 2 + 1;
		}
		else {
			r = (l + r) / 2 - 1;
		}
	}
	j = levelInfluenceSizes[k + 1];
	l = 1;
	r = pointInfluenceSizes[0];
	while (l <= r) {
		temp = pointInfluenceSizes[(l + r) / 2];
		if (temp == j) {
			j = (l + r) / 2;
			break;
		}
		else if (temp < j) {
			l = (l + r) / 2 + 1;
		}
		else {
			r = (l + r) / 2 - 1;
		}
	}
	
	l = 0;
	levelIndexes.clear();
	int startIndex = levelSizes[levelSizes[0] - levelID];
	for (k = 0; k < j - i; k++) {
		if (influenceSource[l] == k) {
			r = pointInfluenceSizes[k + i + 1];
			for (int m = pointInfluenceSizes[k + i]; m < r; m++) {
				if (levelID == 1) {
					levelIndexes.push_back(influenceIndexes[m]);
				}
				else {
					levelIndexes.push_back(indexes[startIndex + influenceIndexes[m]]);
				}
			}
			l++;
		}
	}
	j = levelIndexes.size();
	*_resultSet = new int[j + 1];
	int* resultSet = *_resultSet;
	resultSet[0] = j;
	#pragma omp parallel for
	for(i = 0;i < j;i++){
		resultSet[i + 1] = levelIndexes[i];
	}
	delete[] influenceSource;
}

void LevelList::getTopLevelIndexes(int **_indexes) {
	vector<int> indexes = tail->getIndexes();
	int levelSize = indexes.size();
	*_indexes = new int[levelSize + 1];
	int* result = *_indexes;
	int i;
	result[0] = levelSize;
	#pragma omp parallel for
	for (i = 0; i < levelSize; i++) {
		result[i + 1] = indexes[i];
	}
	return;
}

vector<int> LevelList::computeNextLevelIndexes() {
	vector<int> indexes, preIndexes = tail->getIndexes();
	SparseMatrix<double> preTransition = tail->getTransitionMatrix().adjoint();
	int i, j;
	int tailSize = tail->getSize();
	int* randomWalkResult = (int*) calloc(tailSize, sizeof(int));
	bool flag = true;
	while (flag) {
		#pragma omp for
		for (i = 0; i < tailSize; i++) {
			//#pragma omp for
			for (j = 0; j < walksNum; j++) {
				int l = i;
				for (int k = 0; k < walksLength; k++) {
					double p = 0;
					double randP = rand() / double(RAND_MAX + 1);
					for (SparseMatrix<double>::InnerIterator it(preTransition, l); it; ++it)
					{
						p += it.value();
						if (p >= randP) {
							l = it.row();
							break;
						}
					}
				}
				#pragma omp critical
				{
					randomWalkResult[l]++;
				}
			}
		}
		for (i = 0; i < tailSize; i++) {
			if (randomWalkResult[i] > walksNum) {
				flag = false;
				break;
			}
		}
		if (flag) {
			for (i = 0; i < tailSize; i++) {
				randomWalkResult[i] = 0;
			}
		}
	}
	
	j = threhold;
	while (indexes.size() == 0) {
		for (i = 0; i < tailSize; i++) {
			if (randomWalkResult[i] > j) {
				indexes.push_back(preIndexes[i]);
			}
		}
		j--;
	}
	free(randomWalkResult);
	randomWalkResult = NULL;
	return indexes;
}

SparseMatrix<double> LevelList::computeNextLevelInfluences(vector<int> indexes) {
	vector<int> preIndexes = tail->getIndexes();
	SparseMatrix<double> preTransition = tail->getTransitionMatrix().adjoint();
	int i, j, k, l;
	int tailSize = tail->getSize();
	SparseMatrix<double> influences(tailSize, indexes.size());
	vector<Eigen::Triplet<double> > tInfluences;
	int totalValue = indexes.size() * walksNum;
	double p, randP;
	int* flags = (int*) calloc(tailSize, sizeof(int));
	for (i = 0,j = 0, l = indexes.size(); i < tailSize; i++) {
		if (preIndexes[i] == indexes[j]) {
			tInfluences.push_back(Eigen::Triplet<double>(i, j, double(walksNum)));
			flags[i] = j + 1;
			j++;
			if (j == l) {
				break;
			}
		}
	}

	for (i = 0; i < tailSize; i++) {
		if (flags[i] == 0) {
			for (j = 0; j < walksNum; j++) {
				l = i;
				for (k = 0; k < walksLength * 2; k++) {
					p = 0;
					randP = rand() / double(RAND_MAX + 1);
					for (SparseMatrix<double>::InnerIterator it(preTransition, l); it; ++it)
					{
						p += it.value();
						if (p >= randP) {
							l = it.row();
							break;
						}
					}
					if (flags[l] > 0) {
						tInfluences.push_back(Eigen::Triplet<double>(i, flags[l] - 1, 1.0));
						totalValue += 1;
						break;
					}
				}
			}
		}
	}
	influences.setFromTriplets(tInfluences.begin(), tInfluences.end());
	influences = influences * tailSize / double(totalValue);
	free(flags);
	flags = NULL;
	return influences;
}

int LevelList::computeNextLevel() {
	vector<int> indexes = computeNextLevelIndexes();
	SparseMatrix<double> transition, weight, preWeight = tail->getWeight(), influence = computeNextLevelInfluences(indexes), transposeInfluence;
	transposeInfluence = influence.adjoint();
	weight = preWeight * influence;
	for (int k = 0; k < transposeInfluence.outerSize(); ++k) {
		for (SparseMatrix<double>::InnerIterator it(transposeInfluence, k); it; ++it)
		{
			it.valueRef() *= preWeight.coeff(0, k);
		}
	}
	transition = transposeInfluence * influence;
	double sum;
	for (int k = 0; k < transition.outerSize(); ++k) {
		sum = 0;
		for (SparseMatrix<double>::InnerIterator it(transition, k); it; ++it)
		{
			sum += it.value();
		}
		for (SparseMatrix<double>::InnerIterator it(transition, k); it; ++it)
		{
			it.valueRef() /= sum;
		}
	}
	transition = transition.adjoint();
	Level* newLevel = new Level(tail->getID() + 1, indexes.size(), tail->getSize());
	tail->setNext(newLevel);
	newLevel->initData(tail, transition, weight, influence, indexes);
	tail = newLevel;
	length++;
	return indexes.size();
}

double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
    double dd = .0;
    for (int d = 0; d < t1.dimensionality(); d++) {
        dd += (t1.x(d) - t2.x(d)) * (t1.x(d) - t2.x(d));
    }
    return dd;
}

void LevelList::computeLevelList(int** _levelSizes, int** _indexes, int** _levelInfluenceSizes, int** _pointInfluenceSizes, int** _influenceIndexes, double** _influenceValues) {
	int i = head->getSize();
	while (i > endLevelSize) {
		i = computeNextLevel();
	}

	cout << "end compute levels..." << time(0) << endl;
	vector<int> levelIndexes;
	SparseMatrix<double> levelInfluence;
	if (head == tail) {
		*_levelSizes = new int[3];
		int* levelSizes = *_levelSizes;
		levelSizes[0] = 2;
		levelSizes[1] = 1;
		levelSizes[2] = head->getSize() + 1;

		levelIndexes = head->getIndexes();
		*_indexes = new int[levelIndexes.size() + 1];
		int* indexes = *_indexes;
		indexes[0] = levelIndexes.size();
		for (i = 0; i < levelIndexes.size(); i++)
		{
			indexes[i + 1] = 0;
		}

		*_levelInfluenceSizes = new int[2];
		int* levelInfluenceSizes = *_levelInfluenceSizes;
		levelInfluenceSizes[0] = 1;
		levelInfluenceSizes[1] = 1;

		*_pointInfluenceSizes = new int[2];
		int* pointInfluenceSizes = *_pointInfluenceSizes;
		pointInfluenceSizes[0] = 1;
		pointInfluenceSizes[1] = 1;

		*_influenceIndexes = new int[1];
		int* influenceIndexes = *_influenceIndexes;
		influenceIndexes[0] = 0;

		*_influenceValues = new double[1];
		double* influenceValues = *_influenceValues;
		influenceValues[0] = 0;
		return;
	}
	vector<int> vLevelSizes, vLevelInfluenceSizes, vPointInfluenceSizes, vInfluenceIndexes;
	vector<double> vInfluenceValues;
	Level *p = tail;
	int levelSizeSum = 1, influnceSizesSum = 1, k;
	int headSize = head->getSize(), id;
	int* levelCount = (int*) calloc(headSize, sizeof(int));
	while (p) {
		//compute levelSizes
		levelSizeSum += p->getSize();
		vLevelSizes.push_back(levelSizeSum);

		//compute indexes
		if (p != head) {
			vector<int> tempIndexes = p->getIndexes();
			id = p->getID();
			for (vector<int>::iterator iter = tempIndexes.begin(); iter != tempIndexes.end(); ++iter) {
				if (levelCount[*iter] == 0) {
					levelCount[*iter] = id;
				}
			}
			levelIndexes.insert(levelIndexes.end(), tempIndexes.begin(), tempIndexes.end());
		}
		else {
			for (i = 0; i < headSize; i++) {
				levelIndexes.push_back(levelCount[i]);
			}
		}

		//compute levelInfluenceSizes
		levelInfluence = p->getInfluenceMatrix();
		for (k = 0; k < levelInfluence.outerSize(); ++k) {
			for (SparseMatrix<double>::InnerIterator it(levelInfluence, k); it; ++it)
			{
				vInfluenceValues.push_back(it.value());
				vInfluenceIndexes.push_back(it.row());
			}
			vPointInfluenceSizes.push_back(vInfluenceIndexes.size() + 1);
		}
		if (k > 0) {
			vLevelInfluenceSizes.push_back(vPointInfluenceSizes.back());
		}
		p = p->getPre();
	}
	*_levelSizes = new int[vLevelSizes.size() + 2];
	int* levelSizes = *_levelSizes;
	levelSizes[0] = vLevelSizes.size() + 1;
	levelSizes[1] = 1;
	#pragma omp parallel for
	for (i = 0; i < vLevelSizes.size(); i++) {
		levelSizes[i + 2] = vLevelSizes[i];
	}

	*_indexes = new int[levelIndexes.size() + 1];
	int* indexes = *_indexes;
	indexes[0] = levelIndexes.size();
	#pragma omp parallel for
	for (i = 0; i < levelIndexes.size(); i++) {
		indexes[i + 1] = levelIndexes[i];
	}

	*_levelInfluenceSizes = new int[vLevelInfluenceSizes.size() + 2];
	int* levelInfluenceSizes = *_levelInfluenceSizes;
	levelInfluenceSizes[0] = vLevelInfluenceSizes.size() + 1;
	levelInfluenceSizes[1] = 1;
	#pragma omp parallel for
	for (i = 0; i < vLevelInfluenceSizes.size(); i++) {
		levelInfluenceSizes[i + 2] = vLevelInfluenceSizes[i];
	}

	*_pointInfluenceSizes = new int[vPointInfluenceSizes.size() + 2];
	int* pointInfluenceSizes = *_pointInfluenceSizes;
	pointInfluenceSizes[0] = vPointInfluenceSizes.size() + 1;
	pointInfluenceSizes[1] = 1;
	#pragma omp parallel for
	for (i = 0; i < vPointInfluenceSizes.size(); i++) {
		pointInfluenceSizes[i + 2] = vPointInfluenceSizes[i];
	}

	*_influenceIndexes = new int[vInfluenceIndexes.size() + 1];
	int* influenceIndexes = *_influenceIndexes;
	influenceIndexes[0] = vInfluenceIndexes.size();
	#pragma omp parallel for
	for (i = 0; i < vInfluenceIndexes.size(); i++) {
		influenceIndexes[i + 1] = vInfluenceIndexes[i];
	}

	*_influenceValues = new double[vInfluenceValues.size() + 1];
	double* influenceValues = *_influenceValues;
	influenceValues[0] = vInfluenceValues.size();
	#pragma omp parallel for
	for (i = 0; i < vInfluenceValues.size(); i++) {
		influenceValues[i + 1] = vInfluenceValues[i];
	}
	free(levelCount);
	levelCount = NULL;
}