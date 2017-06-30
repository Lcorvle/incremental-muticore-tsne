#pragma once
#include "Eigen/Eigen"
#include <iostream>
using namespace Eigen;
using namespace std;

class Level {
	int levelId, levelSize, preLevelSize;
	SparseMatrix<double> transitionMatrix, weightMatrix, influenceMatrix;
	vector<int> levelIndexes;
	Level *next;
	Level *pre;
	

public:
	Level(int id, int size, int preSize) {
		levelId = id;
		levelSize = size;
		preLevelSize = preSize;
		next = NULL;
		pre = NULL;
	};
	int getID();
	int getSize();
	int getPreSize();
	SparseMatrix<double> getTransitionMatrix();
	SparseMatrix<double> getWeight();
	SparseMatrix<double> getInfluenceMatrix();
	vector<int> getIndexes();
	Level* getNext();
	Level* getPre();
	void setNext(Level* nextLevel);
	void initData(Level* preLevel, SparseMatrix<double> transition, SparseMatrix<double> weight, SparseMatrix<double> influence, vector<int> indexes);
};