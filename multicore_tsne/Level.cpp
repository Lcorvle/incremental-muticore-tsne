#include "Level.h"
int Level::getID() {
	return levelId;
}

int Level::getSize() {
	return levelSize;
}

int Level::getPreSize() {
	return preLevelSize;
}

SparseMatrix<double> Level::getTransitionMatrix() {
	return transitionMatrix;
}

SparseMatrix<double> Level::getWeight() {
	return weightMatrix;
}

SparseMatrix<double> Level::getInfluenceMatrix() {
	return influenceMatrix;
}

vector<int> Level::getIndexes() {
	return levelIndexes;
}

Level* Level::getNext() {
	return next;
}

Level* Level::getPre() {
	return pre;
}

void Level::setNext(Level* nextLevel) {
	next = nextLevel;
	return;
}

void Level::initData(Level* preLevel, SparseMatrix<double> transition, SparseMatrix<double> weight, SparseMatrix<double> influence, vector<int> indexes) {
	pre = preLevel;
	transitionMatrix = transition;
	weightMatrix = weight;
	influenceMatrix = influence;
	levelIndexes = indexes;
	return;
}