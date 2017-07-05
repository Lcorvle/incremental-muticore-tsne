/*
*  quadtree.cpp
*  Implementation of a quadtree in two dimensions + Barnes-Hut algorithm for t-SNE.
*
*  Created by Laurens van der Maaten.
*  Copyright 2012, Delft University of Technology. All rights reserved.
*
*  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
*/

#include "quadtree.h"



// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{
	if (x - hw > point[0]) return false;
	if (x + hw < point[0]) return false;
	if (y - hh > point[1]) return false;
	if (y + hh < point[1]) return false;
	return true;
}


// Default constructor for quadtree -- build tree, too!
QuadTree::QuadTree(double* inpData, int N)
{

	// Compute mean, width, and height of current map (boundaries of quadtree)
	double* meanY = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++) meanY[d] = .0;
	double*  minY = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++)  minY[d] = DBL_MAX;
	double*  maxY = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++)  maxY[d] = -DBL_MAX;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < QT_NO_DIMS; d++) {
			meanY[d] += inpData[n * QT_NO_DIMS + d];
			if (inpData[n * QT_NO_DIMS + d] < minY[d]) minY[d] = inpData[n * QT_NO_DIMS + d];
			if (inpData[n * QT_NO_DIMS + d] > maxY[d]) maxY[d] = inpData[n * QT_NO_DIMS + d];
		}
	}
	for (int d = 0; d < QT_NO_DIMS; d++) meanY[d] /= (double)N;

	// Construct quadtree
	init(NULL, inpData, meanY[0], meanY[1], max(maxY[0] - meanY[0], meanY[0] - minY[0]) + 1e-5,
		max(maxY[1] - meanY[1], meanY[1] - minY[1]) + 1e-5);
	fill(N);
	delete[] meanY; delete[] maxY; delete[] minY;
}


// Constructor for quadtree with particular size and parent -- build the tree, too!
QuadTree::QuadTree(double* inpData, int N, double inpX, double inpY, double inpHw, double inpHh)
{
	init(NULL, inpData, inpX, inpY, inpHw, inpHh);
	fill(N);
}

// Constructor for quadtree with particular size and parent -- build the tree, too!
QuadTree::QuadTree(QuadTree* inpParent, double* inpData, int N, double inpX, double inpY, double inpHw, double inpHh)
{
	init(inpParent, inpData, inpX, inpY, inpHw, inpHh);
	fill(N);
}


// Constructor for quadtree with particular size (do not fill the tree)
QuadTree::QuadTree(double* inpData, double inpX, double inpY, double inpHw, double inpHh)
{
	init(NULL, inpData, inpX, inpY, inpHw, inpHh);
}


// Constructor for quadtree with particular size and parent (do not fill the tree)
QuadTree::QuadTree(QuadTree* inpParent, double* inpData, double inpX, double inpY, double inpHw, double inpHh)
{
	init(inpParent, inpData, inpX, inpY, inpHw, inpHh);
}


// Main initialization function
void QuadTree::init(QuadTree* inpParent, double* inpData, double inpX, double inpY, double inpHw, double inpHh)
{
	parent = inpParent;
	data = inpData;
	isLeaf = true;
	size = 0;
	cumSize = 0;
	boundary.x = inpX;
	boundary.y = inpY;
	boundary.hw = inpHw;
	boundary.hh = inpHh;
	northWest = NULL;
	northEast = NULL;
	southWest = NULL;
	southEast = NULL;
	for (int i = 0; i < QT_NO_DIMS; i++) {
		centerOfMass[i] = .0;
	}
}


// Destructor for quadtree
QuadTree::~QuadTree()
{
	delete northWest;
	delete northEast;
	delete southWest;
	delete southEast;
}


// Update the data underlying this tree
void QuadTree::setData(double* inpData)
{
	data = inpData;
}


// Get the parent of the current tree
QuadTree* QuadTree::getParent()
{
	return parent;
}


// Insert a point into the QuadTree
bool QuadTree::insert(int newIndex)
{
	// Ignore objects which do not belong in this quad tree
	double* point = data + newIndex * QT_NO_DIMS;
	if (!boundary.containsPoint(point))
		return false;

	// Online update of cumulative size and center-of-mass
	cumSize++;
	double mult1 = (double)(cumSize - 1) / (double)cumSize;
	double mult2 = 1.0 / (double)cumSize;
	for (int d = 0; d < QT_NO_DIMS; d++) {
		centerOfMass[d] = centerOfMass[d] * mult1 + mult2 * point[d];
	}

	// If there is space in this quad tree and it is a leaf, add the object here
	if (isLeaf && size < QT_NODE_CAPACITY) {
		index[size] = newIndex;
		size++;
		return true;
	}

	// Don't add duplicates for now (this is not very nice)
	bool anyDuplicate = false;
	for (int n = 0; n < size; n++) {
		bool duplicate = true;
		for (int d = 0; d < QT_NO_DIMS; d++) {
			if (point[d] != data[index[n] * QT_NO_DIMS + d]) { duplicate = false; break; }
		}
		anyDuplicate = anyDuplicate | duplicate;
	}
	if (anyDuplicate) return true;

	// Otherwise, we need to subdivide the current cell
	if (isLeaf) subdivide();

	// Find out where the point can be inserted
	if (northWest->insert(newIndex)) return true;
	if (northEast->insert(newIndex)) return true;
	if (southWest->insert(newIndex)) return true;
	if (southEast->insert(newIndex)) return true;

	// Otherwise, the point cannot be inserted (this should never happen)
	return false;
}


// Create four children which fully divide this cell into four quads of equal area
void QuadTree::subdivide() {

	// Create four children
	northWest = new QuadTree(this, data, boundary.x - .5 * boundary.hw, boundary.y - .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
	northEast = new QuadTree(this, data, boundary.x + .5 * boundary.hw, boundary.y - .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
	southWest = new QuadTree(this, data, boundary.x - .5 * boundary.hw, boundary.y + .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
	southEast = new QuadTree(this, data, boundary.x + .5 * boundary.hw, boundary.y + .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);

	// Move existing points to correct children
	for (int i = 0; i < size; i++) {
		bool success = false;
		if (!success) success = northWest->insert(index[i]);
		if (!success) success = northEast->insert(index[i]);
		if (!success) success = southWest->insert(index[i]);
		if (!success) success = southEast->insert(index[i]);
		index[i] = -1;
	}

	// Empty parent node
	size = 0;
	isLeaf = false;
}


// Build quadtree on dataset
void QuadTree::fill(int N)
{
	for (int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
bool QuadTree::isCorrect()
{
	for (int n = 0; n < size; n++) {
		double* point = data + index[n] * QT_NO_DIMS;
		if (!boundary.containsPoint(point)) return false;
	}
	if (!isLeaf) return northWest->isCorrect() &&
		northEast->isCorrect() &&
		southWest->isCorrect() &&
		southEast->isCorrect();
	else return true;
}


// Rebuilds a possibly incorrect tree (LAURENS: This function is not tested yet!)
void QuadTree::rebuildTree()
{
	for (int n = 0; n < size; n++) {

		// Check whether point is erroneous
		double* point = data + index[n] * QT_NO_DIMS;
		if (!boundary.containsPoint(point)) {

			// Remove erroneous point
			int remIndex = index[n];
			for (int m = n + 1; m < size; m++) index[m - 1] = index[m];
			index[size - 1] = -1;
			size--;

			// Update center-of-mass and counter in all parents
			bool done = false;
			QuadTree* node = this;
			while (!done) {
				for (int d = 0; d < QT_NO_DIMS; d++) {
					node->centerOfMass[d] = ((double)node->cumSize * node->centerOfMass[d] - point[d]) / (double)(node->cumSize - 1);
				}
				node->cumSize--;
				if (node->getParent() == NULL) done = true;
				else node = node->getParent();
			}

			// Reinsert point in the root tree
			node->insert(remIndex);
		}
	}

	// Rebuild lower parts of the tree
	northWest->rebuildTree();
	northEast->rebuildTree();
	southWest->rebuildTree();
	southEast->rebuildTree();
}


// Build a list of all indices in quadtree
void QuadTree::getAllIndices(int* indices)
{
	getAllIndices(indices, 0);
}


// Build a list of all indices in quadtree
int QuadTree::getAllIndices(int* indices, int loc)
{

	// Gather indices in current quadrant
	for (int i = 0; i < size; i++) indices[loc + i] = index[i];
	loc += size;

	// Gather indices in children
	if (!isLeaf) {
		loc = northWest->getAllIndices(indices, loc);
		loc = northEast->getAllIndices(indices, loc);
		loc = southWest->getAllIndices(indices, loc);
		loc = southEast->getAllIndices(indices, loc);
	}
	return loc;
}


int QuadTree::getDepth() {
	if (isLeaf) return 1;
	return 1 + max(max(northWest->getDepth(),
		northEast->getDepth()),
		max(southWest->getDepth(),
			southEast->getDepth()));

}


// Compute non-edge forces using Barnes-Hut algorithm
void QuadTree::computeNonEdgeForces(int pointIndex, double theta, double negF[], double* sumQ, double buff[])
{

	// Make sure that we spend no time on empty nodes or self-interactions
	if (cumSize == 0 || (isLeaf && size == 1 && index[0] == pointIndex)) return;

	// Compute distance between point and center-of-mass
	double D = .0;
	int ind = pointIndex * QT_NO_DIMS;


	for (int d = 0; d < QT_NO_DIMS; d++) {
		buff[d] = data[ind + d];
		buff[d] -= centerOfMass[d];
		D += buff[d] * buff[d];
	}

	// Check whether we can use this node as a "summary"
	if (isLeaf || max(boundary.hh, boundary.hw) / sqrt(D) < theta) {

		// Compute and add t-SNE force between point and current node
		double Q = 1.0 / (1.0 + D);
		*sumQ += cumSize * Q;
		double mult = cumSize * Q * Q;
		for (int d = 0; d < QT_NO_DIMS; d++)
			negF[d] += mult * buff[d];
	}
	else {
		// Recursively apply Barnes-Hut to children
		northWest->computeNonEdgeForces(pointIndex, theta, negF, sumQ, buff);
		northEast->computeNonEdgeForces(pointIndex, theta, negF, sumQ, buff);
		southWest->computeNonEdgeForces(pointIndex, theta, negF, sumQ, buff);
		southEast->computeNonEdgeForces(pointIndex, theta, negF, sumQ, buff);
	}
}


// Computes edge forces
void QuadTree::computeEdgeForces(int* rowP, int* colP, double* valP, int N, double* posF, int oldNum)
{

	// Loop over all edges in the graph
	int ind1, ind2;
	double D;
	double buff[QT_NO_DIMS];

	for (int n = oldNum; n < N; n++) {
		ind1 = n * QT_NO_DIMS;
		for (int i = rowP[n]; i < rowP[n + 1]; i++) {

			// Compute pairwise distance and Q-value
			D = .0;
			ind2 = colP[i] * QT_NO_DIMS;
			for (int d = 0; d < QT_NO_DIMS; d++) {
				buff[d] = data[ind1 + d];
				buff[d] -= data[ind2 + d];
				D += buff[d] * buff[d];

			}
			D = valP[i] / (1.0 + D);

			// Sum positive force
			for (int d = 0; d < QT_NO_DIMS; d++) {
				posF[ind1 + d] += D * buff[d];
			}
		}
	}
}


// Print out tree
void QuadTree::print()
{
	if (cumSize == 0) {
		printf("Empty node\n");
		return;
	}

	if (isLeaf) {
		printf("Leaf node; data = [");
		for (int i = 0; i < size; i++) {
			double* point = data + index[i] * QT_NO_DIMS;
			for (int d = 0; d < QT_NO_DIMS; d++) printf("%f, ", point[d]);
			printf(" (index = %d)", index[i]);
			if (i < size - 1) printf("\n");
			else printf("]\n");
		}
	}
	else {
		printf("Intersection node with center-of-mass = [");
		for (int d = 0; d < QT_NO_DIMS; d++) printf("%f, ", centerOfMass[d]);
		printf("]; children are:\n");
		northEast->print();
		northWest->print();
		southEast->print();
		southWest->print();
	}
}

