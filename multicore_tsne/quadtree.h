/*
*  quadtree.h
*  Header file for a quadtree.
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

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

class Cell {

public:
	double x;
	double y;
	double hw;
	double hh;
	bool   containsPoint(double point[]);
};


class QuadTree
{

	// Fixed constants
	static const int QT_NO_DIMS = 2;
	static const int QT_NODE_CAPACITY = 1;


	// Properties of this node in the tree
	QuadTree* parent;
	bool isLeaf;
	int size;
	int cumSize;

	// Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
	Cell boundary;

	// Indices in this quad tree node, corresponding center-of-mass, and list of all children
	double* data;
	double centerOfMass[QT_NO_DIMS];
	int index[QT_NODE_CAPACITY];

	// Children
	QuadTree* northWest;
	QuadTree* northEast;
	QuadTree* southWest;
	QuadTree* southEast;

public:
	QuadTree(double* inpData, int N);
	QuadTree(double* inpData, double inpX, double inpY, double inpHw, double inpHh);
	QuadTree(double* inpData, int N, double inpX, double inpY, double inpHw, double inpHh);
	QuadTree(QuadTree* inpParent, double* inpData, int N, double inpX, double inpY, double inpHw, double inpHh);
	QuadTree(QuadTree* inpParent, double* inpData, double inpX, double inpY, double inpHw, double inpHh);
	~QuadTree();
	void setData(double* inpData);
	QuadTree* getParent();
	void construct(Cell boundary);
	bool insert(int newIndex);
	void subdivide();
	bool isCorrect();
	void rebuildTree();
	void getAllIndices(int* indices);
	int getDepth();
	void computeNonEdgeForces(int pointIndex, double theta, double negF[], double* sumQ, double buff[]);
	void computeEdgeForces(int* rowP, int* colP, double* valP, int N, double* posF, int oldNum);
	void print();

private:
	void init(QuadTree* inpParent, double* inpData, double inpX, double inpY, double inpHw, double inpHh);
	void fill(int N);
	int getAllIndices(int* indices, int loc);
	bool isChild(int testIndex, int start, int end);
};
