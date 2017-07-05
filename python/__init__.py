import numpy as np
import cffi
import psutil
import threading
import os
import sys
from ctypes import *

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    '''
        Only
            - nComponents
            - perplexity
            - angle
            - nIter
        parameters are used.
        Other are left for compatibility with sklearn TSNE.
    '''

    def __init__(self,
                 nComponents=2,
                 perplexity=30.0,
                 earlyExaggeration=4.0,
                 learningRate=1000.0,
                 nIter=1000,
                 nIterWithoutProgress=30,
                 minGradNorm=1e-07,
                 metric='euclidean',
                 init='random',
                 verbose=0,
                 randomState=None,
                 method='barnes_hut',
                 angle=0.5,
                 nJobs=1):
        self.data = {}
        self.nComponents = nComponents
        self.angle = angle
        self.perplexity = perplexity
        self.nIter = nIter
        self.nJobs = nJobs
        self.randomState = -1 if randomState is None else randomState

        assert nComponents == 2, 'nComponents should be 2'

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            "void incrementalTsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState, int oldNum);"
            "void tsneRunDouble(double* X, int N, int D, double* Y, int noDims, double perplexity, double theta, int _numThreads, int maxIter, int randomState);"
            "void landMarkSampling(int threads, int perp, int randWalksNum, int randWalksLength, int randWalksThrehold, int endSize, double* data, int rowNum, int dim, int** _levelSizes, int** _indexes, int** _levelInfluenceSizes, int** _pointInfluenceSizes, int** _influenceIndexes, int** _topLevelIndexes, double** _influenceValues);"
            "void getInfluenceIndexes(int threads, int* levelSizes, int* indexes, int* levelInfluenceSizes, int* pointInfluenceSizes, int* influenceIndexes, double* influenceValues, int *indexSet, int size, double minInfluenceValue, int** _resultSet);")

        path = os.path.dirname(os.path.realpath(__file__))
        self.C = self.ffi.dlopen(path + "/libtsne_multicore.dll")

    def fitTransform(self, X, oldNum=0, oldY=[]):

        assert X.ndim == 2, 'X should be 2D array.'
        assert X.dtype == np.float64, 'Only double arrays are supported for now. Use .astype(np.float64) to convert.'

        if self.nJobs == -1:
            self.nJobs = psutil.cpu_count()

        assert self.nJobs > 0, 'Wrong nJobs parameter.'

        N, D = X.shape
        Y = np.zeros((N, self.nComponents))
        perp = 0
        if N > self.perplexity * 3:
            perp = self.perplexity
        else:
            perp = int((N - 1) / 3)
        if oldNum == 0:
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)

            t = FuncThread(self.C.tsneRunDouble,
                           cffiX, N, D,
                           cffiY, self.nComponents,
                           perp, self.angle, self.nJobs, self.nIter, self.randomState)

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
        else:
            Y[0:oldNum] += oldY
            cffiX = self.ffi.cast('double*', X.ctypes.data)
            cffiY = self.ffi.cast('double*', Y.ctypes.data)

            t = FuncThread(self.C.incrementalTsneRunDouble,
                           cffiX, N, D,
                           cffiY, self.nComponents,
                           perp, self.angle, self.nJobs, self.nIter, self.randomState, oldNum)

            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()

        return Y

    def getInfluenceIndexes(self, indexes, minInfluenceValue):
        assert self.data != {}, "You should landMarkSampling first."
        array = np.array(indexes)
        indexSet = self.ffi.cast('int*', array.ctypes.data)
        resultSet = self.ffi.new('int**')
        t = FuncThread(self.C.getInfluenceIndexes, self.nJobs, self.data['levelSizes'], self.data['indexes'],
                       self.data['levelInfluenceSizes'], self.data['pointInfluenceSizes'],
                       self.data['influenceIndexes'], self.data['influenceValues'], indexSet, len(indexes),
                       minInfluenceValue, resultSet)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        tempResult = resultSet[0]
        n = tempResult[0]
        result = []
        for i in range(n):
            result.append(tempResult[i + 1])
        return result

    def getLevelNumber(self):
        assert self.data != {}, "You should landMarkSampling first."
        return self.data['levelSizes'][0] - 1

    def getLevelIndexes(self, levelID):
        assert self.data != {}, "You should landMarkSampling first."
        num = self.data['levelSizes'][0]
        result = []
        if levelID == num - 1:
            temp = self.data['levelSizes'][num] - self.data['levelSizes'][num - 1]
            for i in range(temp):
                result.append(i)
            return result
        i = self.data['levelSizes'][levelID]
        while i < self.data['levelSizes'][levelID + 1]:
            result.append(self.data['indexes'][i])
            i += 1
        return result

    def landmarkSampling(self, X, endSize, randWalksNum=100, randWalksLength=50, randWalksThrehold=100):
        assert X.dtype == np.float64, 'Only double arrays are supported for now. Use .astype(np.float64) to convert.'

        rowNum, dim = X.shape
        cffiX = self.ffi.cast('double*', X.ctypes.data)
        levelSizes = self.ffi.new('int**')
        indexes = self.ffi.new('int**')
        levelInfluenceSizes = self.ffi.new('int**')
        pointInfluenceSizes = self.ffi.new('int**')
        influenceIndexes = self.ffi.new('int**')
        topLevelIndexes = self.ffi.new('int**')
        influenceValues = self.ffi.new('double**')

        perp = 0
        if rowNum > self.perplexity * 3:
            perp = self.perplexity
        else:
            perp = int(rowNum / 3)
        t = FuncThread(self.C.landMarkSampling, self.nJobs, perp, randWalksNum, randWalksLength,
                       randWalksThrehold, endSize, cffiX, rowNum, dim, levelSizes, indexes, levelInfluenceSizes,
                       pointInfluenceSizes, influenceIndexes, topLevelIndexes, influenceValues)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        self.data['levelSizes'] = levelSizes[0]
        self.data['indexes'] = indexes[0]
        self.data['levelInfluenceSizes'] = levelInfluenceSizes[0]
        self.data['pointInfluenceSizes'] = pointInfluenceSizes[0]
        self.data['influenceIndexes'] = influenceIndexes[0]
        self.data['influenceValues'] = influenceValues[0]
        tempResult = topLevelIndexes[0]
        n = tempResult[0]
        result = []
        for i in range(n):
            result.append(tempResult[i + 1])
        return result