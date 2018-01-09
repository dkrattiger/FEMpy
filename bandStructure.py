import numpy as np
#import scipy.sparse as sp
from scipy.sparse import csr_matrix
import scipy
import copy
import cmath
import math
import time


def blochPeriodicity(X, R):
    """ form transformation matrices to enforce Bloch periodicity"""

    iFree = findNodeSets(X, R)

    def sliceMap(iList, count):
        """
        form a list of node indices with the same shape/structure as "iList",
        but that remaps to the Periodic node indices
        """

        iMap = [[] for i in range(3)]
        for i in range(2):

            # if current list level contains a vector of indices (in a numpy
            # array) remap the indices
            if isinstance(iList[i], np.ndarray):

                iMap[i] = count + np.arange(0, np.size(iList[i]))
                count = count + np.size(iList[i])

            # if current list level contains a list itself, recurse on that
            # list item
            else:

                iMap[i], count = sliceMap(iList[i], count)

        iMap[2] = copy.deepcopy(iMap[0])

        return iMap, count

    iPer, nNodesBC = sliceMap(iFree, 0)

    # get array sizes and number of periodicity dimensions
    nNodes, nDim = np.shape(X)
    nDim2, nPer = np.shape(R)
    nDPN = nDim

    nDOF = nDPN*nNodes
    nDOFBC = nDPN*nNodesBC

    def perMat(iSet, iMap):
        """
        define periodicity transformation for each vector in iMap/iList
        """

        #  Tper = [[] for in in range(2)]
        #  Tper = [lil_matrix((nDOF, nDOFBC)) for i in range(2)]
        Tper = [[] for i in range(2)]

        # recurse over list items until innermost level is reached
        if isinstance(iSet[0], np.ndarray):

            # add "left" and "middle" DOFs to first periodicity matrix
            row = node2DOF(np.concatenate((iSet[0], iSet[1])), nDPN)
            col = node2DOF(np.concatenate((iMap[0], iMap[1])), nDPN)
            data = np.ones(np.shape(row))
            Tper[0] = csr_matrix((data, (row, col)), shape=(nDOF, nDOFBC))

            # Add "right" side DOFs to second Periodicity matrix
            row = node2DOF(iSet[2], nDPN)
            col = node2DOF(iMap[2], nDPN)
            data = np.ones(np.shape(row))
            Tper[1] = csr_matrix((data, (row, col)), shape=(nDOF, nDOFBC))
        else:

            TperA = perMat(iSet[0], iMap[0])
            TperB = perMat(iSet[1], iMap[1])

            Tper[0] = listAdd(perMat(iSet[0], iMap[0]),
                              perMat(iSet[1], iMap[1]))
            Tper[1] = perMat(iSet[2], iMap[2])

        return Tper

    Tper = perMat(iFree, iPer)
    return Tper


def listAdd(x, y):
    """
    take two lists with identical shape/structure and perform element by
    element addition
    """

    if isinstance(x, list):
        z = [listAdd(i, j) for i, j in zip(x, y)]
    else:
        z = x+y

    return z


def listMult(x, y):
    """
    take two lists with identical shape/structure and perform element by
    element multiplication
    """
    if isinstance(x, list):
        z = [listMult(x, y) for i, j in zip(x, y)]
    else:
        z = x*y

    return z


def asvoid(arr):
    """
    View the array as dtype np.void (bytes)
    This views the last axis of ND-arrays as bytes so you can perform comparisons on
    the entire row.
    http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
    Warning: When using asvoid for comparison, note that float zeros may compare UNEQUALLY
    >>> asvoid([-0.]) == asvoid([0.])
    array([False], dtype=bool)
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def in1d_index(a, b):
    """
    find indices of rows in a that occur in b
    """

    voida, voidb = map(asvoid, (a, b))
    Lib = np.where(np.in1d(voidb, voida))[0]
    Lia = np.where(np.in1d(voida, voidb))[0]

    a = a[Lia, :]
    b = b[Lib, :]

    isort = np.nonzero(np.all(b == a[:, np.newaxis], axis=2))[1]

    Lib = Lib[isort]
    #  return np.nonzero(np.all(b == a[:, np.newaxis], axis=2))[1]

    return Lia, Lib


def findNodeSets(X, R):
    """
    takes coordinates, X, and lattice vectors, R, and returns a nested list
    that contains the different unit cell boundary node sets
    """

    # find range of values in X
    r = X.max(0)-X.min(0)

    # space data out and convert to integer
    rnew = 1e6

    X = np.round(X*(rnew/r), 0)
    X = X.astype(int)

    R = np.round(R*(rnew/r), 0)
    R = R.astype(int)

    # get array sizes and number of periodicity dimensions
    nNodes, nDim = np.shape(X)
    nDim2, nPer = np.shape(R)

    # initialize Lia and Lib storage arrays
    Lias = []
    Libs = []

    for i in range(nPer):

        #  Xshift = X + np.transpose(R[:, i])*np.ones((nNodes, 1))
        Xshift = X + R[:, i]

        #  [Lia, Lib] = ismember(X, Xshift)
        [Lia, Lib] = in1d_index(X, Xshift)
        Lias.append(Lia)
        Libs.append(Lib)

    # function that divides each member in list "iSet" into a sublist with
    # three entries (left, middle, and right) based on the original set's
    # intersection with the left and right sets, "Lia" and "Lib"
    def sliceNodes(iList, iL, iR):

        if isinstance(iList, np.ndarray):

            iList = [np.intersect1d(iList, iL),
                     np.setdiff1d(np.setdiff1d(iList, iL), iR),
                     np.intersect1d(iList, iR)]

        else:
            for i in range(0, len(iList)):
                iList[i] = sliceNodes(iList[i], iL, iR)

        return iList

    iSet = np.arange(0, nNodes)
    for i in range(nPer):
        iSet = sliceNodes(iSet, Lias[i], Libs[i])

    return iSet


def node2DOF(nodeIndex, nDPN):

    DOFIndex = np.zeros(np.size(nodeIndex)*nDPN, int)
    for i in range(nDPN):
        DOFIndex[i::nDPN] = nDPN*nodeIndex+i

    return DOFIndex


def dispersion_w_k(K, M, X, R, kappa, nbands):

    # compute Bloch periodicity matrices
    Tper = blochPeriodicity(X, R)

    # number of k points
    nkap = len(kappa)

    # number of degrees of freedom
    nDOF = np.shape(K)[0]

    # compute band structure frequencies
    w = np.zeros((nbands, nkap))
    Phi = np.zeros((nDOF, nbands, nkap),complex)

    print(nDOF, "'free' Degrees of freedom")
    for i in range(0, nkap):
        startTime = time.time()

        #  stime = time.time()
        TperK = TperKappa(Tper, kappa[i], R)
        #  print(time.time()-stime)

        #  Khat = np.dot(np.transpose(TperK),np.dot(K,TperK))
        #  Mhat = np.dot(np.transpose(TperK),np.dot(M,TperK))

        #  stime = time.time()
        Khat = np.dot(TperK.conj().T, np.dot(K, TperK))
        Mhat = np.dot(TperK.conj().T, np.dot(M, TperK))
        #  print(time.time()-stime)

        # compute eigenvalues
        L, P = scipy.sparse.linalg.eigsh(Khat, M=Mhat, k=nbands, sigma=0)

        # sort eignevalues and mode shapes
        isort = np.argsort(L)
        L = L[isort]
        P = P[:, isort]

        # store disperseion frequencies
        w[:, i] = np.sqrt(np.absolute(L))

        # apply periodicity transformation and store mode shape
        Phi[:, :, i] = csr_matrix.dot(TperK, P)

        print(
            "Loop",
            i +
            1,
            "of",
            nkap,
            ", calculation time",
            time.time() -
            startTime,
            "seconds")

    return w, Phi


def waveVector(highsym, pwr):

    kappa = []
    nkapseg = 2**pwr
    nhighsym = len(highsym)
    nkap = nkapseg*(nhighsym-1)+1
    kappaplot = np.zeros(nkap)
    for i in range(nhighsym-1):
        for j in range(nkapseg):
            kappanext = highsym[i] + (highsym[i+1]-highsym[i])*(j/nkapseg)
            kappa.append(kappanext)

    kappa.append(highsym[-1])

    return kappa, kappaplot


def TperKappa(Tper, kappa, R):
    """
    Takes a set of Bloch periodicity transformation matrices and adds them with
    phase terms corresponding to kappa
    """

    if isinstance(Tper[0], list):

        lambdai = cmath.exp(1j*np.dot(kappa, R[:, 0]))
        Tperk = TperKappa(Tper[0], kappa, R[:, 1::]
                          ) + lambdai*TperKappa(Tper[1], kappa, R[:, 1::])

    else:

        lambdai = cmath.exp(1j*np.dot(kappa, R[:, 0]))
        Tperk = Tper[0] + lambdai*Tper[1]

    return Tperk
