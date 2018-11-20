import numpy as np
# import scipy.sparse as sp
from scipy.sparse import csr_matrix
#  from scipy import sparse
import scipy
import copy
import cmath
import math
import time
import matplotlib.pylab as plt


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

    nDOF = nDPN * nNodes
    nDOFBC = nDPN * nNodesBC

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
        z = x + y

    return z


def listMult(x, y):
    """
    take two lists with identical shape/structure and perform element by
    element multiplication
    """
    if isinstance(x, list):
        z = [listMult(x, y) for i, j in zip(x, y)]
    else:
        z = x * y

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
    r = X.max(0) - X.min(0)

    # space data out and convert to integer
    rnew = 1e6

    X = np.round(X * (rnew / r), 0)
    X = X.astype(int)

    R = np.round(R * (rnew / r), 0)
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
    DOFIndex = np.zeros(np.size(nodeIndex) * nDPN, int)
    for i in range(nDPN):
        DOFIndex[i::nDPN] = nDPN * nodeIndex + i

    return DOFIndex


def dispersion_w_k(K, M, X, R, kappa, nbands=10, expandmodeshapes=True, verbose=False):
    # compute Bloch periodicity matrices
    Tper = blochPeriodicity(X, R)

    # number of k points
    nkap = len(kappa)

    # number of degrees of freedom
    #  nDOF = np.shape(K)[0]
    #  nDOFper = np.shape(Tper[0])[1]
    temp = Tper
    while isinstance(temp, list):
        temp = temp[0]
    nDOF, nDOFper = np.shape(temp)

    # compute band structure frequencies
    w = np.zeros((nbands, nkap))
    if expandmodeshapes:
        Phi = np.zeros((nDOF, nbands, nkap), complex)
    else:
        Phi = np.zeros((nDOFper, nbands, nkap), complex)

    if verbose:
        print(nDOF, "'free' Degrees of freedom")

    for i in range(0, nkap):
        startTime = time.time()

        # apply Bloch boundary conditions to mass and stiffness matrices
        Khat, Mhat = KMBBC(K, M, Tper, kappa[i], R)

        # compute eigenvalues
        L, P = scipy.sparse.linalg.eigsh(Khat, M=Mhat, k=nbands, sigma=0)

        # sort eigenvalues and mode shapes
        isort = np.argsort(L)
        L = L[isort]
        P = P[:, isort]

        # store disperseion frequencies
        w[:, i] = np.sqrt(np.absolute(L))

        # apply periodicity transformation and store mode shape
        TperK = TperKappa(Tper, kappa[i], R)

        # expand mode shapes with periodicity transformation and store them
        if expandmodeshapes:
            Phi[:, :, i] = TperK @ P
        else:
            Phi[:, :, i] = P

        if verbose:
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
    nkapseg = 2 ** pwr
    nhighsym = len(highsym)
    nkap = nkapseg * (nhighsym - 1) + 1
    for i in range(nhighsym - 1):
        for j in range(nkapseg):
            kappanext = highsym[i] + (highsym[i + 1] - highsym[i]) * (j / nkapseg)
            kappa.append(kappanext)

    kappa.append(highsym[-1])

    # make a plotting vector
    kappaplot = np.zeros(nkap)
    for i in range(nkap - 1):
        kappaplot[i + 1] = kappaplot[i] + np.linalg.norm(kappa[i + 1] - kappa[i])
    kappaplot = kappaplot / kappaplot[-1]

    return kappa, kappaplot


def waveVectorRBME(highsym, pwr, pwrRBME):
    nkapseg = 2 ** pwr
    nhighsym = len(highsym)
    nkap = nkapseg * (nhighsym - 1) + 1
    #  kappaplot = np.zeros(nkap)

    # make segmented kappa vector
    kappa = []
    for i in range(nhighsym - 1):
        kappa.append([])
        for j in range(nkapseg):
            kappanext = highsym[i] + (highsym[i + 1] - highsym[i]) * (j / nkapseg)
            kappa[i].append(kappanext)

    kappa[i].append(highsym[-1])

    # make a plotting vector
    #  kappaplot = np.zeros(nkap)
    kappaplot = [0]
    kappalast = kappa[0][0]
    for i in range(len(kappa)):
        for j in range(len(kappa[i])):
            kappanext = kappa[i][j]
            kappaplot.append(kappaplot[-1] + np.linalg.norm(kappanext - kappalast))
            kappalast = kappanext
    kappaplot = kappaplot / kappaplot[-1]
    kappaplot = kappaplot[1:]

    kappaExp = []
    nkapseg = 2 ** pwrRBME
    nhighsym = len(highsym)
    nkap = nkapseg * (nhighsym - 1) + 1
    for i in range(nhighsym - 1):
        kappaExp.append([])
        for j in range(nkapseg):
            kappanext = highsym[i] + (highsym[i + 1] - highsym[i]) * (j / nkapseg)
            kappaExp[i].append(kappanext)

        kappaExp[i].append(highsym[i + 1])

    #  kappaExp[i].append(highsym[-1])
    return kappa, kappaExp, kappaplot


def TperKappa(Tper, kappa, R):
    """
    Takes a set of Bloch periodicity transformation matrices and adds them with
    phase terms corresponding to kappa. A recursive form is used so that the
    function works for any number of dimensions.
    """

    if isinstance(Tper[0], list):

        lambdai = cmath.exp(1j * np.dot(kappa, R[:, 0]))
        Tperk = TperKappa(Tper[0], kappa, R[:, 1::]
                          ) + lambdai * TperKappa(Tper[1], kappa, R[:, 1::])

    else:

        # base case, add T0+(lambda)*T1 along the last dimension
        lambdai = cmath.exp(1j * np.dot(kappa, R[:, 0]))
        Tperk = Tper[0] + lambdai * Tper[1]

    return Tperk


def KMBBC(K, M, Tper, kappa, R):
    """
    form the Bloch periodic mass and stiffness matrices by applying Bloch
    boundary conditions to the free matrices
    """
    Tperk = TperKappa(Tper, kappa, R)

    #  stime = time.time()
    #  return  np.dot(TperK.conj().T, np.dot(K, TperK)), np.dot(TperK.conj().T, np.dot(M, TperK))
    Khat = Tperk.conj().T @ K @ Tperk
    Mhat = Tperk.conj().T @ M @ Tperk

    Khat = (1 / 2) * (Khat + Khat.conj().T)
    Mhat = (1 / 2) * (Mhat + Mhat.conj().T)

    return Khat, Mhat


def diffKMBBC(K, M, Tper, kappa, R, v):
    """
    Compute the directional derivate of the mass and stiffness matrices
    with respect to the wave vector in direction "v"
    """
    v = v / np.linalg.norm(v)

    def diffT(Tper, R):
        """This function is called recursively to generate the derivative of
        the periodicity transformation matrix (Tperk) with respect to each
        wavevector component"""

        if isinstance(Tper[0], list):

            # The "0" component is the part that is not multiplied by the phase
            # term. Compute first, and second derivatves...hopefully third
            # derivative won't become necessary - but it might :(
            Tperk0, dTperk0, ddTperk0 = diffT(Tper[0], R[:, 1::])
            Tperk1, dTperk1, ddTperk1 = diffT(Tper[1], R[:, 1::])

            # phase term (from Bloch's theorem)
            lambdal = cmath.exp(1j * np.dot(kappa, R[:, 0]))

            # assemble Bloch periodicity transformation matrix at current level
            Tperk = Tperk0 + lambdal * Tperk1

            # initialize Bloch periodicity transformation matrix derivative and
            # loop through the wave vector components
            dTperk = []
            ddTperk = []
            for i in range(np.shape(R)[0]):
                dlamdki = lambdal * 1j * R[i, 0]
                dTperk.append(dTperk0[i] + dlamdki * Tperk1 + lambdal * dTperk1[i])

                # second derivative base case
                ddTperk.append([])
                for j in range(np.shape(R)[0]):
                    #  d2lamdkidkj = -lambdal*R[i,0]*R[j,0]
                    dlamdkj = lambdal * 1j * R[j, 0]
                    ddlamdkidkj = dlamdki * 1j * R[j, 0]
                    ddTperk[i].append(ddTperk0[i][j] + ddlamdkidkj * Tperk1 +
                                      dlamdki * dTperk1[j] + dlamdkj * dTperk1[i] +
                                      ddTperk1[i][j])

        else:

            # base case, add T0+(lambda)*T1 along the last dimension
            lambdal = cmath.exp(1j * np.dot(kappa, R[:, 0]))
            Tperk = Tper[0] + lambdal * Tper[1]

            # periodicity transformation derivatives
            dTperk = []
            ddTperk = []
            for i in range(np.shape(R)[0]):

                # first derivative base case
                dlamdki = lambdal * 1j * R[i, 0]
                dTperk.append(dlamdki * Tper[1])

                # second derivative base case
                ddTperk.append([])
                for j in range(np.shape(R)[0]):
                    #  d2lamdkidkj = -lambdal*R[i,0]*R[j,0]
                    ddlamdkidkj = dlamdki * 1j * R[j, 0]
                    ddTperk[i].append(ddlamdkidkj * Tper[1])

        return Tperk, dTperk, ddTperk

    Tperk, dTperk, ddTperk = diffT(Tper, R)

    # form directional derivatives
    dTperkv = csr_matrix(np.shape(Tperk))
    ddTperkv = csr_matrix(np.shape(Tperk))
    for i in range(np.shape(R)[0]):
        dTperkv = dTperkv + dTperk[i] * v[i]

        for j in range(np.shape(R)[0]):
            ddTperkv = ddTperkv + ddTperk[i][j] * v[i] * v[j]

    # compute directional mass and stiffness matrix derivatives
    Mhatp = Tperk.conj().T @ M @ dTperkv + dTperkv.conj().T @ M @ Tperk
    Khatp = Tperk.conj().T @ K @ dTperkv + dTperkv.conj().T @ K @ Tperk

    # compute directional mass and stiffness matrix derivatives
    Mhatpp = (ddTperkv.conj().T @ M @ Tperk + 2 * dTperkv.conj().T @ M @ dTperkv +
              Tperk.conj().T @ M @ ddTperkv)
    Khatpp = (ddTperkv.conj().T @ K @ Tperk + 2 * dTperkv.conj().T @ K @ dTperkv +
              Tperk.conj().T @ K @ ddTperkv)

    return Khatp, Mhatp, Khatpp, Mhatpp


def dispersion_w_k_RBME(K, M, X, R, kappas, kappasExp, mExp, nbands=10,
                        expandmodeshapes=True):
    # compute Bloch periodicity matrices
    Tper = blochPeriodicity(X, R)

    # number of k points
    nsegs = len(kappas)
    nkap = 0
    for i in range(nsegs):
        nkap = nkap + len(kappas[i])

    # number of degrees of freedom
    temp = Tper
    while isinstance(temp, list):
        temp = temp[0]
    nDOF, nDOFper = np.shape(temp)

    # compute band structure frequencies
    w = np.zeros((nbands, nkap))
    if expandmodeshapes:
        Phi = np.zeros((nDOF, nbands, nkap), complex)
    else:
        Phi = np.zeros((nDOFper, nbands, nkap), complex)

    count = 0
    for i in range(nsegs):

        #  for j in range(length(kappas[i])):
        wRBME, VRBME = dispersion_w_k(K, M, X, R, kappasExp[i], nbands=mExp, expandmodeshapes=False)

        # collect eigenvectors from all expansion points into
        Psi = np.reshape(VRBME, (nDOFper, -1))

        # orthogonalize eigenvectors
        Psi, Trash = np.linalg.qr(Psi)

        nDOFRBME = np.shape(Psi)[1]
        #  print(nDOF, "'free' Degrees of freedom")
        print(nDOFRBME, "'RBME' Degrees of freedom")

        #  for i in range(len(kappas))
        #  nkap = len(kappas[i])
        for j in range(len(kappas[i])):
            startTime = time.time()

            # apply Bloch boundary conditions to mass and stiffness matrices
            Khat, Mhat = KMBBC(K, M, Tper, kappas[i][j], R)

            # apply reduction basis to periodic mass and stiffness matrices
            KRBME = Psi.conj().T @ Khat @ Psi
            MRBME = Psi.conj().T @ Mhat @ Psi

            # compute eigenvalues
            L, P = scipy.linalg.eigh(KRBME, b=MRBME)

            # sort eignevalues and mode shapes
            isort = np.argsort(L)
            L = L[isort[0:nbands]]
            P = P[:, isort[0:nbands]]

            # store dispersion frequencies
            w[:, count] = np.sqrt(np.absolute(L))

            # apply periodicity transformation and store mode shape
            TperK = TperKappa(Tper, kappas[i][j], R)

            # apply periodicity transformation and store mode shape
            P = Psi @ P
            if expandmodeshapes:
                Phi[:, :, count] = TperK @ P
            else:
                Phi[:, :, count] = P

            # print diagnostic information
            print(
                "     RBME Loop",
                count +
                1,
                "of",
                nkap,
                ", calculation time",
                time.time() -
                startTime,
                "seconds")

            # increment counter
            count += 1
    # return w, Phi, dLam, dPhiper, dLam_fd, dPhi_fdper, P1per, P2per
    return w, Phi


def dispersion_w_k_RBMEX(K, M, X, R, kappas, kappasExp, mExp, nbands=10,
                         expandmodeshapes=True):
    # compute Bloch periodicity matrices
    Tper = blochPeriodicity(X, R)

    # number of k points
    nsegs = len(kappas)
    nkap = 0
    for i in range(nsegs):
        nkap = nkap + len(kappas[i])

    # number of degrees of freedom
    temp = Tper
    while isinstance(temp, list):
        temp = temp[0]
    nDOF, nDOFper = np.shape(temp)

    # compute band structure frequencies
    w = np.zeros((nbands, nkap))
    if expandmodeshapes:
        Phi = np.zeros((nDOF, nbands, nkap), complex)
    else:
        Phi = np.zeros((nDOFper, nbands, nkap), complex)

    count = 0
    for i in range(nsegs):

        #  for j in range(length(kappas[i])):
        Psi = np.array([]).reshape(nDOFper, 0)
        for j in range(len(kappasExp[i])):

            # apply Bloch boundary conditions to mass and stiffness matrices
            Khat, Mhat = KMBBC(K, M, Tper, kappasExp[i][j], R)

            # compute system matrix derivatives with respect to wavevector step, z
            z = kappasExp[i][-1] - kappasExp[i][0]
            z = z / np.linalg.norm(z)
            Khatp, Mhatp, Khatpp, Mhatpp = diffKMBBC(K, M, Tper, kappasExp[i][j], R, z)

            # compute eigenvectors
            L, P = scipy.sparse.linalg.eigsh(Khat, M=Mhat, k=mExp, sigma=0)

            # sort eigenvalues and mode shapes
            isort = np.argsort(L)
            L = L[isort]
            P = P[:, isort]

            # compute eigenvector derivative
            # tolerance for eigenvalues to be considered degenerate
            tolDeg = 1e-4 * L[-1]
            dL = np.zeros(mExp, dtype=complex)
            dP = np.zeros((nDOFper, mExp), dtype=complex)
            d2P = np.zeros((nDOFper, mExp), dtype=complex)
            k = 0
            while k < mExp:

                # check for eigenvalue degeneracy
                iDeg = np.where(np.abs(L - L[k]) < tolDeg)[0]
                nDeg = np.size(iDeg)

                PDeg = P[:, iDeg]

                if nDeg == 1:
                    # compute distinct eigenvalue sensitivity
                    dL[iDeg] = PDeg.conj().T @ (Khatp - L[k] * Mhatp) @ PDeg
                    # p = PDeg
                else:
                    #  A = np.dot(phi.conj().T,csr_matrix.dot((K_prime-L*M_prime),phi))
                    A = PDeg.conj().T @ (Khatp - L[k] * Mhatp) @ PDeg

                    # compute eigenvalues of A
                    L2, H = scipy.linalg.eigh(A)
                    dL[iDeg] = L2

                    # normalize H to unit magnitude
                    for j in range(nDeg):
                        H[:, j] = H[:, j] / np.sqrt(H[:, j].conj().T @ H[:, j])

                    PDeg = PDeg @ H

                # compute eigenvector derivatives using Nelson's method with
                # Friswell's extension for repeated eigenvalues
                # (Friswell MI, ASME Transactions 1996, The derivatives of repeated
                # eigenvalues and their associated eigenvectors)
                D = Khat - L[k] * Mhat

                # compute RHS term from Friswell Eq. (10)
                # this could be done with matrix multiplication but might be a little
                # harder to understand
                f = np.zeros((nDOFper, nDeg), dtype=complex)
                for m in range(nDeg):
                    f[:, m] = -(Khatp - L[iDeg[m]] * Mhatp - dL[iDeg[m]] * Mhat) @ PDeg[:, m]

                imax = np.argsort(np.sum(np.abs(PDeg) ** 2, 1))[0:nDeg]

                D[imax, :] = 0
                D[:, imax] = 0
                D[imax, imax] = 1
                f[imax, :] = 0

                v = scipy.sparse.linalg.spsolve(D, f)

                # ensure that single vector v still has 2 array dimensions
                if v.ndim < 2:
                    v = v[:, None]

                dP[:, iDeg] = v

                # # Compute Second Derivative of Eigenvector
                # D = Khat - L[k] * Mhat
                #
                # f = np.zeros((nDOFper, nDeg), dtype=complex)
                # for m in range(nDeg):
                #     f[:, m] = -2*(Khatp - L[iDeg[m]] * Mhatp - dL[iDeg[m]] * Mhat) @ v[:,m]
                #
                # imax = np.argsort(np.sum(np.abs(f) ** 2, 1))[0:nDeg]
                #
                # D[imax, :] = 0
                # D[:, imax] = 0
                # D[imax, imax] = 1
                # f[imax, :] = 0
                #
                # v = scipy.sparse.linalg.spsolve(D, f)
                #
                # # ensure that single vector v still has 2 array dimensions
                # if v.ndim < 2:
                #     v = v[:, None]
                #
                # d2P[:,iDeg] = v

                k = k + nDeg

            Psi = np.hstack((Psi, P, dP))

        # wRBME, VRBME = dispersion_w_k(K, M, X, R, kappasExp[i], nbands=mExp, expandmodeshapes=False)

        # collect eigenvectors from all expansion points into
        # Psi = np.reshape(VRBME, (nDOFper, -1))

        # orthogonalize eigenvectors
        Psi, Trash = np.linalg.qr(Psi)

        nDOFRBME = np.shape(Psi)[1]
        #  print(nDOF, "'free' Degrees of freedom")
        print(nDOFRBME, "'RBME' Degrees of freedom")

        #  for i in range(len(kappas))
        #  nkap = len(kappas[i])
        for j in range(len(kappas[i])):
            startTime = time.time()

            # apply Bloch boundary conditions to mass and stiffness matrices
            Khat, Mhat = KMBBC(K, M, Tper, kappas[i][j], R)

            # apply reduction basis to periodic mass and stiffness matrices
            KRBME = Psi.conj().T @ Khat @ Psi
            MRBME = Psi.conj().T @ Mhat @ Psi

            # compute eigenvalues
            L, P = scipy.linalg.eigh(KRBME, b=MRBME)

            # sort eignevalues and mode shapes
            isort = np.argsort(L)
            L = L[isort[0:nbands]]
            P = P[:, isort[0:nbands]]

            # store dispersion frequencies
            w[:, count] = np.sqrt(np.absolute(L))

            # apply periodicity transformation and store mode shape
            TperK = TperKappa(Tper, kappas[i][j], R)

            # apply periodicity transformation and store mode shape
            P = Psi @ P
            if expandmodeshapes:
                Phi[:, :, count] = TperK @ P
            else:
                Phi[:, :, count] = P

            # print diagnostic information
            print(
                "     RBME Loop",
                count +
                1,
                "of",
                nkap,
                ", calculation time",
                time.time() -
                startTime,
                "seconds")

            # increment counter
            count += 1
    # return w, Phi, dLam, dPhiper, dLam_fd, dPhi_fdper, P1per, P2per
    return w, Phi


# def eigDerivativesNelson(Phi, Lam, K, Kp, Kpp, M, Mp, Mpp):
#     """compute the eigenvalue and eigenvector derivatives"""
#     nEig = np.size(Lam)

#     nDOF = np.shape(K)[0]

#     # tolerance for eigenvalues to be considered degenerate
#     tolDeg = 1e-4 * Lam[-1]

#     # compute eigenvalue derivatives
#     dLam = np.zeros(nEig, dtype=complex)
#     dPhi = np.zeros((nDOF, nEig), dtype=complex)
#     i = 0

#     while i < nEig:

#         # check for eigenvalue degeneracy
#         iDeg = np.where(np.abs(Lam - Lam[i]) < tolDeg)[0]
#         nDeg = np.size(iDeg)

#         phi = Phi[:, iDeg]


#         print(iDeg)


#         if nDeg == 1:
#             # compute distinct eigenvalue sensitivity
#             dLam[iDeg] = phi.conj().T @ (Kp - Lam[i] * Mp) @ phi
#             psi = phi
#         else:
#             #  A = np.dot(phi.conj().T,csr_matrix.dot((K_prime-L*M_prime),phi))
#             A = phi.conj().T @ (Kp - Lam[i] * Mp) @ phi

#             # compute eigenvalues of A
#             L, H = scipy.linalg.eigh(A)
#             dLam[iDeg] = L


#             # normalize H to unit magnitude
#             for j in range(nDeg):
#                 H[:, j] = H[:, j] / np.sqrt(H[:, j].conj().T @ H[:, j])

#             psi = phi @ H

#         # compute eigenvector derivatives using Nelson's method with
#         # Friswell's extension for repeated eigenvalues
#         # (Friswell MI, ASME Transactions 1996, The derivatives of repeated
#         # eigenvalues and their associated eigenvectors)
#         D = K - Lam[i] * M

#         # compute RHS term from Friswell Eq. (10)
#         # this could be done with matrix multiplication but might be a little
#         # harder to understand
#         f = np.zeros((nDOF, nDeg), dtype=complex)
#         for j in range(nDeg):
#             f[:, j] = -(Kp - Lam[iDeg[j]] * Mp - dLam[iDeg[j]] * M) @ psi[:, j]

#         #  f = -(K_prime-dLam*M-Lam[i]*M_prime)*phi
#         #  imax = the index where the maximum eignevector components occur
#         imax = np.argsort(np.sum(psi ** 2, 1))[0:nDeg]
#         #  print(imax)
#         #  print()
#         #  imax = np.arange(0,nDeg)
#         #  print(imax)

#         #  D = np.random.rand(5,5)
#         #  f = np.random.rand(5,2)
#         #  imax = np.arange(2,4)

#         D[imax, :] = 0
#         D[:, imax] = 0
#         D[imax, imax] = 1
#         f[imax, :] = 0

#         v = scipy.sparse.linalg.spsolve(D, f)
#         #  v = np.linalg.solve(D.todense(),f)

#         # ensure that single vector v still has 2 array dimensions
#         if v.ndim < 2:
#             v = v[:, None]

#         c = np.zeros((nDeg, nDeg), dtype=complex)
#         for j in range(nDeg):
#             for k in range(nDeg):
#                 if j == k:
#                     #  c[j,k] = (-1/2)*np.dot(phi.conj().T,np.dot(M_prime,phi)) -
#                     #  np.dot(v[:,k].conj().T

#                     # THIS IS WRONG, BUT THE NEXT EXPRESSION SEEMS WRONG TOO
#                     c[j, j] = ((-1 / 2) * (psi[:, j].conj().T @ Mp @
#                                psi[:, j]) - v[:, j].conj().T @ M @ psi[:, j])

#                     # THIS SEEMS TO WORK, AT LEAST FOR NONDEGENERATE CASES
#                     c[j, j] = ((-1 / 2) * (psi[:, j].conj().T @ Mp @
#                                            psi[:, j]) - psi[:, j].conj().T @ M @ v[:, j])

#                     # # TRY WHAT I THINK IS CORRECT
#                     # c[j, j] = -(1/2)*(psi[:, j].conj().T @ Mp @ psi[:, j]
#                     #                   + psi[:, j].conj().T @ M @ v[:, j]
#                     #                   + v[:, j].conj().T @ M @ psi[:, j])
#                     # c[j, j] = 0

#                 else:
#                     c1 = (-2 * psi[:, j].conj().T @ (Kp - Lam[i] * Mp -
#                                                      dLam[iDeg[k]] * M) @ v[:, k])
#                     c2 = (-psi[:, j].conj().T @ (Kpp - Lam[i] * Mpp - 2 * dLam[iDeg[k]] * Mp)
#                           @ psi[:, k])
#                     c3 = (2 * (dLam[iDeg[j]] - dLam[iDeg[k]]))

#                     c[j, k] = (c1 + c2) / c3

#         dPhi[:, iDeg] = v + psi @ c

#         i += nDeg


#     """
#     # compute eigenvector derivatives using Fox and Kapoor's method (Mode
#     # Superposition)
#     c = np.zeros(nEig,nEig)
#     for i in range(nEig):
#         for j in range(nEig):
#             c[i,j]
#     """
#     return dLam, dPhi

# def eigDerivativesAlgebraic(Phi, Lam, K, Kp, Kpp, M, Mp, Mpp):
#     """compute the eigenvalue and eigenvector derivatives"""
#     nEig = np.size(Lam)

#     nDOF = np.shape(K)[0]

#     # tolerance for eigenvalues to be considered degenerate
#     tolDeg = 1e-4 * Lam[-1]

#     # compute eigenvalue derivatives
#     dLam = np.zeros(nEig, dtype=complex)
#     dPhi = np.zeros((nDOF, nEig), dtype=complex)
#     d2Lam = np.zeros(nEig, dtype=complex)
#     d2Phi = np.zeros((nDOF, nEig), dtype=complex)
#     i = 0

#     while i < nEig:

#         # check for eigenvalue degeneracy
#         iDeg = np.where(np.abs(Lam - Lam[i]) < tolDeg)[0]
#         nDeg = np.size(iDeg)

#         print(iDeg)

#         # degenerate mode set
#         PhiDeg = Phi[:, iDeg]

#         # # Algebraic Modification of Degenerate Matrix
#         # D = scipy.sparse.hstack([
#         #         scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg.conj().T @ M]),
#         #         scipy.sparse.vstack([-M @ PhiDeg,np.zeros((nDeg,nDeg))])])
#         # # Potential issues in this matrix:
#         # #   1) the term Phi'*M*dPhi + dPhi'*M*Phi is not necessarily equal to 2*Phi'*M*dPhi
#         #
#         # f = np.concatenate((-(Kp-Lam[i]*Mp) @ PhiDeg, 0.5*PhiDeg.conj().T @ Mp @ PhiDeg),0)

#         # Algebraic Modification of Degenerate Matrix
#         D = scipy.sparse.hstack([
#                 scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg.T @ M]),
#                 scipy.sparse.vstack([-M @ PhiDeg,np.zeros((nDeg,nDeg))])])

#         f = np.concatenate((-(Kp-Lam[i]*Mp) @ PhiDeg, 0.5*PhiDeg.T @ Mp @ PhiDeg),0)

#         # solve modified system
#         v = scipy.sparse.linalg.spsolve(D, f)

#         # ensure that single vector v still has 2 array dimensions
#         if v.ndim < 2:
#             v = v[:, None]

#         dPhiDeg = v[0:nDOF,:]
#         dLamDeg = np.diag(v[nDOF:,:])
#         print('dLamMat = ',v[nDOF:,:])

#         dPhi[:,iDeg] = dPhiDeg
#         dLam[iDeg] = dLamDeg

#         # tolerance for eigenvalue derivatives to be considered degenerate
#         tolDeg2 = 1e-4 * (np.abs(dLamDeg)).max()
#         j = 0
#         while j < nDeg:

#             # check for eigenvalue degeneracy
#             iDeg2 = np.where(np.abs(dLamDeg - dLamDeg[j]) < tolDeg2)[0]
#             nDeg2 = np.size(iDeg2)

#             print('\t',iDeg2)

#             # Algebraic Modification of Degenerate Matrix
#             D = scipy.sparse.hstack([
#                 scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg[:,iDeg2].conj().T @ M]),
#                 scipy.sparse.vstack([-M @ PhiDeg[:,iDeg2],np.zeros((nDeg2,nDeg2))])])

#             f2 = np.concatenate((-2*(Kp-Lam[i]*Mp-dLamDeg[j]*M) @ dPhiDeg[:,iDeg2] -
#                                 (Kpp-Lam[i]*Mpp - 2*dLamDeg[j]*Mp) @ PhiDeg[:,iDeg2],
#                                 0.5*PhiDeg[:,iDeg2].conj().T @ Mp @ PhiDeg[:,iDeg2]),0)

#             #  # Algebraic Modification of Degenerate Matrix
#             # D = scipy.sparse.hstack([
#             #     scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg[:,iDeg2].T @ M]),
#             #     scipy.sparse.vstack([-M @ PhiDeg[:,iDeg2],np.zeros((nDeg2,nDeg2))])])
#             #
#             # f2 = np.concatenate((-2*(Kp-Lam[i]*Mp-dLamDeg[j]*M) @ dPhiDeg[:,iDeg2] -
#             #                     (Kpp-Lam[i]*Mpp - 2*dLamDeg[j]*Mp) @ PhiDeg[:,iDeg2],
#             #                     0.5*PhiDeg[:,iDeg2].T @ Mp @ PhiDeg[:,iDeg2]),0)

#             # b1 = (Kpp-Lam[i]*Mpp) @

#             # solve modified system
#             v2 = scipy.sparse.linalg.spsolve(D, f2)

#             # ensure that single vector v still has 2 array dimensions
#             if v2.ndim < 2:
#                 v2 = v2[:, None]

#             d2PhiDeg = v2[0:nDOF,:]
#             d2LamDeg = np.diag(v2[nDOF:,:])

#             d2Phi[:,iDeg[iDeg2]] = d2PhiDeg
#             d2Lam[iDeg[iDeg2]] = d2LamDeg

#             j += nDeg2


#         i += nDeg

#     return dLam, dPhi, d2Lam, d2Phi


def modePhaseAlign(V1, L1, M1, V2, L2, M2):
    """scale each eigenvector in the matrix V2 by a complex scalar so that it
    aligns as closely as possible with the corresponding eigenvector in V1.
    When degeneracies exist in the eigenvalues (contained in D1, D2). Then the
    degenerate eigenvectors are linearly combined to match as closely as
    possible with the corresponding vectors in the opposite set."""

    # tolerance for degeneracy (arbitrary small value)
    tolDeg = L1[-1] * 1e-6

    # use pseudoinverse to find linear combination of eigenvectors in V2 that
    # minimizes least squares error with V1
    C2 = np.linalg.pinv(V2) @ V1

    # zero out off-diagonal terms in C (unless degeneracy allows)
    for i in range(np.shape(V2)[1]):
        for j in range(np.shape(V1)[1]):
            if np.abs(L2[i] - L2[j]) > tolDeg:
                C2[i, j] = 0

    V2 = V2 @ C2

    # repeat the previous steps but interchange V1 and V2. This is done in case
    # V1 has a degeneracy that disappears in V2. In this case, the degenerate
    # vectors will be aligned to match the vectors in V2 as closely as
    # possible.
    C1 = np.linalg.pinv(V1) @ V2
    for i in range(np.shape(V1)[1]):
        for j in range(np.shape(V2)[1]):
            if np.abs(L1[i] - L1[j]) > tolDeg:
                C1[i, j] = 0

    V1 = V1 @ C1

    # ensure that both sets of eigenvectors are normalized to the mass matrix
    for i in range(np.shape(V2)[1]):
        V2[:, i] = V2[:, i] / np.sqrt(V2[:, i].conj().T @ M2 @ V2[:, i])

    for i in range(np.shape(V1)[1]):
        V1[:, i] = V1[:, i] / np.sqrt(V1[:, i].conj().T @ M1 @ V1[:, i])

    return V1, V2
    # return V2
