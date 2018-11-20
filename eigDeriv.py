import numpy as np
# import scipy.sparse as scisp
# import scipy.sparse.linalg as scispla
import scipy


def eigDerivativesNelson(Phi, Lam, K, Kp, Kpp, M, Mp, Mpp):
    """compute the eigenvalue and eigenvector derivatives"""
    nEig = np.size(Lam)

    nDOF = np.shape(K)[0]

    # tolerance for eigenvalues to be considered degenerate
    tolDeg = 1e-4 * Lam[-1]

    # compute eigenvalue derivatives
    dLam = np.zeros(nEig, dtype=complex)
    dPhi = np.zeros((nDOF, nEig), dtype=complex)
    i = 0

    while i < nEig:

        # check for eigenvalue degeneracy
        iDeg = np.where(np.abs(Lam - Lam[i]) < tolDeg)[0]
        nDeg = np.size(iDeg)

        phi = Phi[:, iDeg]

        print(iDeg)

        if nDeg == 1:
            # compute distinct eigenvalue sensitivity
            dLam[iDeg] = phi.conj().T @ (Kp - Lam[i] * Mp) @ phi
            psi = phi
        else:
            #  A = np.dot(phi.conj().T,csr_matrix.dot((K_prime-L*M_prime),phi))
            A = phi.conj().T @ (Kp - Lam[i] * Mp) @ phi

            # compute eigenvalues of A
            L, H = scipy.linalg.eigh(A)
            dLam[iDeg] = L

            # normalize H to unit magnitude
            for j in range(nDeg):
                H[:, j] = H[:, j] / np.sqrt(H[:, j].conj().T @ H[:, j])

            psi = phi @ H

        # compute eigenvector derivatives using Nelson's method with
        # Friswell's extension for repeated eigenvalues
        # (Friswell MI, ASME Transactions 1996, The derivatives of repeated
        # eigenvalues and their associated eigenvectors)
        D = K - Lam[i] * M

        # compute RHS term from Friswell Eq. (10)
        # this could be done with matrix multiplication but might be a little
        # harder to understand
        f = np.zeros((nDOF, nDeg), dtype=complex)
        for j in range(nDeg):
            f[:, j] = -(Kp - Lam[iDeg[j]] * Mp - dLam[iDeg[j]] * M) @ psi[:, j]

        #  f = -(K_prime-dLam*M-Lam[i]*M_prime)*phi
        #  imax = the index where the maximum eignevector components occur
        imax = np.argsort(np.sum(psi ** 2, 1))[0:nDeg]
        #  print(imax)
        #  print()
        #  imax = np.arange(0,nDeg)
        #  print(imax)

        #  D = np.random.rand(5,5)
        #  f = np.random.rand(5,2)
        #  imax = np.arange(2,4)

        D[imax, :] = 0
        D[:, imax] = 0
        D[imax, imax] = 1
        f[imax, :] = 0

        v = scipy.sparse.linalg.spsolve(D, f)
        #  v = np.linalg.solve(D.todense(),f)

        # ensure that single vector v still has 2 array dimensions
        if v.ndim < 2:
            v = v[:, None]

        c = np.zeros((nDeg, nDeg), dtype=complex)
        for j in range(nDeg):
            for k in range(nDeg):
                if j == k:
                    #  c[j,k] = (-1/2)*np.dot(phi.conj().T,np.dot(M_prime,phi)) -
                    #  np.dot(v[:,k].conj().T

                    # THIS IS WRONG, BUT THE NEXT EXPRESSION SEEMS WRONG TOO
                    c[j, j] = ((-1 / 2) * (psi[:, j].conj().T @ Mp @
                                           psi[:, j]) - v[:, j].conj().T @ M @ psi[:, j])

                    # THIS SEEMS TO WORK, AT LEAST FOR NONDEGENERATE CASES
                    c[j, j] = ((-1 / 2) * (psi[:, j].conj().T @ Mp @
                                           psi[:, j]) - psi[:, j].conj().T @ M @ v[:, j])

                    # # TRY WHAT I THINK IS CORRECT
                    # c[j, j] = -(1/2)*(psi[:, j].conj().T @ Mp @ psi[:, j]
                    #                   + psi[:, j].conj().T @ M @ v[:, j]
                    #                   + v[:, j].conj().T @ M @ psi[:, j])
                    # c[j, j] = 0

                else:
                    c1 = (-2 * psi[:, j].conj().T @ (Kp - Lam[i] * Mp -
                                                     dLam[iDeg[k]] * M) @ v[:, k])
                    c2 = (-psi[:, j].conj().T @ (Kpp - Lam[i] * Mpp - 2 * dLam[iDeg[k]] * Mp)
                          @ psi[:, k])
                    c3 = (2 * (dLam[iDeg[j]] - dLam[iDeg[k]]))

                    c[j, k] = (c1 + c2) / c3

        dPhi[:, iDeg] = v + psi @ c

        i += nDeg

    """
    # compute eigenvector derivatives using Fox and Kapoor's method (Mode
    # Superposition)
    c = np.zeros(nEig,nEig)
    for i in range(nEig):
        for j in range(nEig):
            c[i,j]
    """
    return dLam, dPhi


def eigDerivativesAlgebraic(Phi, Lam, K, Kp, Kpp, M, Mp, Mpp):
    """compute the eigenvalue and eigenvector derivatives"""
    nEig = np.size(Lam)

    nDOF = np.shape(K)[0]

    # tolerance for eigenvalues to be considered degenerate
    tolDeg = 1e-4 * Lam[-1]

    # compute eigenvalue derivatives
    dLam = np.zeros(nEig, dtype=complex)
    dPhi = np.zeros((nDOF, nEig), dtype=complex)
    d2Lam = np.zeros(nEig, dtype=complex)
    d2Phi = np.zeros((nDOF, nEig), dtype=complex)
    i = 0

    while i < nEig:

        # check for eigenvalue degeneracy
        iDeg = np.where(np.abs(Lam - Lam[i]) < tolDeg)[0]
        nDeg = np.size(iDeg)

        print(iDeg)

        # degenerate mode set
        PhiDeg = Phi[:, iDeg]

        # # Algebraic Modification of Degenerate Matrix
        # D = scipy.sparse.hstack([
        #         scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg.conj().T @ M]),
        #         scipy.sparse.vstack([-M @ PhiDeg,np.zeros((nDeg,nDeg))])])
        # # Potential issues in this matrix:
        # #   1) the term Phi'*M*dPhi + dPhi'*M*Phi is not necessarily equal to 2*Phi'*M*dPhi
        #
        # f = np.concatenate((-(Kp-Lam[i]*Mp) @ PhiDeg, 0.5*PhiDeg.conj().T @ Mp @ PhiDeg),0)

        # Algebraic Modification of Degenerate Matrix
        D = scipy.sparse.hstack([
            scipy.sparse.vstack([K - Lam[i] * M, -PhiDeg.T @ M]),
            scipy.sparse.vstack([-M @ PhiDeg, np.zeros((nDeg, nDeg))])])

        f = np.concatenate((-(Kp - Lam[i] * Mp) @ PhiDeg, 0.5 * PhiDeg.T @ Mp @ PhiDeg), 0)

        # solve modified system
        v = scipy.sparse.linalg.spsolve(D, f)

        # ensure that single vector v still has 2 array dimensions
        if v.ndim < 2:
            v = v[:, None]

        dPhiDeg = v[0:nDOF, :]
        dLamDeg = np.diag(v[nDOF:, :])
        print('dLamMat = ', v[nDOF:, :])

        dPhi[:, iDeg] = dPhiDeg
        dLam[iDeg] = dLamDeg

        # tolerance for eigenvalue derivatives to be considered degenerate
        tolDeg2 = 1e-4 * (np.abs(dLamDeg)).max()
        j = 0
        while j < nDeg:

            # check for eigenvalue degeneracy
            iDeg2 = np.where(np.abs(dLamDeg - dLamDeg[j]) < tolDeg2)[0]
            nDeg2 = np.size(iDeg2)

            print('\t', iDeg2)

            # Algebraic Modification of Degenerate Matrix
            D = scipy.sparse.hstack([
                scipy.sparse.vstack([K - Lam[i] * M, -PhiDeg[:, iDeg2].conj().T @ M]),
                scipy.sparse.vstack([-M @ PhiDeg[:, iDeg2], np.zeros((nDeg2, nDeg2))])])

            f2 = np.concatenate((-2 * (Kp - Lam[i] * Mp - dLamDeg[j] * M) @ dPhiDeg[:, iDeg2] -
                                 (Kpp - Lam[i] * Mpp - 2 * dLamDeg[j] * Mp) @ PhiDeg[:, iDeg2],
                                 0.5 * PhiDeg[:, iDeg2].conj().T @ Mp @ PhiDeg[:, iDeg2]), 0)

            #  # Algebraic Modification of Degenerate Matrix
            # D = scipy.sparse.hstack([
            #     scipy.sparse.vstack([K-Lam[i]*M, -PhiDeg[:,iDeg2].T @ M]),
            #     scipy.sparse.vstack([-M @ PhiDeg[:,iDeg2],np.zeros((nDeg2,nDeg2))])])
            #
            # f2 = np.concatenate((-2*(Kp-Lam[i]*Mp-dLamDeg[j]*M) @ dPhiDeg[:,iDeg2] -
            #                     (Kpp-Lam[i]*Mpp - 2*dLamDeg[j]*Mp) @ PhiDeg[:,iDeg2],
            #                     0.5*PhiDeg[:,iDeg2].T @ Mp @ PhiDeg[:,iDeg2]),0)

            # b1 = (Kpp-Lam[i]*Mpp) @

            # solve modified system
            v2 = scipy.sparse.linalg.spsolve(D, f2)

            # ensure that single vector v still has 2 array dimensions
            if v2.ndim < 2:
                v2 = v2[:, None]

            d2PhiDeg = v2[0:nDOF, :]
            d2LamDeg = np.diag(v2[nDOF:, :])

            d2Phi[:, iDeg[iDeg2]] = d2PhiDeg
            d2Lam[iDeg[iDeg2]] = d2LamDeg

            j += nDeg2

        i += nDeg

    return dLam, dPhi, d2Lam, d2Phi
