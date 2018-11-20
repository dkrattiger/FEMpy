import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import time
import math
import MMA
import mesh2D


def topo2D(nelx, nely, Lx, volFrac, freeDOFs, fixedDOFs, Mlump, ilump,
        rmin = 2, penal = 4, maxiter = 100, costFunc = 'eigval', power = 4, xConstraint = None,optimizer = 'OC'):

# def topo2D(nelx, nely, Lx, volFrac, freeDOFs, fixedDOFs, Mlump, ilump, rmin = 2, penals = 4, iteration = 0, maxiter = 100, x0 = [-1], costFunc = 'eigval', power = 1, xConstraint = None):
    """ Adapted Topology Optimization Code from Ole Sigmund 99-Line Topology
    Optimization Matlab Code
    updated to include some of the changes from the 88-line Topology
    Optimization Matlab Code"""

    # initialize timers
    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    timeslots = [0,0,0,0]
    temptime = [0,0,0,0]


    # SIMP Stiffness interpolation limits
    E0 = 1e-3
    E1 = 1

    # SIMP density interpolation limits
    rho0 = 1e-3
    rho1 = 1

    # Initialize arrays and counters
    nele = nelx*nely
    # if np.shape(x0) == (nele,1):
    #     x = x0
    # else:
    x = np.ones((nele,1))*volFrac

    dc = np.zeros(np.shape(x),dtype = float)
    loop = 0
    change = 1
    startTotal = time.time()

    # define element connectivity matrix
    eDOF = eDOFMaker(nelx,nely)

    # random vector split up into element level vectors
    nDOF = (nelx+1)*(nely+1)*2
    # freeDOFs, fixedDOFs = FE_BCs(nelx, nely)

    np.random.seed(88)
    # np.random.seed(89)
    # np.random.seed(90)
    # y0 = 2*np.random.rand(np.shape(freeDOFs)[0])-1
    # y0 = np.ones(np.shape(freeDOFs))
    # y0 = np.random.rand(np.shape(freeDOFs)[0])
    # y0 = np.random.randn(np.shape(freeDOFs)[0])
    # y0f = np.zeros((nDOF))
    # y0f[freeDOFs] = y0
    # y0f = y0f/np.linalg.norm(y0f)

    # define stiffness matrix sparsity pattern
    irow,icol = FEPattern(eDOF)

    # assemble convolution matrix
    H,Hs = convMat(nelx,nely,rmin)

    # compute element mass and stiffness matrix
    l = Lx/nelx

    # define element material object
    mat1 = mesh2D.Material()
    # mat2 = mesh2D.Material()
    mat1.E = 1
    mat1.nu = 0.3
    mat1.rho = 1

    # define element mass and stiffness matrices
    elex = np.array([0,l,0,l])
    eley = np.array([0,0,l,l])
    ele1 = mesh2D.Quad(elex,eley)
    KE, ME = ele1.KM(mat1,type = 'plane_stress')
    # KE, ME = ele1.KM(mat1,type = 'plane_strain')

    Ly = Lx/3
    Mlump = 0.20*Lx*Ly*volFrac

    # % INITIALIZE MMA OPTIMIZER
    mMMA  = 1                       # The number of general constraints.
    nMMA  = nele                    # The number of design variables x_j.
    xmin  = np.zeros((nMMA,1))      # Column vector with the lower bounds for the variables x_j.
    xmax  = np.ones((nMMA,1))       # Column vector with the upper bounds for the variables x_j.
    xold1 = x                       # xval, one iteration ago (provided that iter>1).
    xold2 = x                       # xval, two iterations ago (provided that iter>2).
    low   = np.ones((nMMA,1))       # Column vector with the lower asymptotes from the previous iteration (provided that iter>1).
    upp   = np.ones((nMMA,1))       # Column vector with the upper asymptotes from the previous iteration (provided that iter>1).
    a0    = 1                       # The constants a_0 in the term a_0*z.
    a     = np.zeros((mMMA,1))      # Column vector with the constants a_i in the terms a_i*z.
    c_MMA = 1e3*np.ones((mMMA,1))   # Column vector with the constants c_i in the terms c_i*y_i.
    d     = np.zeros((mMMA,1))      # Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.


    print('Model size: ', 2*(nelx+1)*(nely+1),' DOFs')


    # initialize terms to be stored for each iteration
    xlist = []
    clist = []
    lamlist = []
    Philist = []

    # define convergence tolerance
    tolchange = 0.01
    tolchange = 1e-4

    # tolchange = 1e-10
    yNext = np.random.randn(np.shape(freeDOFs)[0])
    muNext = 0

    while change>tolchange:

        # get current loops penalty exponent
        # if not isinstance(penals, int):
        #     ipenal = min(loop,np.shape(penals)[0])
        #     penal = penals[ipenal]

        # increment loop counter
        loop += 1

        # store design variables from previous loop
        xOld = x

        # Evaluate Cost function
        startloop = time.time()

        # form mass and stiffness matrices
        K, M = FE_KM(nelx, nely, KE, ME, Mlump, ilump, x, penal, irow, icol, E0, E1, rho0, rho1, volFrac, freeDOFs, fixedDOFs)

        # convert to CSC format
        K = scipy.sparse.csc_matrix(K)
        M = scipy.sparse.csc_matrix(M)

        # print(np.sum(M, axis=None))
        # print(Mlump)

        if costFunc == 'eigval':

            # Eigenvalue sensitivity cost function
            tstart = time.time()
            # w, P = FE_Eigs(nelx, nely, KE, ME, x, penal, irow, icol, E0, E1, rho0, rho1, volFrac)

            neigs = 1
            lam, Phi = sla.eigsh(K, M=M, k=neigs, sigma=0)
            w = np.sqrt(lam)

            # mass normalize Phi
            for i in range(neigs):
                Phi[:,i] = Phi[:,i]/np.sqrt(Phi[:,i].T @ M @ Phi[:,i])

            # store full eigenvalue (including fixed DOFs)
            Phifull = np.zeros((nDOF,neigs))
            Phifull[freeDOFs,:] = Phi

            w1 = w[0]
            c = lam[0]
            cprint = np.array([c])
            Phi1 = Phifull[:,0]
            Phi1e = np.squeeze(Phi1[eDOF]).T
            dc = -((np.sum((KE @ Phi1e) * Phi1e, axis = 0)*(penal)*(x.T**(penal-1))*(E1-E0)
                 -(w1**2)*np.sum((ME @ Phi1e) * Phi1e, axis = 0)*(rho1-rho0)).T)/(2*w1)
            # dc = ((np.sum((KE @ P1e) * P1e, axis = 0)*(penal)*(x.T**(penal-1))*(E1-E0)
            #         -(w1**2)*np.sum((ME @ P1e) * P1e, axis = 0)*(rho1-rho0)).T)/(2*w1)
            time1 += time.time()-tstart

        elif costFunc == 'power':

            # start a timer
            tstart = time.time()

            # starting vector
            # y0 = np.random.randn(np.shape(freeDOFs)[0])
            y0 = yNext

            # add first vector to Krylov basis
            y = [y0/np.linalg.norm(y0)]

            mu = muNext
            factorizeK = 'LU'
            # factorizeK = 'LU2'
            # factorizeK = 'GMRES'
            # factorizeK = 'CG'
            # factorizeK = 'PCG'

            if factorizeK == 'LU':
                Ksolve = sla.factorized(K-mu*M)

            elif factorizeK == 'LU2':
                Ksolve = sla.splu(K-mu*M).solve

            elif factorizeK == 'CG':
                def Ksolve(b):
                    return sla.cg(K-mu*M, b)[0]

            elif factorizeK == 'PCG':

                # Precond = sla.spilu(K-mu*M,drop_tol=1e-4, fill_factor=1)
                # Precond = sla.spilu(K-mu*M,drop_tol=1e-12, fill_factor=10)
                Precond = sla.splu(K-mu*M)
                Precond = sla.LinearOperator(Precond.shape, matvec = Precond.solve)
                # Precond = lambda x: Precond.solve(x)
                # Precond = sla.LinearOperator(np.shape(K), Precond)
                def Ksolve(b):
                    # use sparse incomplete LU factorization to provide a
                    # preconditioner for the conjugate gradient iteration

                    x, info = sla.cg(K-mu*M, b, M=Precond, maxiter=100)
                    if info:
                        print(info)

                    return x

            elif factorizeK == 'GMRES':
                def Ksolve(b):
                    return sla.gmres(K-mu*M, b)[0]

            elif factorizeK == 'SPSOLVE':
                def Ksolve(b):
                    return sla.spsolve(K-mu*M, b)


            # build up krylov basis
            for i in range(power*2):
                if i == power:
                    y.append(Ksolve(y[-1]))
                else:
                    y.append(Ksolve(M @ y[-1]))


            # index full vector
            yf = [np.zeros((nDOF,)) for X in y]
            # print(np.shape(yf[0][freeDOFs]))
            for i in range(2*power+1):
                yf[i][freeDOFs] = y[i]

            # index full vector into element by element vectors
            ye = [np.squeeze(X[eDOF]).T for X in yf]

            # compute cost function sensitivity
            dc = np.zeros(np.shape(x),dtype = float)
            # dc = (1/2)*(np.sum((ME @ ye[power]) * ye[power], axis=0)[:,np.newaxis] * (rho1 - rho0))
            for i in range(power):

                # note, I think the mu-shift should be y[i+1], but y[i] seems to be working better
                dc = dc + (-np.sum((KE @ ye[-i-1]) * ye[i+1], axis = 0)[:,np.newaxis] * (penal) * (x**(penal-1))*(E1-E0)
                           +np.sum((ME @ ye[-i-1]) * ye[i+1], axis = 0)[:,np.newaxis] * (rho1-rho0)*mu
                           +np.sum((ME @ ye[-i-1]) * ye[i], axis = 0)[:,np.newaxis] * (rho1-rho0))


            dc = dc/np.linalg.norm(dc)

            # compute cost function
            c = y[power].T @ M @ y[power]
            c = y[power].T @ y[power]

            # Project mass and stiffness matrices onto Krylov space
            y = np.asarray(y).T
            y,trash = np.linalg.qr(y)
            Khat = y.T @ K @ y
            Khat = (1/2)*(Khat+Khat.T)
            Mhat = y.T @ M @ y
            Mhat = (1/2)*(Mhat+Mhat.T)

            # compute Ritz estimates of eigenvalues
            lam, Phi = scipy.linalg.eigh(Khat, b=Mhat)
            Phi = y @ Phi

            # approximate eigenvalue and use it for next shift (mu)
            muNext = lam[0]*0.5
            # muNext = lam[0]*0.9
            # muNext = lam[0]*0.0

            # use the first eigenvector estimate as the next starting vector
            # yNext = np.random.randn(np.shape(freeDOFs)[0])
            # yNext = y @ Phihat[:,0]
            yNext = Phi[:,0]

            # collect values to summarize the quality of the current design
            cprint = np.array([lam[0]])

            # finish timing the cost function stage of the optimization
            time1 += time.time()-tstart

        # initialize  volume sensitivity
        v = np.sum(x)/(nele)
        dv = np.ones((nele,1))/(nele)

        # filter sensitivities
        start3 = time.time()
        # dc, dv = check3(H, Hs, x, dc, dv)


        if xConstraint:
            dc = xConstraint(dc, nelx, nely)


        ft = 1
        if ft == 1:
            dc = ((H @ (dc * x)) / (Hs * x))
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)

        time3 += time.time()-start3

        start4 = time.time()
        # optimizer = 'OC'
        # optimizer = 'MMA'
        if optimizer == 'OC':

            # design update by the optimality criterion
            x = OC(nelx, nely, x, volFrac, dc, dv, ft, H, Hs)

        elif optimizer == 'MMA':

            # METHOD OF MOVING ASYMPTOTES
            xval = x
            f0val = c
            df0dx = dc

            # fval = np.array([[v - volFrac],[-v + volFrac]])
            # dfdx = np.concatenate((dv.T,-dv.T), axis = 0)

            fval = np.array([[v - volFrac]])
            dfdx = dv.T

            # fval = np.array([[-v + volFrac]])
            # dfdx = -dv.T

            [xmma, ymma, d2, d3, d4, d5, d6, d7, d8, low, upp] = MMA.mmasub(mMMA, nMMA, loop, xval, xmin, xmax, xold1, xold2,
                                                                 f0val, df0dx, fval, dfdx, low, upp, a0, a, c_MMA, d)

            # Update MMA Variables
            # xnew = reshape(xmma, nely, nelx, nelz)
            # xnew = xmma
            if ft == 1:
                x = xmma
            elif ft == 2:
                x = (H @ xmma)/ Hs

            xold2 = xold1
            xold1 = x


        # apply constraints to x
        if xConstraint:
            x = xConstraint(x, nelx, nely)

        time4 += time.time()-start4

        tloop = time.time() - startloop

        # print results
        change = np.max(np.abs(x-xOld))

        vol = np.sum(x)/(nelx*nely)
        printstring = 'It.:%4d,    Obj.:' + ' %8.3e'*np.alen(cprint) + ',   Vol.:%5.4f, Ch.:%5.4f' + ',   Tloop:%5.4f'

        valueTuple = (loop,) + tuple(cprint) +(vol,change) + (tloop,)
        print(printstring % valueTuple)

        # store key info from iteration
        xlist.append(x)
        clist.append(c)
        lamlist.append(lam)
        Philist.append(Phi)


        # exit loop if iteration count reaches "maxiter"
        if loop>=maxiter:
            break


    # # Print Frequency/Eigenvalue Results from 2nd to Last Iteration
    # K, M = FE_KM(nelx, nely, KE, ME, Mlump, xOld, penal, irow, icol,  E0, E1, rho0, rho1, volFrac)
    # neigs = 3
    # L, P = sla.eigsh(K, M=M, k=neigs, sigma=0)
    # print('previous iteration lowest frequencies: ',np.sqrt(L))
    # print('previous iteration lowest eigvals: ',L)

    # Print Frequency/Eignenvalue Results form Last Iteration
    K, M = FE_KM(nelx, nely, KE, ME, Mlump, ilump, x, penal, irow, icol,  E0, E1, rho0, rho1, volFrac, freeDOFs, fixedDOFs)
    neigs = 3
    L, P = sla.eigsh(K, M=M, k=neigs, sigma=0)
    # P2 = np.zeros((nDOF,np.shape(P)[-1]))
    # P2[freeDOFs,:] = P


    # expand mode shapes to full size
    allDOFs = np.hstack((freeDOFs,fixedDOFs))
    isort = np.argsort(allDOFs)
    nphi = np.shape(Philist[0])[1]
    nfix = np.shape(fixedDOFs)[0]
    Philist = [np.vstack((Phi, np.zeros((nfix, nphi))))[isort,:] for Phi in Philist]

    # Philist = [np.zeros]



    print('lowest frequencies: ',np.sqrt(L))
    print('lowest eigvals: ',L)


    # Print Computation Time Summaries for Various Stages in the Optimization
    timeTotal = time.time() - startTotal
    print('cost function time (s)', time1)
    # print('time 1', time1)
    # print('time 2', time2)
    # print('time 3', time3)
    print('linear optimization time (s)', time4)
    print('Total Computation time (s)', timeTotal)

    # store deiagnostic information in a single dictionary
    info = dict(xlist = xlist,
                clist = clist,
                lamlist = lamlist,
                Philist = Philist)

    return x, info



def OC(nelx, nely, x, volFrac, dc, dv, ft, H, Hs):
    """ Optimality Criterion Update"""

    """ This code looks for the value of the lagrange multiplier that satisfies the volume criterion.
        Things to remember:
            1)  smaller "l" gives a larger Delta_x.
            2)  "c" is the strain energy and decreases monotonically with Delta_x
            3)  "l" is effectively the limit on allowable strain energy. As this limit is made smaller, more
                and more material is required. Thus, we decrease the limit until the volume constraint is reached.
    """

    # l1 = 1e-32
    # l1 = 1e-16
    # l2 = 1e290

    l1 = 1e-256
    l2 = 1e256

    # l1 = 1e-16
    # l2 = 1e16
    # move = 0.2
    move = 0.05
    # move = 0.02

    while (l2-l1)/(l2+l1) > 1e-5:

        lmid = 0.5 * (l2+l1)

        # print(x * np.sqrt(-dc)/lmid)
        # print(np.minimum(x+move,x * np.sqrt(-dc)/lmid))

        # The update is calculated using a simple heuristic algorithm (Bendsoe 1995).
        # The step size for each parameter is limited to be smaller than "move". Furthermore,
        # x is required to be between 0.001 and 1.
        xnew = np.maximum(0.001,
                    np.maximum(x - move,
                        np.minimum(1.0,
                            np.minimum(x + move,
                                x * np.sqrt(np.maximum(0,-dc)/(dv*lmid)))
                                   )
                               )
                         )
        # xnew = x * np.sqrt(np.maximum(0,-dc)/(dv*lmid))
        # xnew(xnew>1) = 1
        # xnew(xnew>x+move)=x+move
        # xnew(xnew<0.001) = 0.001
        # xnew(xnew<x-move)=(x-move)[]

        # xnew = x*np.sqrt(0,-dc)/lmid

        # ensure that x does not exceed "move"
        # xmove = x+move
        # xnew[xnew > xmove] = xmove[]

        if ft == 1:
            xPhys = xnew
        elif ft == 2:
            xPhys = (H @ xnew) / Hs


        vol = np.sum(xPhys)/(nelx*nely)
        if vol > volFrac:
            l1 = lmid
        else:
            l2 = lmid

    # print(lmid)

    return xnew


# def check2(nelx, nely, rmin, x, dc):
#     """Mesh Independency Filter"""
#     """Let's see if we can vectorize this a bit"""
#     rmin2 = math.ceil(rmin)
#
#     # pad  dc array with zeros
#     dcp = np.zeros((nely+2*rmin2, nelx+2*rmin2))
#     dcp[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = np.reshape(dc,(nely,nelx), order = 'F')
#
#     # pad x array with zeros
#     xp = np.zeros((nely+2*rmin2, nelx+2*rmin2))
#     xp[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = np.reshape(x,(nely,nelx), order = 'F')
#
#     # create a boolean array that is true for the domain elements and false for the padding elements
#     onep = np.zeros((nely+2*rmin2, nelx+2*rmin2))
#     onep[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = 1
#
#     # determine convolution pencil
#     ijfac = []
#     for i in range(-rmin2, rmin2):
#         for j in range(-rmin2, rmin2):
#             r = np.sqrt(i**2+j**2)
#             if r<rmin:
#                 fac = rmin-r
#                 ijfac.append([i,j,fac])
#
#
#     # compute filtered dc array
#     dcn = np.zeros((nely,nelx))
#     sumVal = np.zeros((nely,nelx))
#     for [i,j,fac] in ijfac:
#         dcn = dcn + fac*(xp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)]
#                          *dcp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])
#         sumVal = sumVal + fac*(onep[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])
#
#
#
#     dcn = dcn.flatten(order = 'F')[:,None]
#     sumVal = sumVal.flatten(order = 'F')[:,None]
#     dcn = dcn/(x*sumVal)
#
#     return dcn

def check3(H,Hs,x,dc):

    dcn = ((H @ (dc * x)) / (Hs * x))

    return dcn

def convMat(nelx,nely,rmin):
    """Let's see if we can vectorize this a bit"""
    rmin2 = math.ceil(rmin)

    # determine convolution pencil (i and j are the x and y offsets in the pencil,
    # fac is the constant multiplier associated with that pencil location)
    ijfac = []
    for i in range(-rmin2, rmin2):
        for j in range(-rmin2, rmin2):
            r = np.sqrt(i**2+j**2)
            if r<rmin:
                # fac = r-rmin
                fac = rmin-r
                ijfac.append([i,j,fac])

    # number of elements in pencil
    nPen = len(ijfac)

    # create a node index array
    nEles = nelx*nely
    ele = np.reshape(np.arange(0,nEles),(nely,nelx),order = 'F')

    # add negative one padding to the node index array
    elep = -1*np.ones((nely+2*rmin2, nelx+2*rmin2),dtype = int)
    elep[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = ele

    # compute filtered dc array
    # dcn = np.zeros((nely,nelx))
    # sumVal = np.zeros((nely,nelx))
    Hrow = np.zeros(nPen*nelx*nely,dtype = int)
    Hcol = np.zeros(nPen*nelx*nely,dtype = int)
    Hval = np.zeros(nPen*nelx*nely,dtype = float)
    count = 0
    for [i,j,fac] in ijfac:
        Hrow[(count+0):(count+nEles)] = np.arange(0,nEles)
        Hcol[(count+0):(count+nEles)] = elep[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)].flatten(order ='F')
        Hval[(count+0):(count+nEles)] = fac

        count += nEles


    # remove padding elements
    mask = Hcol>0
    Hrow = Hrow[mask]
    Hcol = Hcol[mask]
    Hval = Hval[mask]


    # generate the sparse matrix representation of the convolution matrix
    H = scipy.sparse.coo_matrix((Hval,(Hrow,Hcol)))
    H = scipy.sparse.csr_matrix(H)

    # row summation
    Hs = np.array(np.sum(H,axis = 1))

    return H, Hs

# def FE_Eigs(nelx, nely, KE, ME, Mlump, x, penal, irow, icol,  E0, E1, rho0, rho1, volFrac):
#
#     nDOF = 2 * (nely + 1) * (nelx + 1)
#     freeDOFs = FE_BCs(nelx, nely)[0]
#
#     # form system mass and stiffness matrices
#     K,M = FE_KM(nelx, nely, KE, ME, Mlump, x, penal, irow, icol,  E0, E1, rho0, rho1, volFrac)
#
#     # Solve for eigenvalues
#     # neigs = 30

#     neigs = 1
#     L, P = sla.eigsh(K, M=M, k=neigs, sigma=0)
#     w = np.sqrt(L)
#
#     # mass normalize P
#     for i in range(neigs):
#         P[:,i] = P[:,i]/np.sqrt(P[:,i].T @ M @ P[:,i])
#
#     # store full eigenvalue (including fixed DOFs)
#     Pfull = np.zeros((nDOF,neigs))
#     Pfull[freeDOFs,:] = P
#
#     return w, Pfull

# def FE_BCs(nelx, nely):

#     nDOF = 2 * (nely + 1) * (nelx + 1)

#     # fixed degrees of freedom
#     fixedDOFs = np.arange(0,2*(nely+1))
#     allDOFs = np.arange(0, nDOF)
#     freeDOFs = np.setdiff1d(allDOFs,fixedDOFs)

#     return freeDOFs, fixedDOFs

def FE_KM(nelx, nely, KE, ME, Mlump, ilump, x, penal, irow, icol,  E0, E1, rho0, rho1, volFrac, freeDOFs, fixedDOFs):

    # SIMP interpolation
    E = E0 + x**penal * (E1-E0)
    rho = rho0 + x * (rho1-rho0)

    # form stiffness matrix
    KEvec = np.reshape(KE, (-1,1))
    Kval = (E @ KEvec.T).flatten(order = 'C')
    K = scipy.sparse.coo_matrix((Kval,(irow,icol)))

    # form mass matrix
    MEvec = np.reshape(ME, (-1,1))
    Mval = (rho @ MEvec.T).flatten(order = 'C')
    M = scipy.sparse.coo_matrix((Mval,(irow,icol)))

    # convert stiffness to csr sparse representation
    K = scipy.sparse.csr_matrix(K)
    M = scipy.sparse.csr_matrix(M)

    # print(np.sum(M,axis=None))

    # add lumped mass at middle right location
    # ilump = np.array([2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-2,
    #                   2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-1])
    M[ilump,ilump] = M[ilump,ilump] + Mlump

    # keep only "free" rows and columns of stiffness matrix and force vector
    # Note, this can be done in a single call, but for some reason, the two-stage call is much much faster.

    # freeDOFs = FE_BCs(nelx, nely)[0]O

    Khat = K[freeDOFs,:]
    Khat = Khat[:,freeDOFs]

    Mhat = M[freeDOFs,:]
    Mhat = Mhat[:,freeDOFs]

    # symmetrize mass and stiffness matrices
    Khat = (Khat + Khat.T)/2
    Mhat = (Mhat + Mhat.T)/2

    return Khat, Mhat

def FEPattern(eDOF):
    """Create the i,j index vectors to assemble K"""

    # number of elements
    nEle = np.shape(eDOF)[0]

    # number of entries in element stiffness
    nKE = np.shape(eDOF)[1]**2

    # define row and column indices for K
    irow = np.empty(nKE*nEle)
    icol = np.empty(nKE*nEle)
    count = 0
    for el in range(nEle):

        irow[(count+0):(count+nKE)] = np.repeat(eDOF[el,:],8)
        icol[(count+0):(count+nKE)] = np.tile(eDOF[el,:],8)

        count += 8**2

    return irow, icol

def eDOFMaker(nelx,nely):

    nNodes = (nelx+1)*(nely+1)
    nodeArray = np.reshape(np.arange(0,nNodes),(nely+1,nelx+1),order = 'F')


    # element node array
    eNode = np.hstack((nodeArray[0:-1 ,0:-1].flatten(order = 'F')[:,None],
                       nodeArray[0:-1, 1:].flatten(order = 'F')[:,None],
                       nodeArray[1:   ,0:-1].flatten(order = 'F')[:,None],
                       nodeArray[1:   ,1:].flatten(order = 'F')[:,None]))


    # element DOF array
    eDOF = np.empty((nelx*nely,8),dtype = int)
    eDOF[:,0::2] = eNode*2
    eDOF[:,1::2] = eNode*2 + 1

    return eDOF

def eleK(l):

    # element properties
    nu = 0.3
    E = 1
    g = 1 # g = a/b (rectangular aspect ratio)
    h = 1 # element thickness
    a = l
    b = l

    k = [1/2 - nu/6,    1/8 + nu/8,      -1/4 - nu/12,      -1/8 + 3*nu/8,
         -1/4 + nu/12,  -1/8 - nu/8,     nu/6,              1/8 - 3*nu/8]

    KE = (E/(1-nu**2))*np.array(
         [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
          [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
          [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
          [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
          [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
          [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
          [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
          [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    # # plane stress stiffness formulation (taken from C. Felippa Course notes)
    # k = [(1+nu)*g, (1-3*nu)*g, 2+(1-nu)*g**2, 2*g**2+(1-nu),
    #      (1-nu)*g**2-4, (1-nu)*g**2-1, 4*g**2-(1-nu), g**2-(1-nu)]

    # KE = (h/(24*g*(1-nu**2)))*np.array(
    #     [[ 4*k[2],  3*k[0],  2*k[4], -3*k[1], -2*k[2], -3*k[0], -4*k[5],  3*k[1]],
    #      [ 3*k[0],  4*k[3],  3*k[1],  4*k[7], -3*k[0], -2*k[3], -3*k[1], -2*k[6]],
    #      [ 2*k[4],  3*k[1],  4*k[2], -3*k[0], -4*k[5], -3*k[1], -2*k[2],  3*k[0]],
    #      [-3*k[1],  4*k[7], -3*k[0],  4*k[3],  3*k[1], -2*k[6],  3*k[0], -2*k[3]],
    #      [-2*k[2], -3*k[0], -4*k[5],  3*k[1],  4*k[2],  3*k[0],  2*k[4], -3*k[1]],
    #      [-3*k[0], -2*k[3], -3*k[1], -2*k[6],  3*k[0],  4*k[3],  3*k[1],  4*k[7]],
    #      [-4*k[5], -3*k[1], -2*k[2],  3*k[0],  2*k[4],  3*k[1],  4*k[2], -3*k[0]],
    #      [ 3*k[1], -2*k[6],  3*k[0], -2*k[3], -3*k[1],  4*k[7], -3*k[0],  4*k[3]]])


    return KE

def eleM(l):

    # l = 1
    g = 1 # g = a/b (rectangular aspect ratio)
    h = 1 # element thickness
    a = l
    b = l

    # # element order
    # order = 1

    # # number of element nodes
    # n = (order+1)**2

    # # form polynomials and normalize
    # rx = np.linspace(-1,1,order+1)
    # for i in range(order+1):

    #     # represent polynomial with its roots
    #     rxi = rx

    #     # remove ith root to generate Lagrange polynomial
    #     rxi = np.delete(x, (i), axis=0)

    #     # normalize the result so that the value at the ith node is 1
    #     rxi = rxi/np.product((rxi-rx(i)), axis = 0)


    # # compute Gauss Legendre points and weights
    # xG, wG = numpy.polynomial.legendre.leggauss(n)



    ME = (a*b*h/36)*np.array(
        [[4, 0, 2, 0, 1, 0, 2, 0],
         [0, 4, 0, 2, 0, 1, 0, 2],
         [2, 0, 4, 0, 2, 0, 1, 0],
         [0, 2, 0, 4, 0, 2, 0, 1],
         [1, 0, 2, 0, 4, 0, 2, 0],
         [0, 1, 0, 2, 0, 4, 0, 2],
         [2, 0, 1, 0, 2, 0, 4, 0],
         [0, 2, 0, 1, 0, 2, 0, 4]])

    return ME

# def eleK2():

#     return

# def exampleProblem(ny = ):





def example2DCantilever(nelx, nely):

    # mesh density
    nelx = 3*nely

    # model size
    L = 1
    Lx = 3*L
    Ly = L

    # Percentage of Volume to fill with material
    volFrac = 0.3

    # number of Degrees of Freedom
    nDOF = 2 * (nely + 1) * (nelx + 1)

    # fixed degrees of freedom
    fixedDOFs = np.arange(0,2*(nely+1))
    allDOFs = np.arange(0, nDOF)
    freeDOFs = np.setdiff1d(allDOFs,fixedDOFs)

    # add lumped mass at middle right location
    ilump = np.array([2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-2,
                      2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-1])

    Mlump = 0.2*Lx*Ly*volFrac


    # x, info = topOptHarmonic.topo2D(nelx, nely, Lx, volFrac, penal, rmin,
    #                                  maxiter, costFunc, power)

    def xCon(x,nelx,nely):

        # x = np.reshape(x,(nely,nelx))
        # x = x + np.fliplr(x)
        # x = x + np.flipud(x)
        # x = x * (1/4)

        return x

    # x, info = topo2D(nelx, nely, Lx, volFrac, freeDOFs, fixedDOFs, Mlump, ilump,
    #     rmin = rmin, penal = penal, maxiter = maxiter,
    #     costFunc = costfunc, power = power, xConstraint = xCon,)
    # return x, info
    return Lx, Ly, volFrac, freeDOFs, fixedDOFs, Mlump, ilump, xCon

def example2DSymmetric(nelx, nely):

    # mesh density
    # nely = int(np.round(nelx*0.9))

    # model size
    Lx = 1
    Ly = (nely/nelx)*Lx

    # Percentage of Volume to fill with material
    volFrac = 0.3

    # number of Degrees of Freedom
    nDOF = 2 * (nely + 1) * (nelx + 1)

    # fixed degrees of freedom
    up1down1 = np.arange(-1,2)
    left1right1 = np.arange(-(nely+1), nely+2, nely+1)
    fixednodes1 = int((nely+1)/2) + up1down1
    fixednodes2 = nelx*(nely+1) + int((nely+1)/2) + up1down1
    fixednodes3 = int((nelx+1)/2)*(nely+1) + left1right1
    fixednodes4 = int((nelx+1)/2+1)*(nely+1)-1 + left1right1

    fixedNodes = np.concatenate((fixednodes1,
                                fixednodes2,
                                fixednodes3,
                                fixednodes4), axis = 0)
    # print(np.shape(fixedNodes))

    fixedDOFs = np.empty((np.shape(fixedNodes)[0]*2,))
    fixedDOFs[0::2] = fixedNodes*2
    fixedDOFs[1::2] = fixedNodes*2 + 1

    allDOFs = np.arange(0, nDOF)
    freeDOFs = np.setdiff1d(allDOFs,fixedDOFs)

    # add lumped mass at middle right location
    # ilumpNodes = np.array([int((nely+1)/2)*int((nelx+1)/2)])
    ilumpNodes = np.array([(nely+1)*int((nelx+1)/2) + int((nely+1)/2)])
    ilump = np.empty((np.shape(ilumpNodes)[0]*2,))
    ilump[0::2] = ilumpNodes*2
    ilump[1::2] = ilumpNodes*2 + 1

    print('lumped mass node = ', ilumpNodes)

    # ilump = np.array([2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-2,
                      # 2*(nelx+1)*(nely+1)-2*int((nely+1)/2)-1])

    Mlump = 0.2*Lx*Ly*volFrac


    # x, info = topOptHarmonic.topo2D(nelx, nely, Lx, volFrac, penal, rmin,
    #                                  maxiter, costFunc, power)

    def xCon(x,nelx,nely):

        # symmetrize design
        shapex = np.shape(x)
        x = np.reshape(x,(nely,nelx), order = 'F')
        x = x + np.fliplr(x)
        x = x + np.flipud(x)
        x = x * (1/4)
        x = np.reshape(x, shapex, order = 'F')

        return x

    # x, info = topo2D(nelx, nely, Lx, volFrac, freeDOFs, fixedDOFs, Mlump, ilump,
    #     rmin = rmin, penal = penal, maxiter = maxiter,
    #     costFunc = costfunc, power = power, xConstraint = xCon)
    return Lx, Ly, volFrac, freeDOFs, fixedDOFs, Mlump, ilump, xCon
