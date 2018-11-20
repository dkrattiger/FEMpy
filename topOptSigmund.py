import numpy as np
# from scipy.sparse import lil_matrix
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
import math

def topo2D(nelx,nely,volFrac,penal,rmin,maxiter = 100):
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

    # Initialize arrays and counters
    x = np.ones((nely*nelx,1))*volFrac
    dc = np.zeros(np.shape(x),dtype = float)
    loop = 0
    change = 1

    # define element connectivity matrix
    eDOF = eDOFMaker(nelx,nely)


    # define stiffness matrix sparsity pattern
    start = time.time()
    irow,icol = FEPattern(eDOF)
    time0 = time.time() - start

    start = time.time()
    H,Hs = convMat(nelx,nely,rmin)
    time00 = time.time() - start

    print('Model size: ', 2*(nelx+1)*(nely+1),' DOFs')

    while change>0.01:

        loop += 1
        xOld = x

        #FE Analysis
        start = time.time()
        U, timeslots = FE(nelx, nely, x, penal, irow, icol, timeslots)
        time1 += time.time()-start

        #Objective Function and Sensitivity Analysis
        KE  = eleK()

        # c = 0.0
        start = time.time()

        # the following is a vectorized version of taking c_e = U_e.T * Ke * U_e
        Ue = np.squeeze(U[eDOF]).T

        # compute cost function
        c = np.sum(np.sum((KE @ Ue) * Ue, axis = 0)*(x.T**penal))

        # cost function derivative
        dc = (np.sum((KE @ Ue) * Ue, axis = 0)*(-penal)*(x.T**(penal-1))).T

        time2 += time.time()-start

        # initialize  volume sensitivity
        # dv = np.ones((nely,nelx))

        # filter sensitivities
        start = time.time()

        dc = check3(H, Hs, x, dc)
        # dc = check2(nelx, nely, rmin, x, dc)
        # H = convMat(nelx,nely,rmin)
        time3 += time.time()-start

        # design update by the optimality criterion
        start = time.time()
        x = OC(nelx, nely, x, volFrac, dc)
        time4 += time.time()-start

        # print results
        change = np.max(np.abs(x-xOld))
        print('It.: ', loop,
              'Obj.:', c,
              'Vol.:', np.sum(x)/(nelx*nely),
              'ch.:', change)

        if loop>=maxiter:
            break

    print('time 0', time0)
    print('time 1', time1)
    print(timeslots)
    print('time 2', time2)
    print('time 3', time3)
    print('time 4', time4)
    print(temptime)

    return x



def OC(nelx, nely, x, volFrac, dc):
    """ Optimality Criterion Update: I think this code is doing a line search to find a maximum?
    ...or maybe its actually a minimum?"""

    """ This code looks for the value of the lagrange multiplier that satisfies the volume criterion.
        Things to remember: 
            1)  smaller "l" gives a larger Delta_x.
            2)  "c" is the strain energy and decreases monotonically with Delta_x
            3)  "l" is effectively the limit on allowable strain energy. As this limit is made smaller, more
                and more material is required. Thus, we decrease the limit until the volume constraint is reached.
    """

    l1 = 0
    l2 = 1e9
    move = 0.2

    while (l2-l1)/(l2+l1) > 1e-3:

        lmid = 0.5 * (l2+l1)

        # print(x * np.sqrt(-dc)/lmid)
        # print(np.minimum(x+move,x * np.sqrt(-dc)/lmid))

        # The update is calculated using a simple heuristic algorithm (Bendsoe 1995).
        # The step size for each parameter is limited to be smaller than "move". Furthermore,
        # x is required to be between 0.001 and 1.
        xnew = np.maximum(0.001, np.maximum(0.001,
                                np.maximum(x - move,
                                    np.minimum(1.0,
                                        np.minimum(x + move,
                                            x * np.sqrt(-(dc)/lmid))
                                               )
                                           )
                                     )
                   )
        if np.sum(xnew) - volFrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid

    return xnew

# def check(nelx, nely, rmin, x, dc):
#     """Mesh Independency Filter"""
#
#     dcn = np.zeros((nely, nelx))
#     for i in range(nelx):
#         for j in range(nely):
#             sum = 0.0
#             kmin = int(np.maximum(i-np.round(rmin), 0))
#             kmax = int(np.minimum(i+np.round(rmin), nelx))
#             for k in range(kmin, kmax):
#                 lmin = int(np.maximum(j-np.round(rmin), 0))
#                 lmax = int(np.minimum(j+np.round(rmin), nely))
#                 for l in range(lmin, lmax):
#                     fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
#                     sum = sum + np.maximum(0,fac)
#                     dcn[j,i] = dcn[j,i] + np.maximum(0,fac) * x[l, k] * dc[l,k]
#
#             # normalize dcn array
#             dcn[j,i] = dcn[j,i]/(x[j,i]*sum)
#
#     return dcn

def check2(nelx, nely, rmin, x, dc):
    """Mesh Independency Filter"""
    """Let's see if we can vectorize this a bit"""
    rmin2 = math.ceil(rmin)

    # pad  dc array with zeros
    dcp = np.zeros((nely+2*rmin2, nelx+2*rmin2))
    dcp[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = np.reshape(dc,(nely,nelx), order = 'F')

    # pad x array with zeros
    xp = np.zeros((nely+2*rmin2, nelx+2*rmin2))
    xp[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = np.reshape(x,(nely,nelx), order = 'F')

    # create a boolean array that is true for the domain elements and false for the padding elements
    onep = np.zeros((nely+2*rmin2, nelx+2*rmin2))
    onep[rmin2:(rmin2+nely), rmin2:(rmin2+nelx)] = 1

    # determine convolution pencil
    ijfac = []
    for i in range(-rmin2, rmin2):
        for j in range(-rmin2, rmin2):
            r = np.sqrt(i**2+j**2)
            if r<rmin:
                fac = rmin-r
                ijfac.append([i,j,fac])


    # compute filtered dc array
    dcn = np.zeros((nely,nelx))
    sumVal = np.zeros((nely,nelx))
    for [i,j,fac] in ijfac:
        dcn = dcn + fac*(xp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)]
                         *dcp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])
        sumVal = sumVal + fac*(onep[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])



    dcn = dcn.flatten(order = 'F')[:,None]
    sumVal = sumVal.flatten(order = 'F')[:,None]
    dcn = dcn/(x*sumVal)

    return dcn

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


        # dcn = dcn + fac*(xp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)]
        #                   *dcp[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])
        # sumVal = sumVal + fac*(onep[(rmin2+j):(rmin2+j+nely), (rmin2+i):(rmin2+i+nelx)])

    # dcn = dcn/(x*sumVal)

    # print(Hcol)
    # print(elep)
    # print(elep[0:5,0:5])

    # remove padding elements
    mask = Hcol>0
    Hrow = Hrow[mask]
    Hcol = Hcol[mask]
    Hval = Hval[mask]


    H = scipy.sparse.coo_matrix((Hval,(Hrow,Hcol)))
    H = scipy.sparse.csr_matrix(H)

    # row summation
    Hs = np.array(np.sum(H,axis = 1))

    return H, Hs

def FE(nelx, nely, x, penal, irow, icol, timeslots):
    """Finite Element Analysis"""
    KE = eleK()
    nDOF = 2 * (nely + 1) * (nelx + 1)

    # form stiffness matrix
    start = time.time()
    cK = x**penal
    KEvec = np.reshape(KE, (-1,1))
    Kval = (cK @ KEvec.T).flatten(order = 'C')

    # Assemble Stiffness Matrix
    K = scipy.sparse.coo_matrix((Kval,(irow,icol)))

    timeslots[0] += time.time()-start

    # fixed degrees of freedom
    # np.shape(np.array([2*(nelx+1)*(nely+1)]))
    fixedDOFs = np.union1d(np.arange(0,2*(nely+1),2), np.array([2*(nelx+1)*(nely+1)-1]))
    #print(fixedDOFs)
    #fixedDOFs = np.arange(0,2*(nely+1),2)
    allDOFs = np.arange(0, nDOF)
    freeDOFs = np.setdiff1d(allDOFs,fixedDOFs)

    # Define force vector
    F = scipy.sparse.csc_matrix((nDOF,1))
    #F[2 * (nelx + 1) * (nely + 1) - 1, 0] = -1
    F[1,0] = -1
    F = F.A

    # convert stiffness to csr sparse representation
    K = scipy.sparse.csr_matrix(K)

    # keep only "free" rows and columns of stiffness matrix and force vector
    # Note, this can be done in a single call, but for some reason, the two-stage call is much much faster.
    start = time.time()
    Ffree = F[freeDOFs]
    Kfree = K[freeDOFs,:]
    Kfree = Kfree[:,freeDOFs]
    Kfree = Kfree + Kfree.T
    timeslots[1] += time.time()-start

    # Solve
    start = time.time()
    U = np.zeros((nDOF,1))
    soln = scipy.sparse.linalg.spsolve(Kfree, Ffree, use_umfpack=True)
    timeslots[2] += time.time() - start


    start = time.time()
    U[freeDOFs, 0] = soln
    timeslots[3] += time.time()-start


    return U, timeslots
#
# def FE_Dynamic(nelx, nely, x, penal, irow, icol, timeslots):
#     """Finite Element Analysis"""
#     KE = eleK()
#     ME = eleM()
#     nDOF = 2 * (nely + 1) * (nelx + 1)
#
#     # form stiffness matrix
#     start = time.time()
#     cK = x**penal
#     KEvec = np.reshape(KE, (-1,1))
#     Kval = (cK @ KEvec.T).flatten(order = 'C')
#     K = scipy.sparse.coo_matrix((Kval,(irow,icol)))
#
#     # form mass matrix
#     MEvec = np.reshape(ME, (-1,1))
#     Mval = (x @ KEvec.T).flatten(order = 'C')
#     M = scipy.sparse.coo_matrix((Mval,(irow,icol)))
#
#     timeslots[0] += time.time()-start
#
#     # fixed degrees of freedom
#     # np.shape(np.array([2*(nelx+1)*(nely+1)]))
#     fixedDOFs = np.union1d(np.arange(0,2*(nely+1),2), np.array([2*(nelx+1)*(nely+1)-1]))
#     #print(fixedDOFs)
#     #fixedDOFs = np.arange(0,2*(nely+1),2)
#     allDOFs = np.arange(0, nDOF)
#     freeDOFs = np.setdiff1d(allDOFs,fixedDOFs)
#
#     # Define force vector
#     F = scipy.sparse.csc_matrix((nDOF,1))
#     #F[2 * (nelx + 1) * (nely + 1) - 1, 0] = -1
#     F[1,0] = -1
#     F = F.A
#
#     # convert stiffness to csr sparse representation
#     K = scipy.sparse.csr_matrix(K)
#     M = scipy.sparse.csr_matrix(M)
#
#     # keep only "free" rows and columns of stiffness matrix and force vector
#     # Note, this can be done in a single call, but for some reason, the two-stage call is much much faster.
#     start = time.time()
#     Ffree = F[freeDOFs]
#     Kfree = K[freeDOFs,:]
#     Kfree = Kfree[:,freeDOFs]
#     Kfree = Kfree + Kfree.T
#
#     Mfree = M[freeDOFs,:]
#     Mfree = Mfree[:,freeDOFs]
#     Mfree = Mfree + Kfree.T
#     timeslots[1] += time.time()-start
#
#     # Solve
#     start = time.time()
#     U = np.zeros((nDOF,1))
#     soln = scipy.sparse.linalg.spsolve(Kfree, Ffree, use_umfpack=True)
#     timeslots[2] += time.time() - start
#
#
#     start = time.time()
#     U[freeDOFs, 0] = soln
#     timeslots[3] += time.time()-start
#
#
#     return U, timeslots

# def KPattern(nelx,nely):
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
                       nodeArray[1:   ,1:].flatten(order = 'F')[:,None],
                       nodeArray[1:   ,0:-1].flatten(order = 'F')[:,None]))

    # element DOF array
    eDOF = np.empty((nelx*nely,8),dtype = int)
    eDOF[:,0::2] = eNode*2
    eDOF[:,1::2] = eNode*2+1

    return eDOF

def eleK():
    E = 1
    nu = 0.3
    g = 1 # g = a/b (rectangular aspect ratio)
    h = 1 # element thickness

    # k = [1/2 - nu/6,    1/8 + nu/8,      -1/4 - nu/12,      -1/8 + 3*nu/8,
    #      -1/4 + nu/12,  -1/8 - nu/8,     nu/6,              1/8 - 3*nu/8]

    # KE = (E/(1-nu**2))*np.array(
    #      [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    #       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    #       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    #       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    #       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    #       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    #       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    #       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    #      ]
    #     )


    # plane stress stiffness formulation (taken from C. Felippa Course notes)
    k = [(1+nu)*g, (1-3*nu)*g, 2+(1-nu)*g**2, 2*g**2+(1-nu),
         (1-nu)*g**2-4, (1-nu)*g**2-1, 4*g**2-(1-nu), g**2-(1-nu)]

    KE = (E*h/(24*g*(1-nu**2)))*np.array(
        [[ 4*k[2],  3*k[0],  2*k[4], -3*k[1], -2*k[2], -3*k[0], -4*k[5],  3*k[1]],
         [ 3*k[0],  4*k[3],  3*k[1],  4*k[7], -3*k[0], -2*k[3], -3*k[1], -2*k[6]],
         [ 2*k[4],  3*k[1],  4*k[2], -3*k[0], -4*k[5], -3*k[1], -2*k[2],  3*k[0]],
         [-3*k[1],  4*k[7], -3*k[0],  4*k[3],  3*k[1], -2*k[6],  3*k[0], -2*k[3]],
         [-2*k[2], -3*k[0], -4*k[5],  3*k[1],  4*k[2],  3*k[0],  2*k[4], -3*k[1]],
         [-3*k[0], -2*k[3], -3*k[1], -2*k[6],  3*k[0],  4*k[3],  3*k[1],  4*k[7]],
         [-4*k[5], -3*k[1], -2*k[2],  3*k[0],  2*k[4],  3*k[1],  4*k[2], -3*k[0]],
         [ 3*k[1], -2*k[6],  3*k[0], -2*k[3], -3*k[1],  4*k[7], -3*k[0],  4*k[3]],
         ])

    return KE

def eleM():
    rho = 1
    g = 1 # g = a/b (rectangular aspect ratio)
    h = 1 # element thickness
    a = 1
    b = 1

    # plane stress mass formulation (taken from C. Felippa Course notes)

    ME = (rho*a*b*h/72)*np.array(
        [[4, 0, 2, 0, 1, 0, 2, 0],
         [0, 4, 0, 2, 0, 1, 0, 2],
         [2, 0, 4, 0, 2, 0, 1, 0],
         [0, 2, 0, 4, 0, 2, 0, 1],
         [1, 0, 2, 0, 4, 0, 2, 0],
         [0, 1, 0, 2, 0, 4, 0, 2],
         [2, 0, 1, 0, 2, 0, 4, 0],
         [0, 2, 0, 1, 0, 2, 0, 4]])

    return ME
