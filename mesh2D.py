import numpy as np
import math


def SquareWithInclusion(mrl, n, options):

    # default meshing internal parameters
    # TODO allow these parameters to be overwritten with the values passed in
    # through options
    # TODO tune the default values to match what is being done in the GBMS
    # paper

    r_c = 0.25
    #  r_c = 0.35
    r_ss = r_c*0.5
    r_bs = 0.5
    circ_dist = 0.5
    #  circ_dist = 0.8
    circ_dist = 0
    n_petals = 8
    theta_offset = 0*math.pi/8

    mrlr = mrl
    mrlt = 3*mrl

    x_ss = np.linspace(-r_ss, r_ss, n*mrlt+1)
    x_ss, y_ss = np.meshgrid(x_ss, x_ss)
    x_ss = np.transpose(x_ss, (1, 0))
    y_ss = np.transpose(y_ss, (1, 0))

    # allocate quadrant coordinate arrays
    x_quad = np.zeros((2*(n*mrlr+1), n*mrlt+1))
    y_quad = np.zeros((2*(n*mrlr+1), n*mrlt+1))

    thetas = np.linspace(-math.pi/4, math.pi/4, n*mrlt+1)
    for i in range(0, n*mrlt+1):
        theta = thetas[i]

        x_start = r_ss
        x_end = r_bs
        x_middle = r_c*np.cos(theta)
        x_vec1 = np.linspace(x_start, x_middle, n*mrlr+1)
        x_vec2 = np.linspace(x_middle, x_end, n*mrlr+1)
        x_vec = np.concatenate((x_vec1, x_vec2), axis=0)

        y_start = y_ss[-1, i]
        y_end = r_bs*np.tan(theta)
        y_middle = r_c*np.sin(theta)
        y_vec1 = np.linspace(y_start, y_middle, n*mrlr+1)
        y_vec2 = np.linspace(y_middle, y_end, n*mrlr+1)
        y_vec = np.concatenate((y_vec1, y_vec2), axis=0)

        x_quad[:, i] = x_vec
        y_quad[:, i] = y_vec

    # collect small-square coordinates and quadrant coordinates into a
    # coordinate array. Note quadrant coordinates are rotated and appended 4
    # times.
    x = np.vstack((x_ss, x_quad, -y_quad, -x_quad, y_quad))
    y = np.vstack((y_ss, y_quad, x_quad, -y_quad, -x_quad))

    # flatten and collect coordinates
    x = x.flatten()
    y = y.flatten()
    coordinates = np.transpose(np.vstack((x, y)))

    # form element node index for interior square (each row containst the node
    # indices for an element)
    index_ss = np.reshape(
        np.arange(0, (n*mrlt+1)**2),
        ((n*mrlt+1), (n*mrlt+1)))
    emat_ss = np.zeros((mrlt**2, (n+1)**2), dtype='int')
    for i in range(0, mrlt):
        for j in range(0, mrlt):
            emat_ss[i*mrlt+j, :] = index_ss[(i*n):(i*n+n+1),
                                            (j*n):(j*n+n+1)].flatten()

    # form element node index for one quadrant (each row containst the node
    # indices for an element)
    index_quad = np.reshape(
        np.arange(0, (n*mrlt+1)*(n*mrlr+1)),
        ((n*mrlr+1), (n*mrlt+1)))
    emat_quad = np.zeros((mrlt*mrlr, (n+1)**2), dtype='int')
    for i in range(0, mrlr):
        for j in range(0, mrlt):
            emat_quad[i*mrlt+j, :] = index_quad[(i*n):(i*n+n+1),
                                                (j*n):(j*n+n+1)].flatten()

    # concatenate element matrix indices from small square with those from the
    # quad segments
    emat = np.copy(emat_ss)
    for i in range(0, 2*4):
        emat = np.vstack((emat, emat_quad + emat.max()+1))

    # round coordinates to prepare for sorting and unique-ness test
    roundcoordinates = np.round(coordinates, 12)

    # note that at this point the nodes overlap at the edge of each "square" of
    # nodes. We need to find unique nodes and rempap the element matrix indices
    # to refer to this new unique set of nodes
    unique_indices, unique_inverse = np.unique(
        roundcoordinates,
        axis=0,
        return_index=True,
        return_inverse=True)[1:3]

    emat = unique_inverse[emat]
    coordinates = coordinates[unique_indices, :]

    def distortnodes(coordinates):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        thetas = np.arctan2(y, x)
        radii = np.sqrt(x**2 + y**2)

        # compute for each node the length of a radial line extending to the
        # unit-cell's edge
        r_edge = abs(r_bs/np.cos(thetas))
        r_edge[abs(x) < abs(y)] = abs(r_bs/np.sin(thetas[abs(x) < abs(y)]))

        # decide on type of radial distortion here
        model_select = 'helicoid_catenoid'
        if model_select == 'helicoid_catenoid':
            theta_dist = (1 - circ_dist ** 2) * (1 + circ_dist /
                                                 (1 + circ_dist * np.cos(n_petals * (thetas - theta_offset)))) - 1

        radii_old = np.copy(radii)
        dist_select = 4
        if dist_select == 1:
            # first compute "rad_dist" which restricts that extent to which a node can
            # be distorted. (prevents any distortion at the center and at the edge of
            # the square)

            # mesh distortion inside inclusion
            # power=0 stable result, mesh may have strong distortion in center
            # power=1 less stable, less badly distorted mesh
            rad_dist = (radii/r_c)**(0)

            # mesh distortion outside inclusion
            i_o = radii > (r_c-1e-10)
            rad_dist[i_o] = 1-(radii[i_o] - r_c)/(r_edge[i_o] - r_c)
            #  rad_dist[i_o] = rad_dist[i_o]*(r_bs-r_c)/radii[i_o]
            rad_dist[i_o] = rad_dist[i_o]*(r_c/radii[i_o])

            radii = radii*(1+rad_dist*theta_dist)
            #  radii = radii*(1+rad_dist)

        elif dist_select == 2:

            # new radius of circle as a function of theta
            r_cp = r_c*(1 + theta_dist)

            # new radial coordinates for points outside inclusion
            i_o = radii >= (r_c-1e-10)
            r_o = r_cp + (r_edge-r_cp)*(radii - r_c)/(r_edge-r_c)

            # new radial coordinates for points inside inclusion
            i_i = radii < (r_c-1e-10)
            r_i = radii*(r_cp/r_c)
            #  exponent = 0
            #  r_i = radii*(1+(r_cp/r_c-1) * (radii/r_c)**(exponent))

            radii[i_o] = r_o[i_o]
            radii[i_i] = r_i[i_i]

        elif dist_select == 3:

            # new radius of circle as a function of theta
            r_cp = r_c*(1 + theta_dist)

            # new radial coordinates for points outside inclusion
            i_o = radii >= (r_c-1e-10)
            #  z = radii/r_c
            z = (r_c-radii)/(r_c-r_edge)
            c = r_edge
            b = -(r_edge-r_c)
            a = r_cp-b-c
            r_o = a*z**2+b*z+c
            #  z = (r_c-radii)/(r_c-r_edge)
            #  a = (1-(r_cp-r_edge)/(r_c-r_edge))/((r_cp**2-r_edge**2)-2*r_edge*(r_cp-r_edge))
            #  b = 1/(r_c-r_edge) - 2*a*r_edge
            #  f = radii/(r_c-r_edge) - r_edge/(r_c-r_edge)
            #  c = -a*r_edge**2 -b*r_edge

            #  r_o = (-b+np.sqrt(b))
            #  r_o = (-b + np.sqrt(b**2 - 4*a*(c-f)))/(2*a)
            #  r_o = radii

            x_o = r_o[i_o]*np.cos(thetas[i_o])
            y_o = r_o[i_o]*np.sin(thetas[i_o])

            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(x_o, y_o, '.')
            plt.show()

            r_o = r_cp + (r_edge-r_cp)*(radii - r_c)/(r_edge-r_c)

            # new radial coordinates for points inside inclusion
            i_i = radii < (r_c-1e-10)
            #  r_o = radii[i_o]*(1+(r_cp[i_o]/r_c-1)
            #                            * (radii[i_o]/r_c)**(0))
            z = radii/r_c
            r_i = (r_cp-r_c)*z**2 + r_c*z
            #  radiip[i_o] = radiio[i_o]

            # overwrite old radii values with new values
            radii[i_o] = r_o[i_o]
            radii[i_i] = r_i[i_i]

        elif dist_select == 4:

            # new radius of circle as a function of theta
            r_cp = r_c*(1 + theta_dist)

            # new radial coordinates for points outside inclusion
            i_o = radii >= (r_c-1e-10)
            d = (r_c-r_cp)/((r_edge**2-r_c**2)-2*r_edge*(r_edge-r_c))
            e = 1-2*d*r_edge
            f = r_cp-d*r_c**2-e*r_c
            f = r_edge-d*r_edge**2-e*r_edge
            r_o = d*radii**2+e*radii+f

            #  r_o = r_cp + (r_edge-r_cp)*(radii - r_c)/(r_edge-r_c)

            # new radial coordinates for points inside inclusion
            i_i = radii < (r_c-1e-10)
            #  z = radii/r_c
            r_i = ((r_cp-r_c)/r_c**2)*radii**2 + 1*radii
            #  radiip[i_o] = radiio[i_o]

            # overwrite old radii values with new values
            radii[i_o] = r_o[i_o]
            radii[i_i] = r_i[i_i]

        x = radii*np.cos(thetas)
        y = radii*np.sin(thetas)

        if True:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import pyplot as plt
            fig = plt.figure()
            #  ax = fig.add_subplot(111, projection='3d')
            #  plt.plot(radii_old[abs(x-y)<1e-6], radii[abs(x-y)<1e-6], '.')
            i_k = abs(np.tan(math.pi/8)-y/x) < 1e-6
            #  print(any(i_k))
            plt.plot(radii_old[i_k], radii[i_k], '.')
            plt.show()

        #  import math
        #  theta = np.linspace(-1, 2*math.pi, 1000)
        #  helicoid = (1 - circ_dist ** 2) * (1 + circ_dist /
        #                                     (1 + circ_dist * np.cos(n_petals * (theta - theta_offset)))) - 1
        #  r_helicoid = r_c*(1+helicoid)
        #  plt.figure()
        #  plt.plot(r_helicoid*np.cos(theta), r_helicoid*np.sin(theta))
        #  #  plt.plot(theta,helicoid)
        #  plt.show()

        return np.vstack((x, y)).transpose()

    # Distort nodes to form a more interesting inclusion shape
    coordinates = distortnodes(coordinates)
    colorvec = np.concatenate(([0.6] * (mrlt ** 2),  np.tile(
             np.concatenate(([0.6] * (mrlr * mrlt), [0.85] * (mrlr * mrlt)),
                            axis=0), 4)), axis=0)

    return coordinates, emat, colorvec


def plotmesh(coordinates, patchindices, colorvec):

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots()
    plt.clf
    patches = []
    for i in range(0, patchindices.shape[0]):
        polygon = Polygon(coordinates[patchindices[i, :], :], True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet,
                        edgecolors=[(0.8 * x, 0.8 * x, 0.8 * x, 1)
                                    for x in colorvec],
                        facecolors=[(x, x, x, 1) for x in colorvec])

    ax.add_collection(p)

    if False:
        for i in range(0, coordinates.shape[0]):
            plt.text(coordinates[i, 0], coordinates[i, 1], str(i))

    unique_mats = np.unique(colorvec)
    for j in range(0, len(unique_mats)):

        bedges = boundaryedges(patchindices[colorvec == unique_mats[j], :])
        for i in range(0, len(bedges)):
            plt.plot(coordinates[bedges[i], 0],
                     coordinates[bedges[i], 1],
                     'k')

    plt.axis('equal')
    plt.show()
    pass


def boundaryedges(outlines):

    n, m = outlines.shape
    edgeindexlist = np.zeros((n*m, 2), dtype='int')

    for i in range(0, m):
        edgeindexlist[(i*n):((i+1)*n), :] = outlines[:, (i, (i+1) % (m))]

    edgeindexlist = np.sort(edgeindexlist, axis=1)
    edgeindexlist, edgecounts = np.unique(
        edgeindexlist,
        return_counts=True, axis=0)

    edgeindexlist = edgeindexlist[edgecounts == 1, :]
    boundaryedge = []

    # loop through unique edges and connect them (make a list of the nodes in
    # the connecting edges)
    count = 0  # counter variable for number of boundary lists

    while True:
        # as we append edges to the node lists, the values in "edgeindexlist"
        # are overwritten with value -1. Thus, to start a new list, we search
        # for a pair that hasn't been overwritten yet
        startedge = np.nonzero(edgeindexlist != -1)

        if len(startedge[0] > 0):
            boundaryedge.append([edgeindexlist[startedge[0][0], 0],
                                 edgeindexlist[startedge[0][0], 1]])
            edgeindexlist[startedge[0][0], :] = -1

        else:
            break

        while True:
            edgeindexlist[:, 0] == boundaryedge[count][-1]
            nextedge = np.nonzero(
                edgeindexlist == boundaryedge[count][-1])
            if len(nextedge[0]) > 0:
                if nextedge[1][0] == 1:
                    column = 0
                else:
                    column = 1

                boundaryedge[count].append(
                    edgeindexlist[nextedge[0][0], column])
                edgeindexlist[nextedge[0][0], :] = -1
            else:

                boundaryedge[count].append(boundaryedge[count][0])
                count = count + 1
                break

    return boundaryedge


def ShapeFuncEval1D(znode, zeval):

    m = len(znode)
    d = len(zeval)

    # Create array with shape-functions and shape function derivatives
    # evaluated at the quadrature points
    L = np.zeros((m, d))
    dL = np.zeros((m, d))
    for i in range(0, m):

        # the shape function for the ith node has roots at all of the other
        # nodes
        roots = np.copy(znode)
        roots = np.delete(roots, i)

        # convert roots to polynomial coefficients and normalize shape func
        p = np.poly(roots)
        p = p/np.polyval(p, znode[i])

        # compute polynomial derivative
        dp = p[0:-2]*range(1, len(p)-1)

        # evaluate polynomial and polynomial derivatives for current shape
        # function at all of the quadrature points
        L[i, :] = np.polyval(p, zeval)
        dL[i, :] = np.polyval(dp, zeval)

    return L, dL

#  class element:
#      def __init__
#          pass


def QuadPlaneStrain(X, D, rho):

    #  element order
    n = int(np.sqrt(np.shape(X)[0])-1)

    # number of nodes per element edge
    m = n+1

    # gauss quadrature points and weights
    d = n
    zg, wg = np.polynomial.legendre.leggauss(d)

    # form 1D shape functions and evaluate at 1D quadrature points
    zetas = np.linspace(-1, 1, m)
    L, dL = ShapeFuncEval1D(zetas, zg)

    #  preallocate mass and stiffnes matrices
    Ke = np.zeros((2*m**2,2*m**2))
    Me = np.zeros((2*m**2,2*m**2))

    #  loop through quadrature points
    for i in range(0, d):
        for j in range(0, d):

            # use tensor product of 1D shape functions to get the 2D
            # shapefunctions
            N = np.outer(L[:, i], L[:, j]).flatten()
            dNdz = np.outer(dL[:, i], L[:, j]).flatten()
            dNde = np.outer(L[:, i], dL[:, j]).flatten()

            # populate N matrix
            Nmat = np.zeros((2, 2*m**2))
            Nmat[0, 0:2*m**2:2] = N
            Nmat[1, 1:2*m**2:2] = N

            # compute Jacobian and Jacobian determinant
            #  J = np.dot(np.vstack([dNdz, dNde]), X)
            J = np.dot(np.vstack((dNdz, dNde)), X)
            Jdet = np.linalg.det(J)
            Jinv = np.linalg.inv(J)

            # compute dNdx and dNdy
            dNdx = np.dot(Jinv[0,:],vstack((dNdz,dNde)))
            dNdy = np.dot(Jinv[1,:],vstack((dNdz,dNde)))

            # populate strain-displacement matrix, B
            B = np.zeros((3, 2*m**2))
            B[0, 0:2*m**2:2] = dNdx
            B[1, 1:2*m**2:2] = dNdy
            B[2, 1:2*m**2:2] = dNdx
            B[2, 0:2*m**2:2] = dNdy

            # add contribution from current quadrature point into mass and
            # stiffness matrices
            Ke = Ke + wg[i]*wg[j]*Jdet*np.dot(np.transpose(B), np.dot(D, B))
            Me = Me + wg[i]*wg[j]*Jdet*rho*np.dot(np.transpose(Nmat), Nmat)

    return Ke, Me

def KMAssemble(mesh):

    from scipy.sparse import lil_matrix

    dpn = 2
    n_DOF = dpn*mesh.n_nodes

    # preallocate mass and stiffness matrices
    K = lil_matrix((n_DOF,n_DOF))
    M = lil_matrix((n_DOF,n_DOF))

    # loop through elements
    for ii in range(mesh.n_eles):


        Ke,Me = mesh.element[i].KM()


    pass
