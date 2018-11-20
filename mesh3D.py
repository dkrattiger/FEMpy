import math
import numpy as np

class Material:

    def extendElasticProps(self):
        nLameProps = 0

        if hasattr(self, 'E'):
            nLameProps += 1
            E = self.E

        if hasattr(self, 'nu'):
            nLameProps += 1
            nu = self.nu

        if hasattr(self, 'lam'):
            nLameProps += 1
            lam = self.lam

        if hasattr(self, 'G'):
            nLameProps += 1
            G = self.G

        if nLameProps < 2:
            print('two of the following fields must exist: "E", "nu", "lam", "G"')
            raise

        # define E if it doesn't exist
        if not hasattr(self, 'E'):
            if hasattr(self, 'nu'):
                if hasattr(self, 'lam'):
                    E = lam*(1+nu)*(1-2*nu)/nu
                elif hasattr(self, 'G'):
                    E = 2*G*(1+nu)
            else:
                E = G*(3*lam+2*G)/(lam+G)

        # define nu if it doesn't exist
        if not hasattr(self, 'u'):
            if hasattr(self, 'lam'):
                R = np.sqrt(E**2+9*lam**2+2*E*lam)
                nu = 2*lam/(E+lam+R)
            elif hasattr(self, 'G'):
                nu = E/(2*G)-1

        # define lam and G even if they already exist
        lam = E*nu/((1+nu)*(1-2*nu))
        G = E/(2*(1+nu))

        # assign properties to attributes
        self.E = E
        self.nu = nu
        self.lam = lam
        self.G = G

    def ElastMatPlaneStrain(self):

        # expand properties
        self.extendElasticProps()

        # extract necessary properties
        lam = self.lam
        G = self.G

        # plane strain elasticity matrix
        D = np.array([[lam+2*G, lam, 0],
                      [lam, lam+2*G, 0],
                      [0, 0, G]])
        return(D)


class element3D(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        pass

    def ShapeFuncEval1D(self, znode, zeval):

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
            dp = p[0:-1]*np.arange(len(p)-1,0,-1)

            # evaluate polynomial and polynomial derivatives for current shape
            # function at all of the quadrature points
            L[i, :] = np.polyval(p, zeval)
            dL[i, :] = np.polyval(dp, zeval)


        return L, dL


class Brick(element3D):

    def __init__(self, x, y, z):
        n = int(np.shape(x)[0]**(1/3)-1)
        self.x = x
        self.y = y
        self.z = z
        self.n = n
        pass

    def facePerimeterIndex(self):
        # TODO Need to update this function to reflect  the 3D geometry
        m = self.n+1
        i_perim = np.concatenate(
            (np.arange(0, m-1),
             np.arange(m-1, m**2-1, m),
             np.arange(m**2-1, m**2-m, -1),
             np.arange(m**2-m, m-1, -m)))
        return i_perim

    def KM(self, mat):

        # group coordinates together
        X = np.transpose(np.vstack((self.x, self.y, self.z)))

        #  element order
        n = self.n

        # number of nodes per element edge
        m = n+1

        # material properties
        D = mat.ElastMatPlaneStrain()
        rho = mat.rho

        # gauss quadrature points and weights
        d = n+1
        zg, wg = np.polynomial.legendre.leggauss(d)

        # form 1D shape functions and evaluate at 1D quadrature points
        zetas = np.linspace(-1, 1, m)
        L, dL = self.ShapeFuncEval1D(zetas, zg)

        #  preallocate mass and stiffnes matrices
        Ke = np.zeros((2*m**2, 2*m**2))
        Me = np.zeros((2*m**2, 2*m**2))

        #  loop through quadrature points
        for i in range(0, d):
            for j in range(0, d):

                # use tensor product of 1D shape functions to get the 2D
                # shapefunctions
                #  N = np.outer(L[:, i], L[:, j]).flatten()
                #  dNdz = np.outer(dL[:, i], L[:, j]).flatten()
                #  dNde = np.outer(L[:, i], dL[:, j]).flatten()
                N = np.outer(L[:, i], L[:, j]).flatten('F')
                dNdz = np.outer(dL[:, i], L[:, j]).flatten('F')
                dNde = np.outer(L[:, i], dL[:, j]).flatten('F')

                # populate N matrix
                Nmat = np.zeros((2, 2*m**2))
                Nmat[0, 0:2*m**2:2] = N
                Nmat[1, 1:2*m**2:2] = N

                # compute Jacobian and Jacobian determinant
                J = np.dot(np.vstack((dNdz, dNde)), X)
                Jdet = np.linalg.det(J)
                if Jdet<0:
                    print('negative Jacobian')
                Jinv = np.linalg.inv(J)

                # compute dNdx and dNdy
                dNdx = np.dot(Jinv[0, :], np.vstack((dNdz, dNde)))
                dNdy = np.dot(Jinv[1, :], np.vstack((dNdz, dNde)))

                # populate strain-displacement matrix, B
                B = np.zeros((3, 2*m**2))
                B[0, 0:2*m**2:2] = dNdx
                B[1, 1:2*m**2:2] = dNdy
                B[2, 1:2*m**2:2] = dNdx
                B[2, 0:2*m**2:2] = dNdy

                # add contribution from current quadrature point into mass and
                # stiffness matrices
                Ke = Ke + wg[i]*wg[j]*Jdet*np.dot(
                    np.transpose(B), np.dot(D, B))
                Me = Me + wg[i]*wg[j]*Jdet*rho*np.dot(np.transpose(Nmat), Nmat)

        return Ke, Me


#  class mesh:
#
#      def __init__(self, n, mrl):
#          self.n = n
#          self.mrl = mrl
#          pass
#

class SquareSimple(object):

    def __init__(self, mrl, n):
        self.n = n
        self.mrl = mrl
        self.nodecoords()
        pass

    def nodecoords(self):

        # mesh refinement level (number of divisions along specific mesh
        # segments)
        mrl = self.mrl

        # element order
        n = self.n

        # square "radius"
        r_s = 0.5

        # x and y coordinates of small square
        x = np.linspace(-r_s, r_s, n*mrl+1)
        x, y = np.meshgrid(x, x)
        x = np.transpose(x, (1, 0))
        y = np.transpose(y, (1, 0))

        # flatten and collect coordinates
        x = x.flatten()
        y = y.flatten()
        coordinates = np.transpose(np.vstack((x, y)))

        # form element node index for interior square (each row containst the node
        # indices for an element)
        index_s = np.reshape(
            np.arange(0, (n*mrl+1)**2),
            ((n*mrl+1), (n*mrl+1)))
        #  emat = np.zeros((mrl**2, (n+1)**2), dtype='int')
        emat = []
        for i in range(0, mrl):
            for j in range(0, mrl):
                emat.append(index_s[(i*n):(i*n+n+1),
                                    (j*n):(j*n+n+1)].flatten('F'))


        # Distort nodes to form a more interesting inclusion shape
        materialIndex = np.zeros((mrl**2),dtype=int)

        #  return coordinates, emat, colorvec
        self.coordinates = coordinates
        self.eleNodeIndex = emat
        self.materialIndex = materialIndex

        pass

class mesh(object):

    def __init__(self, mrl, n):

        # element order and mesh refinement level
        self.n = n
        self.mrl = mrl

        # node coordinates, element node index list, and material index
        coordinates, eleNodeIndex, materialIndex = self.nodecoords()
        self.coordinates = coordinates
        self.eleNodeIndex = eleNodeIndex
        self.materialIndex = materialIndex

        # element list
        elements = self.elementList()
        self.elements = elements

        # boundary edges
        edges = self.boundaryEdges()
        self.edges = edges
        pass

    def updateCoordinates(self,coordinatesNew):

        self.coordinates = coordinatesNew

    def elementList(self):

        coordinates = self.coordinates
        eleNodeIndex = self.eleNodeIndex
        #  create a list of element objects
        elements = []
        for i in range(0, len(eleNodeIndex)):
            xEle = coordinates[eleNodeIndex[i], 0]
            yEle = coordinates[eleNodeIndex[i], 1]
            elements.append(QuadPlaneStrain(xEle, yEle))

        return elements

    #  def featureEdges(self.coordinates)
    def boundaryEdges(self):

        unique_mats = np.unique(self.materialIndex)
        edgeList = []
        count = 0  # counter variable for number of boundary lists
        for j in range(0, len(unique_mats)):

            # index of which elements are made from current material
            matInd = np.nonzero(
                unique_mats[j] == self.materialIndex)[0]

            edgeArray = np.empty((2,0), int)

            #  for i in range(0,nEle):
            for i in matInd:
                iOutline = self.elements[i].perimeterIndex()
                iOutline = self.eleNodeIndex[i][iOutline]
                ind2 = np.hstack((np.arange(1,np.shape(iOutline)[0]),0))
                edgeArrayEle = np.vstack((iOutline,iOutline[ind2]))
                edgeArray = np.hstack((edgeArray,edgeArrayEle))

            edgeArray = np.transpose(edgeArray)

            # Keep edges that show up exactly once in edge Array
            edgeArray = np.sort(edgeArray, axis=1)
            edgeArray, edgecounts = np.unique(
                edgeArray,
                return_counts=True, axis=0)

            edgeArray  = edgeArray[edgecounts == 1, :]
            # allocate an empty nested list to store list of node indices
            #  edgeList.append([])

            # loop through unique edges and connect them (make a list of the nodes in
            # the connecting edges)
            while True:

                # as we append edges to the node lists, the values in
                # "edgeArray" are overwritten with value -1. Thus, to start
                #  a new list, we search for a pair that hasn't been overwritten yet
                startedge = np.nonzero(edgeArray != -1)

                if len(startedge[0] > 0):
                    edgeList.append([edgeArray[startedge[0][0], 0],
                                        edgeArray[startedge[0][0], 1]])
                    edgeArray[startedge[0][0], :] = -1

                else:
                    break

                while True:
                    edgeArray[:, 0] == edgeList[count][-1]
                    nextedge = np.nonzero(
                        edgeArray == edgeList[count][-1])
                    if len(nextedge[0]) > 0:
                        if nextedge[1][0] == 1:
                            column = 0
                        else:
                            column = 1

                        edgeList[count].append(
                            edgeArray[nextedge[0][0], column])
                        edgeArray[nextedge[0][0], :] = -1
                    else:

                        edgeList[count].append(edgeList[count][0])
                        count = count + 1
                        break
        return edgeList




class SquareWithInclusion(mesh):

    def nodecoords(self):

        # mesh refinement level (number of divisions along specific mesh
        # segments)
        mrl = self.mrl

        # radial-direction mesh refinement level
        mrlr = mrl

        # theta-direction refinement level
        mrlt = 3*mrl

        # element order
        n = self.n

        # default meshing internal parameters
        # TODO allow these parameters to be overwritten with the values passed in
        # through options
        # TODO tune the default values to match what is being done in the GBMS
        # paper

        # outer square "radius"
        r_bs = 0.5

        # circular inclusion radius
        r_c = r_bs*0.5

        # small square "radius" (internal to inclusion)
        r_ss = r_c*0.5

        # circular inclusion distortion magnitude (0<=circ_dist<1)
        circ_dist = 0.5

        # number of "petals" or lobes in inclusion
        n_petals = 8
        theta_offset = 0*math.pi/8

        # x and y coordinates of small square
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

            # first string of points extends from small square internal to
            # inclusion out to inclusion boundary, and second string of points
            # extends from inclusion boundary out to unit cell boundary

            # x-coordinates for point string
            x_start = r_ss
            x_end = r_bs
            x_middle = r_c*np.cos(theta)
            x_vec1 = np.linspace(x_start, x_middle, n*mrlr+1)
            x_vec2 = np.linspace(x_middle, x_end, n*mrlr+1)
            x_vec = np.concatenate((x_vec1, x_vec2), axis=0)

            # y-coordinates for point string
            y_start = y_ss[-1, i]
            y_end = r_bs*np.tan(theta)
            y_middle = r_c*np.sin(theta)
            y_vec1 = np.linspace(y_start, y_middle, n*mrlr+1)
            y_vec2 = np.linspace(y_middle, y_end, n*mrlr+1)
            y_vec = np.concatenate((y_vec1, y_vec2), axis=0)

            # add point strings to array containing all points for an outer
            # quadrant
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
        emat_ss = []
        for i in range(0, mrlt):
            for j in range(0, mrlt):
                emat_ss.append(index_ss[(i*n):(i*n+n+1),
                                       (j*n):(j*n+n+1)].flatten('F'))

        # form element node index for one quadrant (each row containst the node
        # indices for an element)
        index_quad = np.reshape(
            np.arange(0, (n*mrlt+1)*(n*mrlr+1)),
            ((n*mrlr+1), (n*mrlt+1)))
        emat_quad = []
        for i in range(0, mrlr):
            for j in range(0, mrlt):
                emat_quad.append(index_quad[(i*n):(i*n+n+1),
                                            (j*n):(j*n+n+1)].flatten('F'))

        # concatenate element matrix indices from small square with those from the
        # quad segments
        #  emat = np.copy(emat_ss)
        emat = emat_ss
        for i in range(0, 2*4):
            # get maximum index in eleNodeIndex
            ematMax = max([np.max(sublist) for sublist in emat])
            emat.extend([[ind+ematMax+1 for ind in sublist] for sublist in emat_quad])

        # round coordinates to prepare for sorting and unique-ness test
        roundcoordinates = np.round(coordinates, 12)

        # note that at this point the nodes overlap at the edge of each "square" of
        # nodes. We need to find unique nodes and remap the element matrix indices
        # to refer to this new unique set of nodes
        if True:
            unique_indices, unique_inverse = np.unique(
                roundcoordinates,
                axis=0,
                return_index=True,
                return_inverse=True)[1:3]
        else:

            unique_indices, unique_inverse = unique_rows(
                roundcoordinates,
                return_index=True,
                return_inverse=True)[1:3]



        emat = [unique_inverse[i] for i in emat]
        coordinates = coordinates[unique_indices, :]

        def distortnodes(coordinates):

            # split up x and y coordinates
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

            # mesh distortion outside inclusion
            i_o = radii > (r_c-1e-10)
            i_i = np.invert(i_o)

            #  new inclusion radius
            r_cp = r_c*(1+theta_dist)

            #  inner inclusion distortion (use a square root node distribution)
            a = (r_c-r_cp)/(r_cp**2)
            b = 1
            c = -radii
            r_i = (-b + np.sqrt(b**2-4*a*c))/(2*a)
            r_i[a == 0] = radii[a == 0]

            #  outside distortion (use a linear node distribution)
            r_o = r_cp + (r_edge-r_cp)*(radii - r_c)/(r_edge-r_c)

            #  update radii values
            radii[i_o] = r_o[i_o]
            radii[i_i] = r_i[i_i]

            # update x, y coordinates
            x = radii*np.cos(thetas)
            y = radii*np.sin(thetas)

            return np.vstack((x, y)).transpose()

        # Distort nodes to form a more interesting inclusion shape
        coordinates = distortnodes(coordinates)
        materialIndex = np.concatenate(([0] * (mrlt**2), np.tile(
            np.concatenate(([0] * (mrlt * mrlr), [1] * (mrlr * mrlt)),
                           axis=0), 4)), axis=0)


        return coordinates, emat, materialIndex

def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
               return_index=return_index,
               return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

import matplotlib.pyplot as plt

def plotEdges(mesh):

    lineHandle = []
    for i in range(0, len(mesh.edges)):
        lineHandle.append(plt.plot(mesh.coordinates[mesh.edges[i], 0],
                                    mesh.coordinates[mesh.edges[i], 1],
                                    'k'))

    return lineHandle

def plotElements(mesh):

    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    coordinates = mesh.coordinates

    patches = []
    nEle = len(mesh.eleNodeIndex)
    patchindex = np.zeros(
        (nEle, np.shape(mesh.elements[0].perimeterIndex())[0]))

    # loop through elements and create patches and coloring info for patches
    for i in range(0, nEle):

        # x, y coordinates of current element
        eleCoords = np.transpose(
            np.vstack((mesh.elements[i].x, mesh.elements[i].y)))

        # element index of perimeter nodes
        iPerimeter = mesh.elements[i].perimeterIndex()
        patchCoords = eleCoords[iPerimeter, :]
        polygon = Polygon(patchCoords, True)
        patches.append(polygon)

        # create patch-index array
        patchindex[i, :] = mesh.eleNodeIndex[i][iPerimeter]

    fcolors = [(0.25, 0.25, 0.25), (0.8, 0.8, 0.8)]
    ecolors = [(0.20, 0.20, 0.20), (0.6, 0.6, 0.6)]
    flist = [
        fcolors[mesh.materialIndex[i]]
        for i in range(0, np.shape(mesh.materialIndex)[0])]
    #  flist = fcolors[mesh.materialIndex]
    elist = [
        ecolors[mesh.materialIndex[i]]
        for i in range(0, np.shape(mesh.materialIndex)[0])]
    #  elist = ecolocs[mesh.materialIndex]
    p = PatchCollection(patches, cmap=matplotlib.cm.jet,
                        edgecolors=elist,
                        facecolors=flist)

    #  ax.add_collection(p)
    ax = plt.gca()
    #  ax.cla()
    patchHandle=ax.add_collection(p)

    if False:
        for i in range(0, coordinates.shape[0]):
            plt.text(coordinates[i, 0], coordinates[i, 1], str(i))


    plt.axis('equal')
    #  plt.show()
    #  pass
    return patchHandle


def plotMesh(mesh):

    patchHandle = plotElements(mesh)
    lineHandle = plotEdges(mesh)

    return patchHandle, lineHandle

def KMAssemble(mesh, matlist):

    from scipy.sparse import lil_matrix

    dpn = 2
    n_DOF = int(dpn*np.shape(mesh.coordinates)[0])

    # preallocate mass and stiffness matrices
    K = lil_matrix((n_DOF, n_DOF))
    M = lil_matrix((n_DOF, n_DOF))

    # loop through elements
    nEle = len(mesh.elements)
    for ii in range(0, nEle):

        matind = mesh.materialIndex[ii]
        Ke, Me = mesh.elements[ii].KM(matlist[matind])

        nodeind = mesh.eleNodeIndex[ii]
        dofind = np.vstack((nodeind*2.0, nodeind*2.0+1))
        dofind = dofind.flatten('F')
        K[dofind, dofind[:, None]] = K[dofind, dofind[:, None]] + Ke
        M[dofind, dofind[:, None]] = M[dofind, dofind[:, None]] + Me
        #  M[dofind, dofind] = M[dofind, dofind] + Me

    return K, M
