import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


def _arc_length(points):
    dif = np.diff(points, axis=0)
    ds = np.linalg.norm(dif, axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


class FrenetFrame:
    """
    Frenet frame along a centerline.

    Zoidberg coordinates:
        R → normal displacement
        Z → binormal displacement
        phi → centerline coordinate
    """

    def __init__(self, centerline_points):

        pts = np.asarray(centerline_points)

        self.points = pts
        self.s = _arc_length(pts)

        self.sx = CubicSpline(self.s, pts[:,0], bc_type="periodic")
        self.sy = CubicSpline(self.s, pts[:,1], bc_type="periodic")
        self.sz = CubicSpline(self.s, pts[:,2], bc_type="periodic")

        self.tree = cKDTree(pts)

    def frame(self, phi):

        sx,sy,sz = self.sx,self.sy,self.sz

        r  = np.array([sx(phi), sy(phi), sz(phi)])
        r1 = np.array([sx(phi,1), sy(phi,1), sz(phi,1)])
        r2 = np.array([sx(phi,2), sy(phi,2), sz(phi,2)])
        r3 = np.array([sx(phi,3), sy(phi,3), sz(phi,3)])

        T = r1/(np.linalg.norm(r1)+1e-16)

        cross12 = np.cross(r1,r2)

        r2_par = np.dot(r2,T)*T
        N_raw = r2-r2_par

        nrm = np.linalg.norm(N_raw)

        if nrm < 1e-12:
            zaxis = np.array([0,0,1])
            N = np.cross(T,np.cross(zaxis,T))
            N /= np.linalg.norm(N)
        else:
            N = N_raw/nrm

        B = np.cross(T,N)

        return r,T,N,B

    def xyz_from_RZphi(self,R,Z,phi):

        r,T,N,B = self.frame(phi)

        return r + R*N + Z*B

    def project(self,X,Y,Z):

        p = np.array([X,Y,Z])

        _,idx = self.tree.query(p)

        phi = self.s[idx]

        r,T,N,B = self.frame(phi)

        dv = p-r

        R = np.dot(dv,N)
        Z = np.dot(dv,B)

        return R,Z,phi
