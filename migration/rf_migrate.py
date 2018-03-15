#!/bin/env python
"""
Description:
    Example template python script structure.
    .......
    .......
   
References:
 
CreationDate:   3/15/18
Developer:      rakib.hassan@ga.gov.au
 
Revision History:
    LastUpdate:     3/15/18   RH
    LastUpdate:     dd/mm/yyyy  Who     Optional description
"""

import os
import pkg_resources
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from obspy import read_inventory, read_events, UTCDateTime as UTC
from obspy.clients.fdsn import Client
from rf import read_rf, RFStream
from rf import get_profile_boxes, iter_event_data, IterMultipleComponents
from rf.util import _add_processing_info, direct_geodetic

from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from mpi4py import MPI
import logging
log = logging.getLogger('migration')


# define utility functions
def rtp2xyz(r, theta, phi):
    xout = np.zeros((r.shape[0], 3))
    rst = r * np.sin(theta);
    xout[:, 0] = rst * np.cos(phi)
    xout[:, 1] = rst * np.sin(phi)
    xout[:, 2] = r * np.cos(theta)
    return xout
# end func

def xyz2rtp(x, y, z):
    rout = np.zeros((x.shape[0], 3))
    tmp1 = x * x + y * y
    tmp2 = tmp1 + z * z
    rout[0] = np.sqrt(tmp2)
    rout[1] = np.arctan2(sqrt(tmp1), z)
    rout[2] = np.arctan2(y, x)
    return rout
# end func

class Geometry:
    def __init__(self, start_lat_lon, azimuth, lengthkm, nx, widthkm, ny, depthkm, nz, debug=False):
        self._start_lat_lon = np.array(start_lat_lon)
        assert self._start_lat_lon.shape == (2,), 'start lat-lon should be a list of length 2'

        self._azimuth = azimuth
        self._length = lengthkm
        self._width = widthkm
        self._depth = depthkm
        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._ortho = (self._azimuth + 90) % 360 # orthogonal to azimuth
        self._earth_radius = 6371 #km
        self._debug = debug
        # Generate four sets of grids:
        # 1. Lon-Lat-Depth grid, with slowest to fastest index in that order
        # 2. Cartesian axis-aligned x-y-z grid, with slowest to fastest index in
        #    that order, starting from start_lat_lon. Note that this is a local
        #    coordinate system that does not account for spherical curvature and
        #    used for plotting purposes alone
        # 3. Spherical grid in Cartesian coordinates that accounts for spherical
        #    curvature and is used internally for all nearest neighbour calculations
        # 4. Cell-centre coordinates for 3
        self._glon, self._glat, self._gz, self._gxaa, self._gyaa, self._gzaa, \
            self._gxs, self._gys, self._gzs, self._gxsc, self._gysc, self._gzsc = self.generateGrids()

        # Compute centres of depth-node pairs
        self._gzaac = (self._gzaa[0,0,1:] + self._gzaa[0,0,:-1])/2.
    # end func

    def generateGrids(self):
        # Start mesh generation==============================================
        sll = self._start_lat_lon

        result = []
        resultCart = []
        dx = self._length / float(self._nx - 1)
        dy = self._width / float(self._ny - 1)
        dz = self._depth / float(self._nz - 1)

        runLengthll = sll
        runWidthll = sll
        cx = cy = cz = 0
        for ix in range(self._nx):
            runWidthll = runLengthll
            for iy in range(self._ny):
                for iz in range(self._nz):
                    result.append([runWidthll[1], runWidthll[0], iz * dz])
                    resultCart.append([ix * dx, iy * dy, iz * dz])
                # end for
                runWidthll = direct_geodetic(runLengthll, self._ortho, iy * dy)
            # end for
            runLengthll = direct_geodetic(runLengthll, self._azimuth, dx)
        # end for
        result = np.array(result).reshape(self._nx, self._ny, self._nz, 3)
        resultCart = np.array(resultCart).reshape(self._nx, self._ny, self._nz, 3)

        glon = result[:, :, :, 0].reshape(self._nx, self._ny, self._nz)
        glat = result[:, :, :, 1].reshape(self._nx, self._ny, self._nz)

        # Create local cartesian axis-aligned grids
        # Naming convention (Grid [XYZ] Axis-Aligned)
        gxaa = resultCart[:, :, :, 0].reshape(self._nx, self._ny, self._nz)
        gyaa = resultCart[:, :, :, 1].reshape(self._nx, self._ny, self._nz)
        gzaa = resultCart[:, :, :, 2].reshape(self._nx, self._ny, self._nz)

        # Create cartesian mesh with spherical curvature
        # Naming convention (Grid [XYZ] Spherical)
        ts = (90 - glat.flatten()) / 180. * np.pi
        ps = glon.flatten() / 180. * np.pi
        rs = (self._earth_radius - gzaa.flatten()) * np.ones(ts.shape)
        rtps = np.array([rs, ts, ps]).T
        xyzs = rtp2xyz(rtps[:, 0], rtps[:, 1], rtps[:, 2])
        xyzs = np.array(xyzs).reshape(self._nx, self._ny, self._nz, 3)
        gxs = xyzs[:, :, :, 0].reshape(self._nx, self._ny, self._nz)
        gys = xyzs[:, :, :, 1].reshape(self._nx, self._ny, self._nz)
        gzs = xyzs[:, :, :, 2].reshape(self._nx, self._ny, self._nz)

        if(self._debug):
            print np.min(gxs.flatten()), np.max(gxs.flatten())
            print np.min(gys.flatten()), np.max(gys.flatten())
            print np.min(gzs.flatten()), np.max(gzs.flatten())

        # Compute cell-centre coordinates
        # Naming convention (Grid [XYZ] Spherical Centre)
        gxsc = (gxs[:-1, :-1, :-1] + gxs[1:, 1:, 1:]) / 2.
        gysc = (gys[:-1, :-1, :-1] + gys[1:, 1:, 1:]) / 2.
        gzsc = (gzs[:-1, :-1, :-1] + gzs[1:, 1:, 1:]) / 2.

        if(self._debug):
            print '\n'
            print np.min(gxsc.flatten()), np.max(gxsc.flatten())
            print np.min(gysc.flatten()), np.max(gysc.flatten())
            print np.min(gzsc.flatten()), np.max(gzsc.flatten())

        return glon, glat, gzaa, gxaa, gyaa, gzaa, gxs, gys, gzs, gxsc, gysc, gzsc
    # end func
# end class

class Migrate:
    def __init__(self, geometry, stream, velocity_model='iasp91.dat', debug=False):
        assert isinstance(geometry, Geometry), 'Must be an instance of class Geometry..'
        self._geometry = geometry

        assert isinstance(stream, RFStream), 'Must be an instance of class RFStream..'
        self._stream = stream
        if(velocity_model != 'iasp91.dat'): assert 0, 'Only iasp91.dat is currently supported'
        self._velocity_model = velocity_model

        self._debug = debug

        # Initialize MPI
        self._comm = MPI.COMM_WORLD
        self._nproc = self._comm.Get_size()
        self._chunk_index = self._comm.Get_rank()

        self._ppDict = defaultdict(list) # dictionary for piercing point results
        self._proc_zs = defaultdict(list) # depth values that each process works on
        self._treeDict = {} # dictionary for Kd-trees for each depth layer
        self._d2tIO = None

        # Create depth-to-time interpolation object
        rp = 'rf'
        rpf = '/'.join(('data', 'iasp91.dat')) # find where iasp91.dat is located
        fp = pkg_resources.resource_stream(rp, rpf)
        fn = fp.name
        fp.close()

        m = np.loadtxt(fn)
        dlim = m[:, 0] < 2800 # don't need data past 2800 km

        depths = m[:, 0][dlim] # depths in km
        s = 1. / m[:, 2][dlim] # slowness in s/km
        # Integrate slowness with respect to distance to get travel times.
        # TODO: should we use Dix's interval velocity?
        times = np.cumsum(np.diff(depths) * (s[0:-1] + s[1:]) / 2.)
        times = np.insert(times, 0, 0)
        self._d2tIO = interp1d(depths, times)

        # split workload
        self.__split_work()
    # end func

    def __split_work(self):
        """
        Splits up workload over n processors
        """

        if (self._chunk_index == 0):
            count = 0
            for iproc in np.arange(self._nproc):
                for iz in np.arange(np.divide(self._geometry._gzaac.shape[0], self._nproc)):
                    z = self._geometry._gzaac[count]
                    self._proc_zs[iproc].append(z)
                    count += 1
            # end for

            for iproc in np.arange(np.mod(self._geometry._gzaac.shape[0], self._nproc)):
                z = self._geometry._gzaac[count]
                self._proc_zs[iproc].append(z)
                count += 1
        # end if

        # broadcast workload to all procs
        self._proc_zs = self._comm.bcast(self._proc_zs, root=0)
        if (self._chunk_index == 0): log.info(' Distributing workload over %d processors..' % (self._nproc))

        if(self._debug):
            print 'proc: %d, %d depth values\n========='%(self._chunk_index,
                                                   len(self._proc_zs[self._chunk_index]))
            for z in self._proc_zs[self._chunk_index]: print z
        # end if
    # end func

    def __generatePiercingPoints(self):
        for z in self._proc_zs[self._chunk_index]:
            ppoints = self._stream.ppoints(z)
            self._ppDict[z] = ppoints
            #print z, len(ppoints)
        # end for

        # Gather all results on proc 0
        ppDictList = self._comm.gather(self._ppDict, root=0)

        if(self._chunk_index==0):
            self._ppDict = defaultdict(list)
            for ip in np.arange(self._nproc):
                for k in ppDictList[ip].keys():
                    self._ppDict[k] = ppDictList[ip][k]
                # end for
            # end for
            #print(len(self._ppDict))

            # Create Kd-trees for each depth value
            for k in self._ppDict.keys():
                ts = (90 - self._ppDict[k][:, 0]) / 180. * np.pi
                ps = self._ppDict[k][:, 1] / 180. * np.pi
                rs = (self._geometry._earth_radius - k) * np.ones(ts.shape)
                rtps = np.array([rs, ts, ps]).T

                xyzs = rtp2xyz(rtps[:, 0], rtps[:, 1], rtps[:, 2])
                self._treeDict[k] = cKDTree(xyzs)
            # end for
        # end if

        # broadcast Kd-tree dictionary to all procs
        self._treeDict = self._comm.bcast(self._treeDict, root=0)
    # end func

    def execute(self):
        if(self._chunk_index==0): log.info(' Generating Piercing Points..')
        self.__generatePiercingPoints()

        if (self._chunk_index == 0): log.info(' Stacking amplitudes..')

        vol = np.zeros(self._geometry._gxsc.shape)
        volHits = np.zeros(self._geometry._gxsc.shape)
        empty = np.zeros(self._geometry._gxsc.shape)
        times = self._stream[0].times() - 25
        for ix in range(self._geometry._nx - 1):
            for iy in range(self._geometry._ny - 1):
                for z in self._proc_zs[self._chunk_index]:
                    t = treeDict[z]

                    ids = t.query_ball_point([gxc[ix, iy, iz],
                                              gyc[ix, iy, iz],
                                              gzc[ix, iy, iz]], r=20, n_jobs=6)
                    if (len(ids) == 0):
                        numEmpty += 1
                        continue
                    # end if

                    ct = d2tIO(z)
                    for i in ids:
                        tidx = np.argmin(np.fabs(times - ct))
                        # print tidx*(1./stream[i].stats.sampling_rate)-25, d2tIO(z)
                        vol[ix, iy, iz] += stream[i].data[tidx]
                        volHits[ix, iy, iz] += 1.
                        # end for
                        # end for
            # end for
            print
            ix
        # end for

    # end func
# end class

def main():
    """
    define main function
    :return:
    """

    rffile = '/home/rakib/work/pst/rf/notebooks/7X-rf_profile_rfs-cleaned.h5'
    s = read_rf(rffile, 'H5')

    g = Geometry(start_lat_lon=(-18.75, 138.15), azimuth=80,
                 lengthkm=450, nx=45, widthkm=350, ny=35, depthkm=100, nz=50)
    m = Migrate(geometry=g, stream=s)

    m.execute()

    return
# end


# =============================================
# Quick test
# =============================================
if __name__ == "__main__":
    # call main function
    main()