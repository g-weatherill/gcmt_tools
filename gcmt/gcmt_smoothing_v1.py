#!/usr/bin/env/python

"""
Script to perform a focal mechanism smoothing. 
Kernel implementation is based on the Zechar (2010) isotropic Gaussian
approach to kernel smoothing - adapted for application to the whole Earth
"""
import numpy as np
from scipy.special import erf
from math import fabs, floor, log10, sqrt, atan2, pi, degrees, radians
import matplotlib.pyplot as plt
import gcmt_catalogue as gcat
from gcmt_ndk_parser import ParseNDKtoGCMT


GCMT_FILE = 'GCMTcatalogue_ndk_end2012.txt'
BANDWIDTH = 50.
LOWER_DEPTH = 40.
UPPER_DEPTH = 0.
TENSOR_FILE = 'SmoothedGCMT_Tensors_1p0.txt'
RATE_FILE = 'SmoothedGCMT_Rate_1p0.txt'
MESH_PARAMS = [-179.5, 179.5, -89.5, 89.5, 1.0]
#MESH_PARAMS = [-179.9, 179.9, -89.9, 89.9, 0.2]


def _get_length_scales(eqlon, eqlat, bandwidth, rad_earth=6371.0):
    '''
    Returns the bandwidth for the event in terms of decimal degrees
    '''
    xlen_scale = degrees(bandwidth / (rad_earth * np.cos(radians(eqlat))))
    ylen_scale = (bandwidth / (2. * pi * rad_earth)) * 360.
    return xlen_scale * sqrt(2.), ylen_scale * sqrt(2.)


def haversine(lons1, lats1, lons2, lats2, earth_rad=6371.0):
    '''
    '''
    dlat = np.radians(lats2 - lats1)
    dlon = np.radians(lons2 - lons1)
    # Check for dateline crossing
    idl = dlon > (pi / 2.)
    dlon[idl] = (2. * pi) - dlon[idl]
    lats1 = np.radians(lats1)
    lats2 = np.radians(lats2)
    aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lats1) * np.cos(lats2) * 
                                        (np.sin(dlon / 2.) ** 2.))
    return earth_rad * 2. * np.arctan2(np.sqrt(aval), np.sqrt(1. - aval))

def _get_distances(eqlon, eqlat, clons, clats, spcx, spcy=None):
    '''
    Returns the distances (xeq - xcell) and (yeq - ycell) taking into account
    the crossing of the International Dateline
    '''
    if not spcy:
        spcy = spcx
    lx = clons - (spcx / 2.)
    ux = clons + (spcx / 2.)
    # Get longitude length scales
    dx1 = eqlon - lx
    dx2 = eqlon - ux
    # Address cases of lower bound across dateline and earthquake is in
    # eastern hemisphere
    idx = np.logical_and(np.fabs(dx1) > 180., dx1 > 0.)
    #dx1[idx] = -(360. - dx1[idx]) 
    dx1[idx] = eqlon - (360 + lx[idx])
    dx2[idx] = eqlon - (360 + ux[idx])
    
    #dx2[idx] = dx1[idx] + spcx
    # Address cases of lower abound across dateline and earthquake is in 
    # western hemisphere
    idx = np.logical_and(np.fabs(dx1) > 180., dx1 < 0.)
    dx1[idx] = eqlon - (-360. + lx[idx]) 
    dx2[idx] = eqlon - (-360. + ux[idx]) 
    #dx2[idx] = dx1[idx] - spcx
    
    # For latitudes this is not a problem - just take the values
    dy1 = eqlat - (clats - (spcy / 2.))
    dy2 = eqlat - (clats + (spcy / 2.))
    return dx1, dx2, dy1, dy2



class GCMTIsotropicGaussian(object):
    """
    """
    def __init__(self, mesh_params):
        """
        """
        self.spc = mesh_params[-1]
        self.xgrid = np.arange(mesh_params[0], mesh_params[1] + 1E-7, 
                               mesh_params[4])
        self.ygrid = np.arange(mesh_params[2], mesh_params[3] + 1E-7,
                               mesh_params[4])


        (gridx, gridy) = np.meshgrid(self.xgrid, self.ygrid)
        (self.nlat, self.nlon) = np.shape(gridx)
        self.npts = self.nlat * self.nlon
        self.grid = np.column_stack([np.reshape(gridx, self.npts, 1),
                                     np.reshape(gridy, self.npts, 1)])

        self.smoothed_rate = None
        self.smoothed_tensor = None
        self.catalogue = None 
        self.smoothing_distance = None
        self.smoothing_events = None


    def apply_smoothing(self, catalogue, bandwidth, upper_depth=0., 
                        lower_depth=np.inf, use_centroids=True):
        """
        """
        assert isinstance(catalogue, gcat.GCMTCatalogue)
        self.catalogue = catalogue
        self.number_earthquakes = self.catalogue.number_events()
        self.smoothed_rate = np.zeros(self.npts, dtype=float)
        self.smoothing_distance = np.zeros(self.npts, dtype=float)
        self.smoothing_events = np.zeros(self.npts, dtype=int)
        self._initialise_smoothed_moment_tensors()
        print 'Smoothing Catalogue ...'
        for iloc, gcmt in enumerate(self.catalogue.gcmts):
            if iloc % 100 == 0:
                print '%s of %s' %(iloc, self.number_earthquakes)
            
            if use_centroids:
                if (gcmt.centroid.depth < upper_depth) or \
                   (gcmt.centroid.depth > lower_depth):
                    continue
        
                longitude = gcmt.centroid.longitude
                latitude = gcmt.centroid.latitude
            else:
                if (gcmt.hypocentre.depth < upper_depth) or \
                    (gcmt.hypocentre.depth > lower_depth):
                    continue
                longitude = gcmt.hypocentre.longitude
                latitude = gcmt.hypocentre.latitude
            kernel, idx = self.get_gaussian_kernel(longitude,
                                                   latitude,
                                                   bandwidth,
                                                   self.grid[:, 0],
                                                   self.grid[:, 1],
                                                   self.spc)
            self.smoothed_rate[idx] = self.smoothed_rate[idx] + kernel
            self._update_smoothed_tensor(gcmt, kernel, idx)
            self._get_mean_distance(longitude, latitude, idx)
            self.smoothing_events[idx] += 1
        print 'done!'
        validx = self.smoothing_events > 0
        self.smoothing_distance[validx] = self.smoothing_distance[validx] /\
            self.smoothing_events[validx].astype(float)

    def _initialise_smoothed_moment_tensors(self):
        """
        """
        # Create list of instances of the GCMTMomentTensor Class
        self.smoothed_tensor = [gcat.GCMTMomentTensor() 
                                for i in range(0, self.npts)]
        # Initialise all points with null tensor
        for mom_tensor in self.smoothed_tensor:
            mom_tensor.tensor = np.zeros([3, 3], dtype=float)
            mom_tensor.tensor_sigma = np.zeros([3, 3], dtype=float)
        
    def _update_smoothed_tensor(self, gcmt, kernel, idx):
        """
        Updates the smoothed tensors with the weighted tensor 
        """
        for iloc, locn in enumerate(idx):
            self.smoothed_tensor[locn].tensor += gcmt.moment_tensor.tensor *\
                kernel[iloc]

    def _get_mean_distance(self, longitude, latitude, idx):
        """
        Finds the weighted mean distance of the events used to smooth
        at a site
        """ 
        distances = haversine(longitude, latitude, self.grid[idx, 0],
                                  self.grid[idx, 1])
        self.smoothing_distance[idx] += distances

    def get_gaussian_kernel(self, cmtlong, cmtlat, bandwidth, celllong, 
                            celllat, spcx, spcy=None):
        """
        Return the contribution of moment tensor [cmtlong, cmtlat] to cell
        specified by midpoint [midcell] with widths [x, y]. Gaussian kernel
        definition from Zechar et al (2010).
        """
        # Get the length scales (in degrees})
        xls, yls = _get_length_scales(cmtlong, cmtlat, bandwidth)
        # Get the distances (in degrees)
        dx1, dx2, dy1, dy2 = _get_distances(cmtlong, cmtlat, celllong, celllat,
                                            spcx, spcy)

        # Find earthquakes in distance
        valid_long = np.logical_and(dx1 <= 5.92 * xls, dx2 >= -5.92 * xls)
        valid_lat = np.logical_and(dy1 <= 5.92 * yls, dy2 >= -5.92 * yls)
        select = np.where(np.logical_and(valid_long, valid_lat))[0]
        kernel = 0.25 * (erf(dx2[select] / xls) - erf(dx1[select] / xls)) *\
            (erf(dy2[select] / yls) - erf(dy1[select] / yls))
        
        if np.any(kernel < 0.):
            print cmtlong, cmtlat, celllong[select], celllat[select], \
                np.column_stack([dx1[select], dx2[select], dy1[select],
                                 dy2[select]]), kernel
            plt.plot(cmtlong, cmtlat, 's')
            plt.plot(celllong[select], celllat[select], '.')
            breaker = here
        return kernel, select


    def plot_smoothing(self, plot_log10=False):
        '''
        '''
        output_rate = np.reshape(self.smoothed_rate, [self.nlon, self.nlat]).T
        if plot_log10:
            output_rate = np.log10(output_rate)
        plt.imshow(np.flipud(output_rate), 
                   extent=(np.min(self.xgrid), np.max(self.xgrid), 
                           np.min(self.ygrid), np.max(self.ygrid)))

    def export_smoothed_rate_to_csv(self, output_file):
        """
        Exports the grid longitude, latitude, smoothed rate and smoothing 
        distance to a simple csv file
        """
        fid = open(output_file, 'w')
        print >> fid, 'Longitude, Latitude, Smoothed Rate, Mean Distance'
        for i in range(0, self.npts):
            print >> fid, '%8.3f, %8.3f, %.8e, %12.4f' % (
                self.grid[i, 0], 
                self.grid[i, 1],
                self.smoothed_rate[i],
                self.smoothing_distance[i])
        fid.close()
            

    def export_smoothed_tensors_to_gmt(self, gmtfile):
        """
        Exports the smoothed tensor set to GMT format
        N.B. In GMT psmeca use the -Sc option!!!
        """
        fid = open(gmtfile, 'w')
        for iloc in range(0, self.npts):
            if self.smoothing_events[iloc] == 0:
                print >> fid, ("%9.4f" * 2 + "%6.1f" + "%6.0f" * 6 + "%6.2f" + 
                    " %d" + "%10.4f" * 2) %(self.grid[iloc, 0], 
                                            self.grid[iloc, 1], 
                                            0., 
                                            0.0,
                                            90.0,
                                            0.,
                                            90.,
                                            90.,
                                            0.,
                                            5.0,
                                            23,
                                            self.grid[iloc, 0],
                                            self.grid[iloc, 1])
            else:
                # Get the nodal planes from the new tensor
                nps = self.smoothed_tensor[iloc].get_nodal_planes()
                print >> fid, ("%9.4f" * 2 + "%6.1f" + "%6.0f" * 6 + "%6.2f" + 
                    " %d" + "%10.4f" * 2) %(self.grid[iloc, 0], 
                                            self.grid[iloc, 1], 
                                            0., 
                                            nps.nodal_plane_1['strike'],
                                            nps.nodal_plane_1['dip'],
                                            nps.nodal_plane_1['rake'],
                                            nps.nodal_plane_2['strike'],
                                            nps.nodal_plane_2['dip'],
                                            nps.nodal_plane_2['rake'],
                                            5.0,
                                            23,
                                            self.grid[iloc, 0],
                                            self.grid[iloc, 1])
        fid.close()
    


if __name__ == '__main__':
    # Load catalogue
    print "Loading GCMT Catalogue ..."
    parser = ParseNDKtoGCMT(GCMT_FILE)
    catalogue = parser.read_file()
    print "Setting up smoothing algorithm"
    # Greate class
    smooth = GCMTIsotropicGaussian(MESH_PARAMS)
    # Apply smoothing
    smooth.apply_smoothing(catalogue, 
                           50., 
                           upper_depth=UPPER_DEPTH,
                           lower_depth=LOWER_DEPTH)
    # Exporting to GMT format
    print "Exporting tensors to GMT format"
    smooth.export_smoothed_tensors_to_gmt(TENSOR_FILE)
    print "Exporting rates to csv"
    smooth.export_smoothed_rate_to_csv(RATE_FILE)



#
#def bearing(lons1, lats1, lons2, lats2):
#    '''
#    '''
#    lons1 = np.radians(lons1)
#    lats1 = np.radians(lats1)
#    lons2 = np.radians(lons2)
#    lats2 = np.radians(lats1)
#    cos_lat2 = np.cos(lats2)
#    true_course = np.arctan2(
#        np.sin(lons1 - lons2) * cos_lat2,
#        np.cos(lats1) * np.sin(lats2)
#        - np.sin(lats1) * cos_lat2 * np.cos(lons1 - lons2))
#    return (360. - np.degrees(true_course)) % 360.
#
#
#def xy_distances(lons, lats, clon, clat):
#    npts = len(lons)
#    lx = lons - (spc / 2.)
#    ux = lons + (spc / 2.)
#    ly = lats - (spc / 2.)
#    uy = lats + (spc / 2.)
#    sign = np.ones(npts)
#
#    dlonl = clon - lx
#    dlonu = clon - ux
#    # For longitudes if clon > lx and dlonl < 180. (i.e. does not cross idl)
#
#    sign[idx] = -sign[idx] # Values should be positive
#    dx1 =haversine(clon, clat, lx, clat)
#    # Where clon > lx but dlonl > 180. (has crossed dateline so make negative) 
#    idx = np.logical_and( dlonl > 0., dlonl < 180.)
#    
#    
#    # For longitudes if clon > ux and dlonl < 180. (i.e. does not cross idl)
#    sign = -np.ones(npts)
#    idx = np.logical_and(dlonu > 0., dlonu < 180.)
#    sign[idx] = -sign[idx] # Values should be positive
#    dx2 = sign * haversine(clon, clat, ux, clat)
#
#    # For latitudes just take the actual distance
#    dy1 = haversine(clon, clat, clon, ly)
#    idx = clat < ly
#    dy1[idx] = -dy1[idx]
#
#    dy2 = haversine(clon, clat, clon, uy)
#    idx = clat < uy
#    dy2[idx] = -dy2[idx]
#    return 





#    if not spcy:
#        spcy = spcx
#    x1 = celllong - spcx / 2.
#    x2 = celllong + spcx / 2.
#    y1 = celllat - spcy / 2.
#    y2 = celllat + spcy / 2.
#    # Pre-allocate kernel
#    kernel = np.zeros(np.shape(celllong)[0], dtype=float)
#    dx1 = -distance_to_arc(cmtlong, cmtlat, 0., x1, y1)
#    dx2 = -distance_to_arc(cmtlong, cmtlat, 0., x2, y2)
#    dy1 = -distance_to_arc(cmtlong, cmtlat, 90., x1, y1)
#    dy2 = -distance_to_arc(cmtlong, cmtlat, 90., x2, y2)
#    print np.max(dx1), np.min(dx1), np.max(dx2), np.min(dx2) 
#    # Select only events close enough to make a significant contribution
#    val_long = np.logical_or(np.fabs(dx1 <= 5.92 * bandwidth * sqrt(2.)), 
#                              np.fabs(dx2 <= 5.92 * bandwidth * sqrt(2.)))
#    
#    val_lat = np.logical_or(np.fabs(dy1 <= 5.92 * bandwidth * sqrt(2.)), 
#                             np.fabs(dy2 <= 5.92 * bandwidth * sqrt(2.)))
#    sel = np.where(np.logical_and(val_long, val_lat))[0]
#    print celllong[sel], celllat[sel] 
#    print np.max(dx1[sel]), np.min(dx1[sel]), np.max(dx2[sel]), np.min(dx2[sel]) 
#    # Function to normalise the distance by the bandwidth
#    adj = lambda dval, bandwidth: dval / (bandwidth * sqrt(2.)) 
#    # Kernel smoothing
#    kernel = 0.25 * (erf(adj(dx2[sel], bandwidth)) - 
#                     erf(adj(dx1[sel], bandwidth))) * \
#                    (erf(adj(dy2[sel], bandwidth)) - 
#                     erf(adj(dy1[sel],  bandwidth)))
#    return kernel, sel










                                          


