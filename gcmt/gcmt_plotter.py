#!/usr/bin/env/python

'''
Test scripts for plotting GCMT information
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import Beach
from gcmt_catalogue import GCMTCatalogue
from gcmt_ndk_parser import ParseNDKtoGCMT

class GCMTCataloguePlotter(object):
    '''
    Set of plotting functions for a catalogue of GCMTs
    '''
    def __init__(self, catalogue):
        '''
        Instantiate from a GCMT catalogue
        '''
        self.catalogue = catalogue

    def plot_global_gcmt_map(self, n_samples=None, mid_long=None, 
        mid_lat=None):
        '''
        Creates a global map of Moment Tensors
        '''
        # If n_samples then only sample n_samples events
        
        idx = np.arange(self.catalogue.number_events())
        if n_samples:
            np.random.shuffle(idx)
            idx = idx[:n_samples]

        if not mid_long and not mid_lat:
            mid_long = 0.
            mid_lat = 
        

    def _create_global_basemap(self, mid_long, mid_lat, proj='robin'):
        '''
        Creates a basemap
        '''
        m = Basemap(projection=proj, lon_0=mid_long, resolution='l')
        m.drawcoastlines()
        m.fillcontinents(color='mistyrose', lake_color='lightskyblue')
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 
        

