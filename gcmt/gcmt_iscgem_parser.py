#!/usr/bin/env/python

'''
Parser of the GEM-ISC catalogue into GCMT classes
'''
import csv
import datetime
import numpy as np
import gcmt_utils as utils
from math import floor
from gcmt_catalogue import (GCMTHypocentre, GCMTCentroid, 
                            GCMTPrincipalAxes, GCMTNodalPlanes,
                            GCMTMomentTensor, GCMTEvent, GCMTCatalogue)
from isf_catalogue import Magnitude, Location, Origin, Event, ISFCatalogue


class ISCGEMCatalogue(object):
    """
    Class to parse the ISC GEM file to a complete GCMT catalogue class
    """
    FLOAT_ATTRIBUTE_LIST = ['second', 'timeError', 'longitude', 'latitude', 
                        'SemiMajor90', 'SemiMinor90', 'ErrorStrike', 'depth',
                        'depthError', 'magnitude', 'sigmaMagnitude', 'moment', 
                        'mpp', 'mpr', 'mrr', 'mrt', 'mtp', 'mtt']

    INT_ATTRIBUTE_LIST = ['eventID','year', 'month', 'day', 'hour', 'minute',
                          'flag', 'scaling']

    STRING_ATTRIBUTE_LIST = ['Agency', 'magnitudeType','comment', 'source']
    
    TOTAL_ATTRIBUTE_LIST = list(
        (set(FLOAT_ATTRIBUTE_LIST).union(
            set(INT_ATTRIBUTE_LIST))).union(
                 set(STRING_ATTRIBUTE_LIST)))

    def __init__(self):
        """
        Initialise the catalogue dictionary
        """
        self.data = {}

        for attribute in self.TOTAL_ATTRIBUTE_LIST:
            if attribute in self.FLOAT_ATTRIBUTE_LIST:
                self.data[attribute] = np.array([], dtype=float)
            elif attribute in self.INT_ATTRIBUTE_LIST:
                self.data[attribute] = np.array([], dtype=int)
            else:
                self.data[attribute] = []

        self.number_earthquakes = 0
        self.gcmt_catalogue = GCMTCatalogue()
    
    def get_number_events(self):
        return len(self.data[self.data.keys()[0]])


    def write_to_gcmt_class(self):
        """
        Writes the ISC-GEM catalogue into a GCMT catalogue format
        """
        for iloc in range(0, self.get_number_events()):
            print iloc
            gcmt = GCMTEvent()
            gcmt.identifier = self.data['eventID'][iloc]
            gcmt.magnitude = self.data['magnitude'][iloc]
            # Get moment plus scaling
            if not np.isnan(self.data['moment'][iloc]):
                scaling = float(self.data['scaling'][iloc])
                gcmt.moment = self.data['moment'][iloc] * (10. ** scaling)
            gcmt.metadata = {'Agency': self.data['Agency'][iloc],
                             'source': self.data['source'][iloc]}

            # Get the hypocentre
            gcmt.hypocentre = GCMTHypocentre()
            gcmt.hypocentre.source = self.data['source'][iloc]
            gcmt.hypocentre.date = datetime.date(self.data['year'][iloc],
                                                 self.data['month'][iloc],
                                                 self.data['day'][iloc])
            second = self.data['second'][iloc]
            microseconds = int((second - floor(second)) * 1000000)

            gcmt.hypocentre.time = datetime.time(self.data['hour'][iloc],
                                                 self.data['minute'][iloc],
                                                 int(floor(second)), 
                                                 microseconds)
            gcmt.hypocentre.longitude = self.data['longitude'][iloc]
            gcmt.hypocentre.latitude = self.data['latitude'][iloc]

            setattr(gcmt.hypocentre, 
                    'semi_major_90', 
                    self.data['SemiMajor90'][iloc])

            setattr(gcmt.hypocentre, 
                    'semi_minor_90', 
                    self.data['SemiMinor90'][iloc])

            setattr(gcmt.hypocentre, 
                    'error_strike', 
                    self.data['ErrorStrike'][iloc])

            # Get the centroid - basically just copying across the hypocentre
            gcmt.centroid = GCMTCentroid(gcmt.hypocentre.date,
                                         gcmt.hypocentre.time)
            gcmt.centroid.longitude = gcmt.hypocentre.longitude
            gcmt.centroid.latitude = gcmt.hypocentre.latitude
            gcmt.centroid.depth = gcmt.hypocentre.depth
            gcmt.centroid.depth_error = self.data['depthError'][iloc]

            if self._check_moment_tensor_components(iloc):
                # Import tensor components
                gcmt.moment_tensor = GCMTMomentTensor()
                # Check moment tensor has all the components!
                gcmt.moment_tensor.tensor = utils.COORD_SYSTEM['USE'](
                    self.data['mrr'][iloc],
                    self.data['mtt'][iloc],
                    self.data['mpp'][iloc],
                    self.data['mrt'][iloc],
                    self.data['mpr'][iloc],
                    self.data['mtp'][iloc])
                gcmt.moment_tensor.tensor_sigma = np.array([[0., 0., 0.],
                                                            [0., 0., 0.],
                                                            [0., 0., 0.]])
                #print gcmt.moment_tensor.tensor
                # Get nodal planes
                gcmt.nodal_planes = gcmt.moment_tensor.get_nodal_planes()
                gcmt.principal_axes = gcmt.moment_tensor.get_principal_axes()

                # Done - append to catalogue
                self.gcmt_catalogue.gcmts.append(gcmt)

        return self.gcmt_catalogue

    def write_to_isf_catalogue(self):
        """
        
        """
        isf_cat = ISFCatalogue('ISC-GEM', 'ISC-GEM')
        for iloc in range(0, self.get_number_events()):
            # Origin ID
            event_id = self.data['eventID'][iloc]
            origin_id = event_id
            # Create Magnitude
            mag = [Magnitude(origin_id, 
                            self.data['magnitude'][iloc], 
                            'ISC-GEM', 
                            scale='Mw', 
                            sigma=self.data['sigmaMagnitude'][iloc])]
            # Create Moment
            if not np.isnan(self.data['moment'][iloc]):
                moment = self.data['moment'][iloc] *\
                    (10. ** self.data['scaling'][iloc]) 
                mag.append(Magnitude(origin_id, moment, 'ISC-GEM', scale='Mo'))

            # Create Location
            semimajor90 = self.data['SemiMajor90'][iloc]
            semiminor90 = self.data['SemiMinor90'][iloc]
            error_strike = self.data['ErrorStrike'][iloc]
            if np.isnan(semimajor90):
                semimajor90 = None
            if np.isnan(semiminor90):
                semiminor90 = None
            if np.isnan(error_strike):
                error_strike = None
            depth_error = self.data['depthError'][iloc]
            if np.isnan(depth_error):
                depth_error = None
            locn = Location(origin_id,
                            self.data['longitude'][iloc],
                            self.data['latitude'][iloc],
                            self.data['depth'][iloc],
                            semimajor90,
                            semiminor90,
                            error_strike,
                            depth_error)
            

            # Create Origin
            # Date
            eq_date = datetime.date(self.data['year'][iloc],
                                    self.data['month'][iloc],
                                    self.data['day'][iloc])
            # Time
            secs = self.data['second'][iloc]
            
            microsecs = int((secs - floor(secs)) * 1E6)
            eq_time = datetime.time(self.data['hour'][iloc],
                                    self.data['minute'][iloc],
                                    int(secs),
                                    microsecs)
            origin = Origin(origin_id, eq_date, eq_time, locn, 'ISC-GEM', 
                            is_prime=True)
            origin.magnitudes = mag
            event = Event(event_id, [origin], origin.magnitudes)
            if self._check_moment_tensor_components(iloc):
                # If a moment tensor is found then add it to the event
                moment_tensor = GCMTMomentTensor()
                scaling = 10. ** self.data['scaling'][iloc]
                moment_tensor.tensor = scaling * utils.COORD_SYSTEM['USE'](
                    self.data['mrr'][iloc],
                    self.data['mtt'][iloc],
                    self.data['mpp'][iloc],
                    self.data['mrt'][iloc],
                    self.data['mpr'][iloc],
                    self.data['mtp'][iloc])
                moment_tensor.exponent = self.data['scaling'][iloc]
                setattr(event, 'tensor', moment_tensor)
            isf_cat.events.append(event)
        return isf_cat

    def _check_moment_tensor_components(self, iloc):
        '''
  
        '''
        for component in ['mrr', 'mtt', 'mpp', 'mrt', 'mpr', 'mtp']:
            if np.isnan(self.data[component][iloc]):
                return False
        return True
