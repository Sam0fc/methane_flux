import numpy as np 
import pandas as pd
from geopy.distance import geodesic 
import constants as constants

DENOMINATOR_ZERO_THRESHOLD = 1e-10
ZERO_THRESHOLD_KM = 0.001
KILOMETRE = 1000.0
## Theory from https://gaftp.epa.gov/Air/aqmg/SCRAM/models/other/isc3/isc3v2.pdf
PASQUILL_GIFFORD_STABILITY_CLASSES = pd.Index(['A','B','C','D','E','F'])
PG_STABILITY_DICT  = {"a" : [170.0, 98.0, 61.0, 32.0, 21.0, 14.0], 
                              "b" : [1.09, 0.98, 0.91, 0.81, 0.75, 0.68], 
                              "c" : [24.0, 18.0, 12.0, 8.0, 6.0, 4.0],
                              "d" : [2.5, 1.8, 1.1, 0.72, 0.54, 0.36]}
PG_STABILITY_PARAMETERS = pd.DataFrame(data = PG_STABILITY_DICT, 
                                             index = PASQUILL_GIFFORD_STABILITY_CLASSES)
PG_Y_PRE_EXPONENTIAL = 465.11628

class PointSampler: 
    def __init__ (self, sampler_location, sampler_height):
        self.sampler_location = sampler_location 
        self.sampler_height = np.asarray(sampler_height)

    def add_source(self, source_location, source_width_meters, effective_stack_height):
        self.source_location = source_location
        self.source_width = source_width_meters
        self.stack_height = effective_stack_height 

    def sample_concentration(self, wind_speed, stability_class, emission_rate, wind_angle_deg):
        downwind_distance, crosswind_distance = self.get_relative_distances(wind_angle_deg)
        return self.conc_at_point(downwind_distance, crosswind_distance, wind_speed, stability_class, emission_rate)

    def conc_at_point(self, downwind_distance, crosswind_distance, wind_speed, stability_class, emission_rate):
        downwind_distance_km = downwind_distance / KILOMETRE 

        concentration = np.zeros_like(downwind_distance) 

        downwind_distance_km = np.asarray(downwind_distance_km)
        positive_distance_index = downwind_distance_km >= ZERO_THRESHOLD_KM
        positive_downwind_distance = downwind_distance_km[positive_distance_index]

        if np.any(positive_distance_index): 
            standard_deviations = self.calculate_standard_deviations(positive_downwind_distance, wind_speed, 
                                                                        stability_class)
            denominator = self.calculate_denominator(wind_speed, standard_deviations[1]) 

            numerator = self.calculate_numerator(


        return concentration

    def emission_at_sampler(self, wind_speed, stability_class, emission_rate, wind_angle_deg):

        return None 

    def get_relative_distances(self, wind_angle_deg):
        wind_angle_deg = (wind_angle_deg + 180) % 360

        total_distance = geodesic(self.source_location, self.sampler_location).meters 
        sampler_bearing = geodesic(self.source_location, self.sampler_location).bearing 

        downwind_distance = total_distance * np.cos((wind_angle_deg - sampler_bearing) * np.pi / 180)
        crosswind_distance = total_distance * np.sin((wind_angle_deg - sampler_bearing) * np.pi / 180) 

        return downwind_distance, crosswind_distance 

    def calculate_standard_deviations(self, downwind_distance_km, wind_speed, stability_class):
        current_pg_params = {'a': PG_STABILITY_PARAMETERS.loc[stability_class,'a'],
                             'b': PG_STABILITY_PARAMETERS.loc[stability_class,'b'],
                             'c': PG_STABILITY_PARAMETERS.loc[stability_class,'c'],
                             'd': PG_STABILITY_PARAMETERS.loc[stability_class,'d']}

        standard_deviation_z = current_pg_params['a'] * downwind_distance_km ** current_pg_params['b'] 

        y_angle_term = np.radians(current_pg_params['c'] - current_pg_params['d'] * np.log(downwind_distance_km))
        standard_deviation_y = PG_Y_PRE_EXPONENTIAL * downwind_distance_km * np.tan(y_angle_term)

        return standard_deviation_y, standard_deviation_z

    def calculate_denominator(self, wind_speed, standard_deviation_z):
        denominator = 2 * np.sqrt(2 * np.pi) * wind_speed * standard_deviation_z 
        denominator_no_zeroes = np.maximum(denominator,DENOMINATOR_ZERO_THRESHOLD)
        return denominator_no_zeroes

