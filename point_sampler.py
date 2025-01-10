import numpy as np 
import pandas as pd
from geopy.distance import geodesic 
from scipy.special import erf
from shapely.geometry import Polygon
import math

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
        ZERO_THRESHOLD_KM = 0.000001
        KILOMETRE = 1000.0

        downwind_distance_km = downwind_distance / KILOMETRE 
        concentration = np.zeros_like(downwind_distance) 

        positive_emission_rate = emission_rate
        if not np.isscalar(downwind_distance):
            downwind_distance_km = np.asarray(downwind_distance_km)
        
            positive_distance_index = downwind_distance_km >= ZERO_THRESHOLD_KM
            positive_downwind_distance = downwind_distance_km[positive_distance_index]
            positive_crosswind_distance = crosswind_distance[positive_distance_index]
            positive_stability_class = stability_class[positive_distance_index] 
            positive_wind_speed = wind_speed[positive_distance_index]
            if not np.isscalar(emission_rate):
                positive_emission_rate = emission_rate[positive_distance_index]
            if not np.isscalar(self.source_width):
                self.source_width = self.source_width[positive_distance_index]
        else:
            positive_distance_index = downwind_distance_km >= ZERO_THRESHOLD_KM
            positive_downwind_distance = downwind_distance_km
            positive_crosswind_distance = crosswind_distance
            positive_stability_class = stability_class
            positive_wind_speed = wind_speed
        

        if positive_distance_index.any(): 
            standard_deviations = self.calculate_standard_deviations(positive_downwind_distance, positive_stability_class)

            denominator = self.calculate_denominator(positive_wind_speed, standard_deviations[1], standard_deviations[0]) 
            numerator = self.calculate_numerator(positive_emission_rate, standard_deviations, positive_crosswind_distance)

            positive_distance_concentration = numerator / denominator 
            concentration[positive_distance_index] = positive_distance_concentration 

        return concentration

    def emission_at_sampler(self, wind_speed, stability_class, concentration, wind_angle_deg):
        BACKGROUND_CH4 = 1978.5
        concentration = concentration - BACKGROUND_CH4
        concentration = concentration/1e3
        print(concentration)
        model_scale = self.g_m3_to_ppm(self.sample_concentration(wind_speed, stability_class, 1, wind_angle_deg))
        if not np.isscalar(concentration):
            model_scale[model_scale == 0] = np.nan
        return concentration/model_scale
    
    def g_m3_to_ppm(self, methane_g_m3, temperature=273.15, pressure=101325):
        # T in K, P in Pa
        # Constants
        MOLAR_MASS_METHANE = 16.04  # g/mol for methane (CH4)
        R = 8.314  # J/(mol*K), ideal gas constant
        
        # Calculate the molar concentration of methane in mol/m^3
        molar_concentration = methane_g_m3 / MOLAR_MASS_METHANE  # mol/m^3
        
        # Calculate volume at specified conditions (temperature in Kelvin, pressure in Pascals)
        molar_volume = R * temperature / pressure #m^3/mol

        # Convert molar concentration to ppm
        
        ppm = molar_volume * molar_concentration * 1e6
        
        return ppm

    def calculate_error_functions(self, crosswind_distance_km, standard_deviation_y):
        crosswind_distance_km = np.asarray(crosswind_distance_km)
        standard_deviation_y = np.asarray(standard_deviation_y)
        right_y = np.divide((crosswind_distance_km + self.source_width / 2), (np.sqrt(2) * standard_deviation_y))
        left_y = (crosswind_distance_km - self.source_width / 2) / (np.sqrt(2) * standard_deviation_y) 
        diff_error_functions = erf(right_y) - erf(left_y) 

        return diff_error_functions

    def get_relative_distances(self, wind_angle_deg):

        total_distance = geodesic(self.source_location, self.sampler_location).meters 
        sampler_bearing = self.get_bearing()

        downwind_distance = total_distance * np.cos((wind_angle_deg - sampler_bearing) * np.pi / 180)
        crosswind_distance = total_distance * np.sin((wind_angle_deg - sampler_bearing) * np.pi / 180) 
        return downwind_distance, crosswind_distance 

    def get_bearing(self):
        pointA = self.sampler_location
        pointB = self.source_location

        lat1, lat2 = map(math.radians, [pointA[0], pointB[0]])
        diffLong = math.radians(pointB[1] - pointA[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)

        initial_bearing = math.degrees(math.atan2(x, y))

        return (initial_bearing + 360) % 360



    def calculate_standard_deviations(self, downwind_distance_km, stability_class):
        ## Theory from https://gaftp.epa.gov/Air/aqmg/SCRAM/models/other/isc3/isc3v2.pdf
        PASQUILL_GIFFORD_STABILITY_CLASSES = pd.Index(['A','B','C','D','E','F'])
        PG_STABILITY_DICT  = {"a" : [170.0, 98.0, 61.0, 32.0, 21.0, 14.0], 
                                      "b" : [1.09, 0.98, 0.91, 0.81, 0.75, 0.68], 
                                      "c" : [24.0, 18.0, 12.0, 8.0, 6.0, 4.0],
                                      "d" : [2.5, 1.8, 1.1, 0.72, 0.54, 0.36]}
        PG_STABILITY_PARAMETERS = pd.DataFrame(data = PG_STABILITY_DICT, 
                                                     index = PASQUILL_GIFFORD_STABILITY_CLASSES)
        PG_Y_PRE_EXPONENTIAL = 465.11628

        current_pg_params = {'a': PG_STABILITY_PARAMETERS.loc[stability_class,'a'],
                             'b': PG_STABILITY_PARAMETERS.loc[stability_class,'b'],
                             'c': PG_STABILITY_PARAMETERS.loc[stability_class,'c'],
                             'd': PG_STABILITY_PARAMETERS.loc[stability_class,'d']}
        
        standard_deviation_z = current_pg_params['a'] * downwind_distance_km ** current_pg_params['b'] 

        y_angle_term = np.radians(current_pg_params['c'] - current_pg_params['d'] * np.log(downwind_distance_km))
        standard_deviation_y = PG_Y_PRE_EXPONENTIAL * downwind_distance_km * np.tan(y_angle_term)

        return np.asarray(standard_deviation_y), np.asarray(standard_deviation_z)

    def calculate_denominator(self, wind_speed, standard_deviation_z, standard_deviation_y):
        DENOMINATOR_ZERO_THRESHOLD = 1e-10
        if np.isscalar(self.source_width) and self.source_width == 0: 
            return 2 * np.pi * wind_speed * standard_deviation_z * standard_deviation_y

        denominator = 2 * np.sqrt(2 * np.pi) * wind_speed * standard_deviation_z 
        denominator_no_zeroes = np.maximum(denominator, DENOMINATOR_ZERO_THRESHOLD)
        return denominator_no_zeroes 

    def calculate_numerator(self, emission_rate, standard_deviations, crosswind_distance_km):
        exp1 = np.exp(-((self.sampler_height - self.stack_height) ** 2) / (2 * standard_deviations[1] ** 2)) 
        exp2 = np.exp(-((self.sampler_height + self.stack_height) ** 2) / (2 * standard_deviations[1] ** 2))
        if np.isscalar(self.source_width) and self.source_width == 0: 
            exp3 = np.exp(-(crosswind_distance_km **2)/(2 * standard_deviations[0] ** 2))
            return emission_rate * exp3 * (exp1 + exp2)
        error_functions = self.calculate_error_functions(crosswind_distance_km, standard_deviations[0])
        return emission_rate * (exp1 + exp2) * error_functions

