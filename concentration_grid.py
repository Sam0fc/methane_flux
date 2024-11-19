import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.patheffects import withStroke
from matplotlib.patches import Polygon
from geopy.distance import geodesic
import math
import numpy as np
import flux_model as model 
import flux_maps as maps

class ConcentrationGrid:
    def __init__(self, x_range, y_range, z_range):
        self.xyz_grid = np.meshgrid(x_range, y_range, z_range)

    def add_concentration_source(self, emission_rate, wind_speed, effective_stack_height, stability_class, 
                                 source_width=0):
        if source_width == 0:
            self.concentration_data = model.conc_point(emission_rate, wind_speed, self.xyz_grid[0], self.xyz_grid[1], 
                                                       self.xyz_grid[2], effective_stack_height, stability_class)
        else:
            self.concentration_data = model.conc_line(emission_rate, wind_speed, self.xyz_grid[0], self.xyz_grid[1], 
                                                      self.xyz_grid[2], effective_stack_height, source_width, 
                                                      stability_class)

    def to_concentration_map(self, wind_angle, source_location):
        projection = ccrs.TransverseMercator(central_longitude=source_location[0], central_latitude=source_location[1]) 
        grid_origin_location = projection.transform_point(source_location[0], source_location[1], ccrs.PlateCaree())






