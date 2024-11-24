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

    def build_concentration_source(self, emission_rate, wind_speed, effective_stack_height, stability_class, 
                                 source_width=0.0):
        if source_width == 0:
            self.concentration_data = model.conc_point(emission_rate, wind_speed, self.xyz_grid[0], self.xyz_grid[1], 
                                                       self.xyz_grid[2], effective_stack_height, stability_class)
        else:
            self.concentration_data = model.conc_line(emission_rate, wind_speed, self.xyz_grid[0], self.xyz_grid[1], 
                                                      self.xyz_grid[2], effective_stack_height, source_width, 
                                                      stability_class)

    def build_location_grid(self, origin_location, wind_angle_deg):
        self.origin_location = origin_location 
        self.update_location_grid(wind_angle_deg)

    def update_location_grid(self, wind_angle_deg):
        source_projection = ccrs.TransverseMercator(central_longitude=self.origin_location[0],
                                                    central_latitude=self.origin_location[1]) 
        grid_origin_location = source_projection.transform_point(self.origin_location[0], self.origin_location[1], 
                                                                 ccrs.PlateCarree())

        shifted_x_coords = self.xyz_grid[0][:,:,0] + grid_origin_location[0] 
        shifted_y_coords = self.xyz_grid[1][:,:,0] + grid_origin_location[1] 

        wind_flow_direction = 360 - wind_angle_deg 
        x_axis_angle = wind_flow_direction + 90

        x_axis_angle_rads = np.radians(x_axis_angle) 
        rotated_x_coords = shifted_x_coords * np.cos(x_axis_angle_rads) - shifted_y_coords * np.sin(x_axis_angle_rads) 
        rotated_y_coords = shifted_x_coords * np.sin(x_axis_angle_rads) + shifted_y_coords * np.cos(x_axis_angle_rads)

        self.lonlat_coords = ccrs.PlateCarree().transform_points(source_projection, rotated_x_coords, rotated_y_coords) 

        return (self.lonlat_coords[:,:,0], self.lonlat_coords[:,:,1])

    def find_closest_index(self, target_lon, target_lat):
        lon_values = self.lonlat_coords[0]
        lat_values = self.lonlat_coords[1]

        lon_diff = lon_values - target_lon
        lat_diff = lat_values - target_lat

        distance = np.sqrt(lon_diff**2 + lat_diff**2)
        min_distance_index = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

        return min_distance_index 

    def concentration_at_point(self, target_lon, target_lat):
        closest_index = self.find_closest_index(target_lon, target_lat)
        return self.concentration_data[closest_index]

