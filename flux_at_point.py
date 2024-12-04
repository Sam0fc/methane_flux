import numpy as np

class Source:
    def __init__(self, latitude: float, longitude: float, height: np.array_like, source_width: float):
        self.latitude = latitude
        self.longitude = longitude
        self.height = np.asarray(height)


    def sample(self, sampler_location: tuple, sampler_height: float)
     