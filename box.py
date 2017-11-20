from units import unit
import numpy as np
from copy import copy


R_earth = 6.371e3 * unit.km
day = 0.0027378507871321013 * unit.year


def is_linear(a, eps=1e-3):
    x = np.diff(a[1:-1]).std() / np.diff(a[1:-1]).mean()
    return x < eps


class Box:
    """Stores properties of the coordinate system used."""
    def __init__(self, nc):
        self.lat = nc.variables['lat'][:]
        self.lon = nc.variables['lon'][:]
        self.time = nc.variables['time'][:]

    def __getitem__(self, s):
        new_box = copy(self)

        if not isinstance(s, tuple):
            s = (s,)

        for q, t in zip(s, ['time', 'lat', 'lon']):
            setattr(new_box, t, getattr(self, t).__getitem__(s))

        return new_box

    @property
    def rectangular(self):
        return is_linear(self.lat) and is_linear(self.lon)

    @property
    def resolution(self):
        res_lat = np.diff(self.lat[1:-1]).mean() * (np.pi / 180) * R_earth
        res_lon = np.diff(self.lon[1:-1]).mean() * (np.pi / 180) * R_earth
        res_time = np.diff(self.time[1:-1]).mean() * day
        return res_time, res_lat, res_lon
