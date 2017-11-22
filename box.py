from units import unit
import numpy as np
from copy import copy

from filters import (
    gaussian_filter_2d, gaussian_filter_3d,
    sobel_filter_2d, sobel_filter_3d)

R_earth = 6.371e3 * unit.km
day = 0.0027378507871321013 * unit.year


def is_linear(a, eps=1e-3):
    x = np.diff(a[1:-1]).std() / np.diff(a[1:-1]).mean()
    return x < eps


class Box:
    """Stores properties of the coordinate system used."""
    def __init__(self, time, lat, lon):
        self.time = time
        self.lat = lat
        self.lon = lon

    @staticmethod
    def from_netcdf(nc):
        """Obtain latitude, longitude and time axes from a given NetCDF
        object."""
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        time = nc.variables['time'][:]
        return Box(time, lat, lon)

    @staticmethod
    def generate(n_lon):
        """Generate a box without time axis, with the given number of pixels in
        the longitudinal direction. Latitudes are given half that size to
        create a 2:1 rectangular projection."""
        n_lat = n_lon // 2
        lat = np.linspace(-90.0, 90, n_lat + 1)[1:-1]
        lon = np.linspace(0., 360., n_lon, endpoint=False)
        time = None
        return Box(time, lat, lon)

    def __getitem__(self, s):
        """Slices the box in the same way as you would slice data."""
        new_box = copy(self)

        if not isinstance(s, tuple):
            s = (s,)

        for q, t in zip(s, ['time', 'lat', 'lon']):
            setattr(new_box, t, getattr(self, t).__getitem__(s))

        return new_box

    @property
    def shape(self):
        if self.time is None:
            return (self.lat.size, self.lon.size)
        else:
            return (self.time.size, self.lat.size, self.lon.size)

    @property
    def rectangular(self):
        """Check wether the given latitudes and longitudes are linear, hence
        if the data is in geographic projection."""
        return is_linear(self.lat) and is_linear(self.lon)

    @property
    def resolution(self):
        """Gives the resolution of the box in units of years and km.
        The resolution of longitude is given as measured on the equator."""
        res_lat = np.diff(self.lat[1:-1]).mean() * (np.pi / 180) * R_earth
        res_lon = np.diff(self.lon[1:-1]).mean() * (np.pi / 180) * R_earth

        if self.time is not None:
            res_time = np.diff(self.time[1:-1]).mean() * day
            return res_time, res_lat, res_lon
        else:
            return res_lat, res_lon

    def gaussian_filter(self, data, sigma):
        """Filters a data set with a Gaussian, correcting for the distortion
        from the geographic projection.

        :param box: instance of :py:class:`Box`.
        :param data: data set, dimensions should match ``box.shape``.
        :param sigma: list of sigmas with the correct dimension.
        :return: :py:class:`numpy.ndarray` with the same shape as input.
        """
        if self.time is not None:
            return gaussian_filter_3d(self, data, *sigma)
        else:
            return gaussian_filter_2d(self, data, *sigma)

    def sobel_filter(self, data, weight=None, physical=True):
        """Sobel filter. Effectively computes a derivative.  This filter is
        normalised to return a rate of change per pixel, or if weights are
        given, the value is multiplied by the weight to obtain a unitless
        quantity of change over the given weight.

        :param box: :py:class:`Box` instance
        :param data: input data, :py:class:`numpy.ndarray` with same shape
            as ``box.shape``.
        :param weight: weight of each dimension in combining components into
            a vector magnitude; should have units corresponding those given
            by ``box.resolution``."""
        if self.time is not None:
            return sobel_filter_3d(self, data, weight, physical)
        else:
            return sobel_filter_2d(self, data, weight, physical)
