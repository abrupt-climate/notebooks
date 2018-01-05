from units import unit
import numpy as np
from copy import copy
import netCDF4

from filters import (
    gaussian_filter_2d, gaussian_filter_3d,
    sobel_filter_2d, sobel_filter_3d)

from pyparsing import Word, alphas, nums, Group, Suppress, Combine, tokenMap
from datetime import date, timedelta
from stats import weighted_quartiles


R_earth = 6.371e3 * unit.km
day = 0.0027378507871321013 * unit.year


def parse_time_units(units):
    p_int = Word(nums).setParseAction(tokenMap(int))
    p_date = Group(p_int('year') + Suppress('-') +
                   p_int('month') + Suppress('-') +
                   p_int('day'))('date').setParseAction(
                       tokenMap(lambda args: date(*args)))
    p_time_unit = Word(alphas)('units') + Suppress("since") + p_date
    result = p_time_unit.parseString(units)
    return result['units'], result['date'][0]


def is_linear(a, eps=1e-3):
    x = np.diff(a[1:-1]).std() / np.diff(a[1:-1]).mean()
    return x < eps


def overlap_idx(t1, t2):
    idx = np.where(t1 >= t2[0])[0]
    if len(idx) == 0:
        return slice(None)
    else:
        return slice(0, idx[0])


class File(object):
    def __init__(self, f):
        self.path = f
        self.data = netCDF4.Dataset(f, 'r', format='NETCDF4')
        self.bounds = slice(None)

    @property
    def time(self):
        return self.data.variables['time'][self.bounds]

    @property
    def lat(self):
        return self.data.variables['lat'][:]

    @property
    def lon(self):
        return self.data.variables['lon'][:]

    @property
    def lat_bnds(self):
        return self.data.variables['lat_bnds'][:]

    @property
    def lon_bnds(self):
        return self.data.variables['lon_bnds'][:]


    def get(self, var):
        return self.data.variables[var][self.bounds]

    @property
    def time_units(self):
        dt, t0 = parse_time_units(self.data.variables['time'].units)
        return dt, t0


class DataSet(object):
    """Deals with sets of NetCDF files, combines data, and
    generates a valid :py:class:`Box` instance from these files.
    """
    pattern = "{variable}_*mon_{model}_{scenario}" \
              "_{realization}_??????-??????.nc"

    def __init__(self, *, path, model, variable, scenario, realization):
        self.path = path
        self.model = model
        self.variable = variable
        self.scenario = scenario
        self.realization = realization

        self._box = None
        self._data = None

    def glob(self):
        return list(self.path.glob(self.pattern.format(**self.__dict__)))

    def load(self):
        self.files = sorted(
            list(map(File, self.glob())),
            key=lambda f: f.time[0])

        bounds = [overlap_idx(self.files[i].time, self.files[i+1].time)
                  for i in range(len(self.files) - 1)] + [slice(None)]

        for f, b in zip(self.files, bounds):
            f.bounds = b

        self._data = None

    @property
    def box(self):
        if self._box is None:
            time = np.concatenate([f.time for f in self.files])
            lat = self.files[0].lat
            lon = self.files[0].lon
            lat_bnds = self.files[0].lat_bnds
            lon_bnds = self.files[0].lon_bnds
            dt, t0 = self.files[0].time_units
            self._box = Box(time, lat, lon, lat_bnds, lon_bnds, dt, t0)

        return self._box

    @property
    def data(self):
        if self._data is None:
            self._data = np.concatenate([f.get(self.variable) for f in self.files])

        return self._data


class Box:
    """Stores properties of the coordinate system used."""
    def __init__(self, time, lat, lon,
                 lat_bnds=None, lon_bnds=None,
                 time_units='days',
                 time_start=date(1850, 1, 1)):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.lat_bnds = lat_bnds
        self.lon_bnds = lon_bnds

        self.time_units = time_units
        self.time_start = time_start

    @staticmethod
    def from_netcdf(nc):
        """Obtain latitude, longitude and time axes from a given
        NetCDF object."""
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        lat_bnds = nc.variables['lat_bnds'][:]
        lon_bnds = nc.variables['lon_bnds'][:]
        time = nc.variables['time'][:]
        dt, t0 = parse_time_units(nc.variables['time'].units)
        return Box(time, lat, lon, lat_bnds, lon_bnds, dt, t0)

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

    def date(self, value):
        try:
            return (self.date(t) for t in value)
        except TypeError:
            pass

        kwargs = {self.time_units: value}
        return self.time_start + timedelta(**kwargs)

    @property
    def dates(self):
        return list(self.date(self.time))

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

    def sobel_filter(self, data, weight=None, physical=True, variability=None):
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
            return sobel_filter_3d(self, data, weight, physical, variability)
        else:
            return sobel_filter_2d(self, data, weight, physical)

    @property
    def relative_grid_area(self):
        lat_bnds = np.radians(self.lat_bnds)
        lon_bnds = np.radians(self.lon_bnds)
        delta_lat = (np.sin(lat_bnds[:,1]) - np.sin(lat_bnds[:,0])) / 2
        delta_lon = (lon_bnds[:,1] - lon_bnds[:,0]) / (2*np.pi)
        return delta_lon[None,:] * delta_lat[:, None]

    def calibrate_sobel(self, data, delta_t, delta_d):
        sbc = self.sobel_filter(data, weight=[delta_t, delta_d, delta_d])
        var_t = (sbc[0]**2 / sbc[3]**2)
        var_x = (sbc[1]**2 + sbc[2]**2) / sbc[3]**2

        weights = np.repeat(
            self.relative_grid_area[None, :, :], self.shape[0], axis=0)
        ft = weighted_quartiles(var_t.flat, weights.flat)
        fx = weighted_quartiles(var_x.flat, weights.flat)
        fm = weighted_quartiles((1.0 / sbc[3]).flat, weights.flat)

        return {
            'time': np.sqrt(ft),
            'distance': np.sqrt(fx),
            'magnitude': fm,
            'gamma': np.sqrt(ft / fx)
        }
