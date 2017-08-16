import numpy as np
import netCDF4

from scipy import (ndimage)
from scipy.ndimage import gaussian_filter

from copy import (copy)
from units import (unit)

from hyper_canny import (edge_thinning, double_threshold)


def smooth(box, data, sigma_t, sigma_l):
    res_t, res_lat, res_lon = box.resolution
    outp = np.zeros_like(data)
    s_t = (sigma_t / res_t).m_as('')
    s_lon = (sigma_l / res_lon).m_as('')
    s_lat = (sigma_l / res_lat).m_as('')

    gaussian_filter(data, [s_t, s_lat, 0.0], output=outp)

    data[:] = outp
    h = data.shape[1] / 2

    # Each lattitude is smoothed in the longitudinal direction with a
    # gaussian of a different fwhm, scaling with the cosine of the lattitude.
    for i in range(data.shape[1]):
        gaussian_filter(
            data[:, i, :],
            min(data.shape[2], s_lon / np.cos((h - i) / (2*h) * np.pi)),
            output=outp[:, i, :])

    return outp


def full_sobel(box, d, w_t, w_l):
    res_t, res_lat, res_lon = box.resolution

    sb_t = ndimage.sobel(
        d, mode=['reflect', 'reflect', 'wrap'], axis=0) / res_t
    sb_lat = ndimage.sobel(
        d, mode=['reflect', 'reflect', 'wrap'], axis=1) / res_lat
    sb_lon = ndimage.sobel(
        d, mode=['reflect', 'reflect', 'wrap'], axis=1) / res_lon

    # derivatives are now correct for sb_lon on the equator, but not anywhere
    # else; we devide by the cosine of the lattitude to correct for this
    sb_lon /= np.cos(box.lat[:] * (np.pi / 180.))[None, :, None]

    sb = np.array([
        (sb_t * w_t).m_as(''),
        (sb_lat * w_l).m_as(''),
        (sb_lon * w_l).m_as(''),
        np.ones_like(sb_t.m)])
    norm = np.sqrt((sb[:3]**2).sum(axis=0))
    sb /= norm

    return sb


R_earth = 6.371e3 * unit.km
day = 0.0027378507871321013 * unit.year


def is_linear(a, eps=1e-3):
    x = np.diff(a[1:-1]).std() / np.diff(a[1:-1]).mean()
    return x < eps


class Box:
    def __init__(self, nc, config):
        lat_kw = config['lattitude_kw'].value
        lon_kw = config['longitude_kw'].value
        self.lat = nc.variables[lat_kw]
        self.lon = nc.variables[lon_kw]
        self.time = nc.variables['time']

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


class Data:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.box = None

        self.config['load'].on_click(self.on_load_click)
        self.config['select'].on_click(self.on_select_click)
        self.config['filter'].on_click(self.on_filter_click)

    def on_load_click(self, button):
        nc_data = netCDF4.Dataset(
            self.config['input_file'].value, 'r', format='NETCDF4')
        self.data = nc_data.variables[self.config['variable'].value]
        self.box = Box(nc_data, self.config)

    def on_select_click(self, button):
        if self.config['selection_tab'].selected_index == 0:
            self.selected_data = self.data[:]
            self.selected_box = self.box

        if self.config['selection_tab'].selected_index == 1:
            init = self.config['month'].value
            self.selected_data = self.data[init::12]
            self.selected_box = self.box[init::12]

    def on_filter_click(self, button):
        sigma_t = unit(self.config['sigma_t'].value)
        sigma_l = unit(self.config['sigma_l'].value)
        self.filtered_data = smooth(
                self.selected_box, self.selected_data, sigma_t, sigma_l)
        w_l = unit(self.config['weight_l'].value)
        w_t = unit(self.config['weight_t'].value)
        self.sobel_data = full_sobel(
                self.selected_box, self.filtered_data, w_t, w_l)
        self.thinned_mask = edge_thinning(self.sobel_data)
