import numpy as np
from copy import copy
import netCDF4

from scipy import (special, signal, ndimage)
from scipy.ndimage import gaussian_filter

from ipywidgets import (interact)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

plt.rcParams['figure.figsize'] = (25, 10)

import os
from hyper_canny import cp_edge_thinning, cp_double_threshold


path = "/mnt/Knolselderij/bulk/Abrupt/"
fn1 = "tas_Amon_MPI-ESM-LR_rcp85_r1i1p1_200601-210012.nc"
fn2 = "tas_Amon_MPI-ESM-LR_rcp85_r1i1p1_210101-230012.nc"


def plot_mollweide(lats, lons, tas, **pargs):
    tas = np.concatenate([tas[:,:], tas[:, 0:1]], axis=1)
    lons = np.concatenate([lons, lons[0:1]])

    proj=ccrs.Mollweide()
    ax = plt.axes(projection=proj)
    plt.pcolormesh(
        lons, lats, tas, **pargs,
        transform=ccrs.RotatedPole(pole_longitude=180, pole_latitude=90))
    ax.coastlines()
    plt.colorbar()
    plt.show()


def plot_plate_carree(lats, lons, tas):
    tas = np.concatenate([tas[:,:], tas[:, 0:1]], axis=1)
    lons = np.concatenate([lons, lons[0:1]])

    proj=ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    plt.pcolormesh(
        lons, lats, tas,
        transform=ccrs.RotatedPole(pole_longitude=180, pole_latitude=90))
    ax.coastlines()
    plt.colorbar()
    plt.show()


# box = Box(get_data(os.path.join(path, fn1)))

def smooth(box, data, sigma_t, sigma_lat, sigma_lon):
    res_t, res_lat, res_lon = box.resolution
    outp = np.zeros_like(data)
    # FIXME: set the 'mode' parameter for cilindric coordinates.
    gaussian_filter(data, [sigma_t / res_t, sigma_lat / res_lat, 0.0], mode=['reflect', 'reflect', 'wrap'], output=outp)


def full_sobel(box, smooth_data, weights):
    dim = len(smooth_data.shape)
    y = [w * r for r, w in zip(box.resolution, weights)]

    sb = np.array([
        ndimage.sobel(smooth_data, mode=['reflect', 'reflect', 'wrap'], axis=i) * y[i]
        for i in range(dim)])

    h = lats[:] / 180 * np.pi
    sb[2,:,:,:] /= np.cos(h)[None,:,None]

    sb = np.r_[sb, np.ones_like(sb[0:1])]
    norm = np.sqrt((sb[:dim]**2).sum(axis=0))
    sb /= norm

# sb = full_sobel(box[2::12], smooth_data, weights=[100., 0.001, 0.001])
    return sb
    data[:] = outp
    h = data.shape[1] / 2
    for i in range(data.shape[1]):
        gaussian_filter(
            data[:,i,:],
            min(data.shape[2], (sigma_lon /res_lon) / np.cos((h - i) / (2*h) * np.pi)),
            mode=['reflect', 'wrap'],
            output=outp[:,i,:])

    return outp

# dat = sb.transpose([3,2,1,0]).copy()
# mask = cp_edge_thinning(dat)

# edges = cp_double_threshold(data=dat, mask=mask, a=1./600, b=1./500)
# m = edges.transpose([2, 1, 0])

signal = np.abs(1./sb[3,:,:,:])

def plot_signal_hist(signal, n_bins=100):
    n_times = signal.shape[0]
    signal_flat = signal.reshape([n_times, -1])
    bins = np.linspace(signal.min(), signal.max(), n_bins+1)
    hist = np.zeros(shape=(n_times, n_bins))

    for t in range(n_times):
        h, _ = np.histogram(signal_flat[t], bins=bins)
        hist[t, :] = h

    plt.pcolormesh(np.arange(n_times)+2006, bins[:-1], hist.T,
                   norm=colors.LogNorm(vmin=1, vmax=hist.max()))
    plt.show()
