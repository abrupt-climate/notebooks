import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def earth_plot(
        box, value,
        projection=ccrs.PlateCarree(),
        transform=ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90),
        patch_greenwich=True,
        patch_north_pole=False,
        **pargs):

    lons = box.lon.copy()
    lats = box.lat.copy()

    if patch_greenwich:
        value = np.concatenate([value[:, :], value[:, 0:1]], axis=1)
        lons = np.concatenate([lons, lons[0:1]])

    if patch_north_pole:
        lats = np.concatenate([lats, [90.0]])
        value = np.concatenate([value[:, :], value[0:1, :]], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    pcm = ax.pcolormesh(
        lons, lats, value, **pargs,
        transform=transform)
    ax.coastlines()
    fig.colorbar(pcm)
    plt.close()
    return fig


def plot_orthographic_np(box, value, central_latitude=90, **pargs):
    return earth_plot(
        box, value,
        projection=ccrs.Orthographic(
            central_latitude=central_latitude),
        patch_north_pole=True,
        **pargs)


def plot_mollweide(box, value, **pargs):
    return earth_plot(
        box, value,
        projection=ccrs.Mollweide(),
        **pargs)


def plot_plate_carree(box, value, **pargs):
    return earth_plot(box, value, **pargs)


def plot_signal_histogram(box, signal, **pargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # flatten sobel signal
    n_years = box.shape[0]
    signal_flat = signal.reshape([n_years, -1])

    # create bins for histograms
    n_bins = 100
    bins = np.linspace(signal.min(), signal.max(), n_bins+1)

    # generate histogram for each time step
    hist = np.zeros(shape=(n_years, n_bins))
    for t in range(n_years):
        h, _ = np.histogram(signal_flat[t], bins=bins)
        hist[t, :] = h

    # plot
    ax.pcolormesh(box.dates, bins[:-1], hist.T,
                  norm=colors.LogNorm(vmin=0.1, vmax=hist.max()),
                  cmap='Purples')
    ax.plot(box.dates, signal_flat.max(axis=1), '-', c='gray')
    plt.close()
    return fig
