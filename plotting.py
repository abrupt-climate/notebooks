import cartopy.crs as ccrs
import matplotlib.pyplot as plt
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
