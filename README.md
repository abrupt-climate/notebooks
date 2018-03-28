Notebooks
=========

Shared space for Notebooks.

Using
-----
These notebooks depend on:
* [Jupyter](https://jupyter.org) notebook interface.
* [NumPy](https://numpy.org) array manipulation.
* [SciPy](https://scipy.org) numerical algorithms.
* [Matplotlib](https://matplotlib.org/) makes plots.
* [Cython](http://cython.org/) glue C and Python.
* [HyperCanny](https://github.com/abrupt-climate/hyper-canny) N-dimensional Canny edge detection.
* [Cartopy](http://scitools.org.uk/cartopy/) enable plotting with cartographic projections.
* [Pint](http://pint.readthedocs.io/en/latest/) manage physical units.

This last package needs some libraries to be installed. Installation
instructions differ slightly, depending on your OS.

See Installation_Instructions.txt for more detailed information.

### Debian / Ubuntu

```bash
> apt install g++ proj-bin libproj-dev libgeos-dev
```

### Fedora

```bash
> dnf install gcc-c++ geos-devel proj-devel
```

### Python modules
And for the Python modules (Jupyter, Matplotlib, Cartopy, Cython, Pint), we
recommend using a [virtual
environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

```bash
> pip install -r requirements.txt
```

### Running
To run the notebook server

```bash
> jupyter notebook
```

A browser window should open automatically.
