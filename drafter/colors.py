import functools
import logging
import warnings

from cycler import cycler
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np


logger = logging.getLogger(__name__)

def _reverser(func, x):
    # adapted from matplotlib.colors.LinearSegmentedColormap._reverser
    return func(1 - x)

def reversed(cmap):
    # adapted from matplotlib.colors.LinearSegmentedColormap.reversed
    # Using a partial object keeps the cmap picklable.
    data_r = {
        key: (functools.partial(_reverser, data))
        for key, data in cmap._segmentdata.items()}

    new_cmap = mcolors.LinearSegmentedColormap(
        cmap.name,
        data_r,
        cmap.N,
        cmap._gamma,
    )

    # Reverse the over/under values too
    new_cmap._rgba_over = cmap._rgba_under
    new_cmap._rgba_under = cmap._rgba_over
    new_cmap._rgba_bad = cmap._rgba_bad

    return new_cmap


def _truncator(func, light, dark, x):
    # adapted from matplotlib.colors.LinearSegmentedColormap._reverser
    if (light > 1) or (light < 0):
        raise ValueError(f"light {light} not in [0, 1]")
    if (dark > 1) or (dark < 0):
        raise ValueError(f"dark {dark} not in [0, 1]")

    return func(x * (light - dark) + dark)


def truncated(cmap, light, dark):
    # adapted from matplotlib.colors.LinearSegmentedColormap.reversed
    # Using a partial object keeps the cmap picklable.
    data_t = {
        key: (functools.partial(_truncator, data, light, dark))
        for key, data in cmap._segmentdata.items()
    }

    new_cmap = mcolors.LinearSegmentedColormap(
        cmap.name,
        data_t,
        cmap.N,
        cmap._gamma,
    )

    # Truncate the over/under values too
    new_cmap._rgba_over = cmap._rgba_under
    new_cmap._rgba_under = cmap._rgba_over
    new_cmap._rgba_bad = cmap._rgba_bad

    return new_cmap


def cubehelix_colormap(
    *,
    start=None,
    rot=None,
    gamma=None,
    hue=None,
    light=1,
    dark=0,
    name=None,
    reverse=False,
):
    """
    cubehelix color scheme by Dave Green (https://people.phy.cam.ac.uk/dag9/CUBEHELIX/)
    """
    # Note: this relies on an internal matplotlib function, so may need to be
    # updated in the future
    cdict = mpl._cm.cubehelix(gamma=gamma, s=start, r=rot, h=hue)

    cmap = mcolors.LinearSegmentedColormap(name, cdict)

    cmap = truncated(cmap, light, dark)

    if reverse:
        cmap = reversed(cmap)

    return cmap


def cubehelix_palette(
    n_colors=6,
    start=0,
    rot=0.4,
    gamma=1.0,
    hue=0.8,
    light=0.85,
    dark=0.15,
    reverse=False,
):
    cmap = cubehelix_colormap(
        start=start,
        rot=rot,
        gamma=gamma,
        hue=hue,
        light=light,
        dark=dark,
        name=None,
        reverse=reverse,
    )

    x = np.linspace(0, 1, n_colors)
    palette = cmap(x)[:, :3].tolist()

    return palette


# @contextmanager
def set_palette(
    *,
    axes=None,
    palette=None,
):
    cycle = cycler(color=palette)
    # mpl.rcParams["axes.prop_cycle"] = cycle
    # return None

    # with mpl.rc_context({"axes.prop_cycle": cycle}):
    #     yield True

    axes.set_prop_cycle(cycle)

    return None


_drafter = cubehelix_colormap(
   start=1,
   rot=-0.1,
   gamma=1,
   hue=0,
   light=0.85,
   dark=0.15,
   name="drafter",
)
_drafter_r = _drafter.reversed()

_drafter_reds = cubehelix_colormap(
   start=1,
   rot=-0.1,
   gamma=1,
   hue=1,
   light=0.85,
   dark=0.15,
   name="drafter_reds",
)
_drafter_reds_r = _drafter_reds.reversed()

_drafter_greens = cubehelix_colormap(
   start=2,
   rot=-0.1,
   gamma=1,
   hue=1,
   light=0.85,
   dark=0.15,
   name="drafter_greens",
)
_drafter_greens_r = _drafter_greens.reversed()

_drafter_blues = cubehelix_colormap(
   start=0,
   rot=-0.1,
   gamma=1,
   hue=1,
   light=0.85,
   dark=0.15,
   name="drafter_blues",
)
_drafter_blues_r = _drafter_blues.reversed()


mpl.colormaps.register(cmap=_drafter)
mpl.colormaps.register(cmap=_drafter_r)
mpl.colormaps.register(cmap=_drafter_reds)
mpl.colormaps.register(cmap=_drafter_reds_r)
mpl.colormaps.register(cmap=_drafter_greens)
mpl.colormaps.register(cmap=_drafter_greens_r)
mpl.colormaps.register(cmap=_drafter_blues)
mpl.colormaps.register(cmap=_drafter_blues_r)
