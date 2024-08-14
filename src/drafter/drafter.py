from datetime import datetime, timezone
import importlib
import getpass
import logging
import warnings

import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.figure as mfigure
import matplotlib.image as mimage
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np


from . import typesetting
from . import style


logger = logging.getLogger(__name__)


# US Letter
MAX_WIDTH = 8.5
MAX_HEIGHT = 11

# Beamer
# MAX_WIDTH = 6.30  # cm_to_in(16)
# MAX_HEIGHT = 3.54  # cm_to_in(9)


def setup(styles=[]):
    style.set_style(styles=styles)
    typesetting.set_type()


def cm_to_in(l, /):
    return l / 2.54


def in_to_pt(l, /):
    return l * 6 * 12


def make_figure(**kwargs):
    import matplotlib.pyplot as plt

    backend_name = mpl.get_backend()
    logger.info(f"making figure with backend {backend_name}")

    fig = plt.figure(**kwargs)

    # fig.patch.set_alpha(0)

    # backend_name = mpl.get_backend()
    # logger.info(f"making figure with backend {backend_name}")

    # module = importlib.import_module(cbook._backend_module_name(backend_name))
    # canvas_class = module.FigureCanvas

    # fig = mfigure.Figure(**kwargs)
    # canvas = canvas_class(fig)

    # logger.info(f"making figure with pdf backend")

    # from matplotlib.backends.backend_pdf import FigureCanvas

    # fig = mfigure.Figure(**kwargs)
    # canvas = FigureCanvas(fig)

    return fig


def make_axes(
    nrows,
    ncols,
    *,
    fig_width=None,
    fig_height=None,
    width=2,
    height=2,
    margin=1,
    gutter=1,
    horizontal_margin=None,
    vertical_margin=None,
    left_margin=None,
    right_margin=None,
    top_margin=None,
    bottom_margin=None,
    horizontal_gutter=None,
    vertical_gutter=None,
    cbar_width=1/8,
    cbar_pad=1/8,
    sharex=None,
    sharey=None,
    squeeze=False,
):

    logger.info(f"making figure of size ({fig_width}, {fig_height})")
    # fig = plt.figure(figsize=(fig_width, fig_height))
    # fig = mfigure.Figure(figsize=(fig_width, fig_height))
    fig = make_figure(figsize=(fig_width, fig_height))

    fig_width, fig_height = fig.get_size_inches()
    if fig_width > MAX_WIDTH:
        warnings.warn(f"Figure width ({fig_width}) greater than maximum width ({MAX_WIDTH})")
    if fig_height > MAX_HEIGHT:
        warnings.warn(f"Figure height ({fig_height}) greater than maximum height ({MAX_HEIGHT})")

    if horizontal_gutter is None:
        horizontal_gutter = gutter

    if vertical_gutter is None:
        vertical_gutter = gutter

    if left_margin is None:
        if horizontal_margin is None:
            left_margin = margin
        else:
            left_margin = horizontal_margin

    if right_margin is None:
        if horizontal_margin is None:
            right_margin = margin
        else:
            right_margin = horizontal_margin

    if top_margin is None:
        if vertical_margin is None:
            top_margin = margin
        else:
            top_margin = vertical_margin

    if bottom_margin is None:
        if vertical_margin is None:
            bottom_margin = margin
        else:
            bottom_margin = vertical_margin

    if left_margin is None:
        raise ValueError(f"unspecified left margin!")
    if right_margin is None:
        raise ValueError(f"unspecified right margin!")
    if top_margin is None:
        raise ValueError(f"unspecified top margin!")
    if bottom_margin is None:
        raise ValueError(f"unspecified bottom margin!")

    if cbar_pad is not None:
        horizontal_gutter -= cbar_pad
        right_margin -= cbar_pad

    if cbar_width is not None:
        horizontal_gutter -= cbar_width
        right_margin -= cbar_width

    h = [Size.Fixed(left_margin)]
    h_idx = []
    for i in range(ncols):
        h_idx.append(len(h))
        h.append(Size.Fixed(width))
        h.append(Size.Fixed(cbar_pad))
        h.append(Size.Fixed(cbar_width))
        if i == ncols - 1:
            h.append(Size.Fixed(right_margin))
        else:
            h.append(Size.Fixed(horizontal_gutter))
    logger.info(f"made {len(h)} horizontal locations")

    v = [Size.Fixed(bottom_margin)]
    v_idx = []
    for i in range(nrows):
        v_idx.append(len(v))
        v.append(Size.Fixed(height))
        if i == nrows - 1:
            v.append(Size.Fixed(top_margin))
        else:
            v.append(Size.Fixed(vertical_gutter))
    logger.info(f"made {len(v)} vertical locations")

    total_width = sum(_h.fixed_size for _h in h)
    total_height = sum(_v.fixed_size for _v in v)
    logger.info(f"made axes of total size ({total_width}, {total_height})")

    if total_width > fig_width:
        warnings.warn(f"Total axes width ({total_width}) greater than figure width ({fig_width})")
    elif total_width < fig_width:
        warnings.warn(f"Total axes width ({total_width}) less than figure width ({fig_width})")

    if total_height > fig_height:
        warnings.warn(f"Total axes height ({total_height}) greater than figure height ({fig_height})")
    elif total_height < fig_height:
        warnings.warn(f"Total axes height ({total_height}) less than figure height ({fig_height})")

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    # return array of primary axes
    vaxes = []
    for v_id in v_idx:
        haxes = []
        for h_id in h_idx:
            ax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=h_id, ny=v_id),
            )
            cax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=h_id + 2, ny=v_id),
            )
            cax.set_visible(False)
            ax.cax = cax
            haxes.append(ax)
        vaxes.append(haxes)
    vaxes = vaxes[::-1]

    axes = np.array(vaxes)

    if sharex == "all":
        head_ax = axes[0, 0]
        for ax in axes.flat:
            ax.sharex(head_ax)
    elif sharex == "row":
        for row in axes:
            head_ax = row[0]
            for ax in row:
                ax.sharex(head_ax)
    elif sharex == "col":
        for col in axes.T:
            head_ax = col[0]
            for ax in col:
                ax.sharex(head_ax)
    elif sharex:
        raise ValueError(f"sharex for {sharex} not supported")

    if sharey == "all":
        head_ax = axes[0, 0]
        for ax in axes.flat:
            ax.sharey(head_ax)
    elif sharey == "row":
        for row in axes:
            head_ax = row[0]
            for ax in row:
                ax.sharey(head_ax)
    elif sharey == "col":
        for col in axes.T:
            head_ax = col[0]
            for ax in col:
                ax.sharey(head_ax)
    elif sharey:
        raise ValueError(f"sharex for {sharex} not supported")

    # https://github.com/matplotlib/matplotlib/blob/v3.8.4/lib/matplotlib/gridspec.py
    # turn off redundant tick labeling
    if sharex in ["col", "all"]:
        for ax in axes.flat:
            ax._label_outer_xaxis(skip_non_rectangular_axes=True)
    if sharey in ["row", "all"]:
        for ax in axes.flat:
            ax._label_outer_yaxis(skip_non_rectangular_axes=True)

    logger.info(f"made {axes.size} axes")

    # NOTE this currently does not work because we bypass an internal gridspec...
    # fig.align_labels()
    # fig.align_titles()


    if squeeze:
        _axes = axes.item() if axes.size == 1 else axes.squeeze()
    else:
        _axes = axes

    return fig, _axes


def add_colorbar(
    axes,
    *args,
    **kwargs
):
    logger.info(f"adding colorbar to {axes}")
    axes.cax.set_visible(True)

    ax_fig = axes.get_figure()
    cb = ax_fig.colorbar(*args, cax=axes.cax, **kwargs)

    return cb


def imshow(
    axes,
    *args,
    **kwargs,
):
    kwargs = cbook.normalize_kwargs(kwargs, mimage.AxesImage)

    im = axes.imshow(*args, aspect="auto", **kwargs)

    axes.xaxis.set_major_locator(mticker.MaxNLocator(nbins="auto", integer=True))
    axes.xaxis.set_minor_locator(mticker.NullLocator())
    axes.yaxis.set_major_locator(mticker.MaxNLocator(nbins="auto", integer=True))
    axes.yaxis.set_minor_locator(mticker.NullLocator())

    return im


def mesh(
    axes,
    *args,
    edges=False,
    **kwargs,
):
    kwargs = cbook.normalize_kwargs(kwargs, mcoll.QuadMesh)

    if edges:
        # Draw mesh edges, styled as gridlines by default
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = mpl.rcParams["grid.color"]
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = mpl.rcParams["grid.linewidth"] / 2

    mesh = axes.pcolormesh(*args, **kwargs)

    return mesh


def save_fig(
    fig,
    fname,
):
    logger.info(f"saving {fig} to {fname}")
    author = getpass.getuser()

    fig.savefig(
        fname,
        dpi=300,
        metadata={
            "Author": author,
        },
    )

    return None


def sign(
    fig
):
    author = getpass.getuser()
    utc_dt = datetime.now(timezone.utc)
    dt = utc_dt.astimezone()
    today = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    fig.text(
        1.0,
        0.0,
        f"{author} {today}",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    return None


def watermark(
    fig,
    text="preliminary",
):
    fig_width, fig_height = fig.get_size_inches()
    angle = np.degrees(np.arctan2(fig_height, fig_width))
    size = in_to_pt(np.hypot(fig_width, fig_height) / 12)
    fig.text(
        0.5,
        0.5,
        text,
        color="k",
        alpha=0.1,
        fontsize=size,
        rotation=angle,
        horizontalalignment="center",
        verticalalignment="center",
    )

    return None


