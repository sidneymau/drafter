import logging
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.style as mstyle

from . import colors


logger = logging.getLogger(__name__)


# _drafter_cmap = mcolors.LinearSegmentedColormap.from_list("drafter", ["#ffffff", "#000000"])
# mpl.colormaps.register(cmap=_drafter_cmap)


def set_style(styles=[], restore_defaults=False):
    if restore_defaults:
        logger.info(f"restoring matplotlib defaults")
        mpl.rcdefaults()

    logger.info("using drafter style")
    mstyle.use("drafter.styles.drafter")
    for style in styles:
        logger.info(f"using {style} style")
        mstyle.use(f"drafter.styles.{style}")

    # if "drafter" in mpl.colormaps:
    #     logger.info(f"using drafter colormap")
    #     mpl.rcParams["image.cmap"] = "drafter"
    # else:
    #     warnings.warn(f"drafter colormap not found")

    return None


