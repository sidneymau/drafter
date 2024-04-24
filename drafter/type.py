import logging
import shutil
import warnings

import matplotlib as mpl
from matplotlib.texmanager import TexManager


logger = logging.getLogger(__name__)


def set_type(font="monospace", latex=True):
    logger.info(f"setting font in {font}")
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = 10.0

    if latex:
        logger.info(f"setting type in latex")
        latex_path = shutil.which("latex")

        if latex_path is None:
            warnings.warn(f"No executable found for 'latex'")
        else:
            logger.debug(f"latex is {latex_path}")
            mpl.rcParams["text.usetex"] = True
            # mpl.rcParams["text.latex.preamble"] = "\everymath{\\tt}"
            mpl.rcParams["text.latex.preamble"] = (
                r"\renewcommand*\familydefault{\ttdefault}"  # use typewriter text as default
                r"\usepackage[noendash,LGRgreek]{mathastext}"  # typeset math as text
                r"\MTfamily{cmtt}\MTgreekfont{cmtt}\Mathastext"  # use typewriter text for math
            )

        if font not in TexManager._font_families:
            warnings.warn(f"{font} is invalid latex font")

    return None
