"""Site-matching colors and matplotlib style for the demo clips.

Single source of truth for palette, font, and figure rcParams so all five
clips render consistently with the supplementary website's white +
blue-ember palette (see styles.css).
"""
from __future__ import annotations
import matplotlib as mpl


# Colors mirror the website's CSS palette (styles.css :root tokens).
COL_BG          = "#ffffff"   # --paper
COL_PANEL       = "#f3f6fb"   # --paper-2 (chart panel fill)
COL_GRID        = "#e5e7eb"   # subtle grid (~rule)
COL_TEXT        = "#0d1117"   # --ink
COL_MUTED       = "#4b5563"   # --muted (caption / labels)
COL_REF         = "#9ca3af"   # reference lemniscate (medium gray)

# Method colors: bright green LQR, near-black Frozen, terra ER, ember blue ACE.
COL_LQR    = "#15803d"   # emerald-700, saturated green
COL_FROZEN = "#1f2937"   # gray-800, near-black
COL_ER     = "#b91c1c"   # red-700, terra (uniform-replay CL baseline)
COL_ACE    = "#1d4ed8"   # ember blue (the site accent)

METHOD_COLORS = {
    "LQR":    COL_LQR,
    "Frozen": COL_FROZEN,
    "ER":     COL_ER,
    "ACE":    COL_ACE,
}
METHOD_ORDER = ["LQR", "Frozen", "ACE"]


def apply_style():
    """Apply matplotlib rcParams that match the site's typography and look."""
    mpl.rcParams.update({
        "figure.facecolor": COL_BG,
        "axes.facecolor":   COL_BG,
        "axes.edgecolor":   COL_GRID,
        "axes.labelcolor":  COL_TEXT,
        "axes.titlesize":   15,
        "axes.titleweight": "semibold",
        "axes.labelsize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "xtick.color":      COL_MUTED,
        "ytick.color":      COL_MUTED,
        "grid.color":       COL_GRID,
        "grid.alpha":       0.7,
        "font.family":      ["DejaVu Sans"],
        "font.size":        13,
        "text.color":       COL_TEXT,
        "savefig.facecolor": COL_BG,
    })
