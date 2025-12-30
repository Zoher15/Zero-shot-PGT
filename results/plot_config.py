"""
Shared configuration for all plotting scripts.
Ensures consistent colors, naming, and styling across all figures.
"""

from pathlib import Path

# =============================================================================
# COLORS (Colorblind-friendly palette)
# =============================================================================

BASELINE_COLOR = "#2A9D8F"  # Teal
COT_COLOR = "#E76F51"       # Coral/Clay
S2_COLOR = "#7F4CA5"        # Purple

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAMES = {
    'llava-onevision-7b': 'LLaVA',
    'qwen25-vl-7b': 'Qwen',
    'llama32-vision-11b': 'Llama'
}

MODEL_ORDER = ['llava-onevision-7b', 'qwen25-vl-7b', 'llama32-vision-11b']

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_NAMES = {
    'd3': 'D3',
    'df40': 'DF40',
    'genimage': 'GenImage'
}

DATASET_ORDER = ['d3', 'df40', 'genimage']

# =============================================================================
# METHOD/PHRASE CONFIGURATION
# =============================================================================

METHOD_NAMES = {
    'baseline': 'Baseline',
    'cot': 'CoT',
    's2': 'S2'
}

METHOD_ORDER = ['baseline', 'cot', 's2']

METHOD_COLORS = {
    'baseline': BASELINE_COLOR,
    'cot': COT_COLOR,
    's2': S2_COLOR
}

# =============================================================================
# FIGURE SETTINGS (ACL paper format)
# =============================================================================

# ACL page dimensions (measured from acl.sty with 2.5cm margins)
# Full text width: 455.24pt = 6.30 inches
# Column width: 219.09pt = 3.03 inches
# Column separation: 17.07pt = 0.24 inches

# Figure widths (rounded for convenience)
ACL_FULL_WIDTH = 6.3      # inches (2-column span, exact match)
ACL_COLUMN_WIDTH = 3.0    # inches (1-column, slightly conservative)

# Aliases for backward compatibility
FIGURE_WIDTH_2COL = ACL_FULL_WIDTH
FIGURE_WIDTH_1COL = ACL_COLUMN_WIDTH

# DPI for publication-quality figures
PUBLICATION_DPI = 300

# Output directory
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EVALUATION PARAMETERS (for loading results)
# =============================================================================

DEFAULT_PHRASE_MODE = 'prefill'
DEFAULT_N_RESPONSES = 1

# =============================================================================
# FONT SIZES (ACL publication quality)
# =============================================================================

TITLE_FONT_SIZE = 8         # Subplot/panel titles (D3, DF40, GenImage)
LEGEND_FONT_SIZE = 8        # Legend text
AXIS_LABEL_FONT_SIZE = 8    # Axis labels (e.g., "Macro F1 (%)", "Odds Change (%)")
TICK_LABEL_FONT_SIZE = 7    # Tick labels (numbers/categories on axes)
ANNOTATION_FONT_SIZE = 6    # Small annotations (improvement markers, etc.)

# =============================================================================
# PLOT STYLING (consistent across all figures)
# =============================================================================

# Bar plot styling
BAR_WIDTH = 0.25          # Width of individual bars in grouped bar charts
BAR_HEIGHT = 0.6          # Height of horizontal bars
BAR_EDGE_WIDTH = 0.5      # Width of bar borders

# Error bar styling
ERROR_BAR_WIDTH = 0.05    # Width of error bar lines
ERROR_BAR_CAPSIZE = 1     # Size of error bar caps

# Line plot styling (radar plots)
LINE_WIDTH = 1.5          # Width of lines in line/radar plots
MARKER_SIZE = 4           # Size of markers on lines

# Fill/shading styling
FILL_ALPHA = 0.15         # Transparency for filled regions (radar plots)

# Grid styling
GRID_COLOR = 'gray'
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'     # Dashed grid lines
GRID_LINEWIDTH = 0.5

# Spacing configuration
GROUP_SPACING = 1.0       # Spacing between bar groups
WORD_SPACING = 1.0        # Vertical spacing between words (vocab plots)

# Subplot spacing (for multi-panel figures)
SUBPLOT_TOP = 0.97        # Top margin
SUBPLOT_BOTTOM = 0.08     # Bottom margin
SUBPLOT_LEFT = 0.02       # Left margin (minimal for tight layout)
SUBPLOT_RIGHT = 0.98      # Right margin
SUBPLOT_WSPACE = 0.15     # Horizontal space between subplots (increased for independent y-axes)

# Legend positioning
LEGEND_Y_POSITION = 1.15  # Vertical position for shared legends (1.05 for radar)

# =============================================================================
# GENERATOR NAME MAPPING (for radar plots)
# =============================================================================

GENERATOR_NAMES = {
    # D3 generators
    "image_gen0": "DFIF",
    "image_gen1": "SD1.4",
    "image_gen2": "SD2.1",
    "image_gen3": "SDXL",

    # DF40 generators
    "collabdiff": "CDif",
    "midjourney": "MidJ",
    "Midjourney": "MidJ",
    "stargan": "SGan1",
    "starganv2": "SGan2",
    "styleclip": "SClip",
    "whichfaceisreal": "WFIR",

    # GenImage generators
    "adm": "ADM",
    "biggan": "BGan",
    "glide_copy": "Glide",
    "stable_diffusion_v_1_4": "SD1.4",
    "stable_diffusion_v_1_5": "SD1.5",
    "vqdm": "VQDM",
    "wukong": "Wukong",
}

# =============================================================================
# MATPLOTLIB STYLING FUNCTION
# =============================================================================

def set_publication_style():
    """
    Set consistent matplotlib style matching ACL paper format.

    Uses LaTeX rendering with Times Roman font matching ACL papers.
    Requires LaTeX installation on system with times package.
    """
    import matplotlib.pyplot as plt

    # Use LaTeX rendering with Times font (matching ACL style)
    plt.rcParams['text.usetex'] = True

    # Match ACL's LaTeX preamble: \usepackage[T1]{fontenc}\usepackage{times}
    plt.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}\usepackage{times}\usepackage{latexsym}'

    # Font family - use default serif (will be Times from LaTeX preamble)
    plt.rcParams['font.family'] = 'serif'

    # Ensure PDF uses TrueType fonts (required by publishers)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Professional styling
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['patch.linewidth'] = 0.5

    # Tick styling
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5


def calculate_middle_column_center(left_margin, right_margin, wspace, n_columns=3):
    """
    Calculate the x-position of the center column in a multi-subplot figure.

    Args:
        left_margin: Left margin (e.g., 0.06 for 6%)
        right_margin: Right margin (e.g., 0.99 for 1% from right)
        wspace: Horizontal spacing between subplots (e.g., 0.15)
        n_columns: Number of columns (default: 3)

    Returns:
        Float x-position of the middle column center in figure coordinates

    Example:
        middle_x = calculate_middle_column_center(0.06, 0.99, 0.15, n_columns=3)
        fig.legend(..., bbox_to_anchor=(middle_x, 1.02))
    """
    available_width = right_margin - left_margin
    subplot_width = available_width / (n_columns + (n_columns - 1) * wspace)
    gap_width = wspace * subplot_width

    # Center of middle column (for 3 columns, this is the 2nd column)
    middle_col_idx = n_columns // 2
    middle_column_center = left_margin + (middle_col_idx * (subplot_width + gap_width)) + (subplot_width / 2)

    return middle_column_center


def save_figure(fig, filepath, dpi=PUBLICATION_DPI):
    """
    Save figure in both PDF and PNG formats with exact dimensions.

    Args:
        fig: matplotlib figure object
        filepath: Path object or string (without extension)
        dpi: Resolution for PNG output (default: 300)

    Returns:
        Tuple of (pdf_path, png_path)

    Example:
        pdf_path, png_path = save_figure(fig, FIGURES_DIR / "my_plot")

    Note:
        Saves with exact figure dimensions (no bbox_inches='tight').
        Ensure proper spacing with plt.subplots_adjust() or fig.tight_layout().
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    filepath = Path(filepath)

    # Remove extension if present
    if filepath.suffix in ['.pdf', '.png']:
        filepath = filepath.with_suffix('')

    pdf_path = filepath.with_suffix('.pdf')
    png_path = filepath.with_suffix('.png')

    # Save both formats with exact dimensions
    plt.figure(fig.number)
    plt.savefig(pdf_path, format='pdf', dpi=dpi)
    plt.savefig(png_path, format='png', dpi=dpi)

    return pdf_path, png_path
