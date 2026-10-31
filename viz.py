import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COLORS = {
    "softmax": "#2a78d6",
    "p3": "#eb6834",
    "nested": "#1baf7a",
    "learned": "#6250d6",
    "adaptive": "#6250d6",
    "cascade": "#e87ba4",
    "chance": "#8a8a82",
}
RANK_COLORS = {8: "#c9c8c2", 16: "#a3a29b", 32: "#6f6e68", 64: "#3d3c38"}
def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.frameon": False,
    })
def chance_line(ax, n_classes=None):
    if n_classes is None:
        import data
        n_classes = len(data.CLASSES)
    y = 100.0 / n_classes
    ax.axhline(y, color=COLORS["chance"], linestyle=":", linewidth=1.5, zorder=1, label=f"chance ({y:.0f}%, {n_classes} classes)")
    return y
def label_bars(ax, bars, values, fmt="{:.1f}", dy=0.01):
    span = max(values) - min(0, min(values)) or 1
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * span, fmt.format(v), ha="center", va="bottom", fontsize=9)