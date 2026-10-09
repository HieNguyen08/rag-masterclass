"""
figkit — bộ công cụ vẽ hình minh họa cho RAG Masterclass.

Mỗi hình được vẽ HAI lần (theme sáng và tối) từ cùng một hàm, xuất ra SVG:
    docs/assets/figures/<module>/<tên>.light.svg
    docs/assets/figures/<module>/<tên>.dark.svg
Trang MkDocs Material hiển thị đúng bản theo theme qua hậu tố #only-light / #only-dark.

Cách dùng trong một file chương (vd ch03.py):

    from figkit import figure, run, T
    @figure("ten-hinh", size=(7, 3.2))
    def _(fig, t):            # t: bảng màu/chữ của theme hiện tại
        ax = fig.subplots()
        ...
    if __name__ == "__main__":
        run("03")
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "figures"

# --- Bảng màu (categorical, thứ tự cố định; bước sáng/tối riêng) -----------------
LIGHT = dict(
    name="light",
    fg="#1b1b1a", fg2="#52514e", muted="#8a8984", grid="#e4e3df", line="#c9c8c2",
    panel="#f4f3ef", panel2="#ebeae5", bg="#ffffff",
    c=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    good="#1f8f4e", bad="#d03b3b", warn="#c98500",
)
DARK = dict(
    name="dark",
    fg="#f2f1ec", fg2="#c3c2b7", muted="#8f8e86", grid="#3a3a38", line="#5a5955",
    panel="#262624", panel2="#2f2f2c", bg="#1e1e1c",
    c=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    good="#3fb071", bad="#e66767", warn="#d6a020",
)
# Tên gọi cho từng slot màu, dễ đọc trong code chương
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = range(8)

_REGISTRY: list[tuple[str, tuple[float, float], callable]] = []


def figure(name: str, size=(7.0, 3.4)):
    """Đăng ký một hàm vẽ. Hàm nhận (fig, t)."""
    def deco(fn):
        _REGISTRY.append((name, size, fn))
        return fn
    return deco


def _style(t):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],
        "font.size": 10,
        "svg.fonttype": "none",          # giữ chữ là text → file nhỏ, sắc nét, tìm kiếm được
        "text.color": t["fg"],
        "axes.labelcolor": t["fg2"],
        "axes.edgecolor": t["line"],
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": t["fg"],
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": t["fg2"],
        "ytick.color": t["fg2"],
        "xtick.labelcolor": t["fg2"],
        "ytick.labelcolor": t["fg2"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "mathtext.fontset": "dejavusans",
    })


_FONT_STACK = "'DejaVu Sans', 'Segoe UI', 'Helvetica Neue', Arial, 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', sans-serif"


def _postprocess(svg: str) -> str:
    # Thêm font dự phòng (SVG nhúng qua <img> chỉ dùng được font hệ thống)
    svg = re.sub(r"font-family:\s*'DejaVu Sans'", f"font-family: {_FONT_STACK}", svg)
    svg = svg.replace('font-family="DejaVu Sans"', f'font-family="{_FONT_STACK}"')
    svg = re.sub(r"font:\s*([\d.]+px)\s*'DejaVu Sans'", rf"font: \1 {_FONT_STACK}", svg)
    # Bỏ metadata ngày giờ để file ổn định giữa các lần chạy
    svg = re.sub(r"<metadata>.*?</metadata>\s*", "", svg, flags=re.S)
    return svg


def run(module: str, only: list[str] | None = None):
    out = OUT / module
    out.mkdir(parents=True, exist_ok=True)
    only = only or sys.argv[1:] or None
    for name, size, fn in _REGISTRY:
        if only and name not in only:
            continue
        for t in (LIGHT, DARK):
            _style(t)
            fig = plt.figure(figsize=size)
            fn(fig, t)
            prev = os.environ.get("FIG_PREVIEW")  # thư mục xuất PNG xem thử (tùy chọn)
            if prev:
                Path(prev).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(prev) / f"{module}-{name}.{t['name']}.png", dpi=110,
                            bbox_inches="tight", pad_inches=0.08, facecolor=t["bg"])
            p = out / f"{name}.{t['name']}.svg"
            fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.08,
                        metadata={"Date": None, "Creator": None})
            plt.close(fig)
            p.write_text(_postprocess(p.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"[{module}] {name}")


# --- Trợ giúp vẽ sơ đồ ----------------------------------------------------------
def canvas(fig, xlim=(0, 100), ylim=(0, 50)):
    """Trục 'vải vẽ' không có khung, đơn vị tùy chọn, tỉ lệ 1:1."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, text="", t=None, color=None, fill=None, fs=9.5, weight="normal",
        tc=None, radius=1.2, lw=1.4, alpha=1.0, ls="-", ha="center", va="center"):
    """Hộp bo góc; (x, y) là góc dưới-trái. color: viền; fill: nền."""
    ec = color or t["line"]
    fc = fill if fill is not None else t["panel"]
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, linestyle=ls)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 1.2
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, color=tc or t["fg"],
                weight=weight, linespacing=1.35)
    return p


def arrow(ax, x1, y1, x2, y2, t, color=None, lw=1.5, style="-|>", ls="-", rad=0.0, ms=12):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=color or t["fg2"], linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def tint(hex_color: str, t, k=0.18):
    """Pha màu series với nền theme → màu nền nhạt cho hộp."""
    def h2r(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    a, b = h2r(hex_color), h2r(t["bg"])
    m = [round(b[i] + (a[i] - b[i]) * k) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in m)


def ygrid(ax, t):
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def xgrid(ax, t):
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def note(ax, x, y, s, t, fs=8.5, color=None, **kw):
    ax.text(x, y, s, fontsize=fs, color=color or t["fg2"], **kw)
