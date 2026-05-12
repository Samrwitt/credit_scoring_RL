from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LineSeries:
    label: str
    y: np.ndarray
    color: str


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return [lo]
    span = hi - lo
    raw = span / max(1, (n - 1))
    mag = 10 ** np.floor(np.log10(raw))
    candidates = np.array([1, 2, 5, 10], dtype=float) * mag
    step = candidates[np.argmin(np.abs(candidates - raw))]
    start = np.floor(lo / step) * step
    end = np.ceil(hi / step) * step
    ticks = np.arange(start, end + 0.5 * step, step)
    return [float(x) for x in ticks]


def line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x: np.ndarray,
    series: Sequence[LineSeries],
    width: int = 1000,
    height: int = 600,
    y_lim: Optional[Tuple[float, float]] = None,
) -> str:
    # Layout
    margin_left, margin_right = 80, 20
    margin_top, margin_bottom = 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be non-empty")

    y_all = np.concatenate([np.asarray(s.y, dtype=float) for s in series]) if series else np.array([0.0])
    if y_lim is None:
        y_lo = float(np.nanmin(y_all))
        y_hi = float(np.nanmax(y_all))
        if y_lo == y_hi:
            y_lo -= 1.0
            y_hi += 1.0
    else:
        y_lo, y_hi = float(y_lim[0]), float(y_lim[1])

    x_lo, x_hi = float(np.nanmin(x)), float(np.nanmax(x))
    if x_lo == x_hi:
        x_lo -= 1.0
        x_hi += 1.0

    def sx(xv: float) -> float:
        return margin_left + (xv - x_lo) / (x_hi - x_lo) * plot_w

    def sy(yv: float) -> float:
        return margin_top + (1.0 - (yv - y_lo) / (y_hi - y_lo)) * plot_h

    # Colors for grid/axes/text
    axis = "#222"
    grid = "#D9D9D9"
    text = "#111"

    y_ticks = _nice_ticks(y_lo, y_hi, n=6)
    x_ticks = _nice_ticks(x_lo, x_hi, n=6)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # Title
    parts.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="18" fill="{text}">{_escape(title)}</text>')

    # Grid + ticks
    for yt in y_ticks:
        ypix = sy(yt)
        parts.append(f'<line x1="{margin_left}" y1="{ypix:.2f}" x2="{width-margin_right}" y2="{ypix:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left-10}" y="{ypix+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{text}">{yt:g}</text>')

    for xt in x_ticks:
        xpix = sx(xt)
        parts.append(f'<line x1="{xpix:.2f}" y1="{margin_top}" x2="{xpix:.2f}" y2="{height-margin_bottom}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{xpix:.2f}" y="{height-margin_bottom+20}" text-anchor="middle" font-family="Arial" font-size="12" fill="{text}">{xt:g}</text>')

    # Axes
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="{axis}" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="{axis}" stroke-width="2"/>')

    # Axis labels
    parts.append(f'<text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14" fill="{text}">{_escape(x_label)}</text>')
    parts.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" fill="{text}" transform="rotate(-90 24 {height/2:.1f})">{_escape(y_label)}</text>'
    )

    # Series paths
    for s in series:
        yv = np.asarray(s.y, dtype=float)
        if yv.size != x.size:
            raise ValueError(f"Series '{s.label}' length does not match x")
        pts = " ".join(f"{sx(float(xi)):.2f},{sy(float(yi)):.2f}" for xi, yi in zip(x, yv))
        parts.append(f'<polyline fill="none" stroke="{s.color}" stroke-width="2" points="{pts}"/>')

    # Legend
    legend_x = margin_left + 10
    legend_y = margin_top + 10
    box_w = 320
    box_h = 18 * max(1, len(series)) + 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{box_w}" height="{box_h}" fill="white" stroke="#BBB" stroke-width="1" opacity="0.95"/>')
    for i, s in enumerate(series):
        yy = legend_y + 22 + i * 18
        parts.append(f'<line x1="{legend_x+10}" y1="{yy-4}" x2="{legend_x+40}" y2="{yy-4}" stroke="{s.color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+50}" y="{yy}" font-family="Arial" font-size="12" fill="{text}">{_escape(s.label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def grouped_bar_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, np.ndarray, str]],
    overlay_line: Optional[Tuple[str, np.ndarray, str]] = None,
    width: int = 1000,
    height: int = 600,
    y_lim: Tuple[float, float] = (0.0, 1.0),
) -> str:
    # Layout
    margin_left, margin_right = 80, 20
    margin_top, margin_bottom = 60, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    k = len(categories)
    if k == 0:
        raise ValueError("categories must be non-empty")
    m = max(1, len(series))

    y_lo, y_hi = float(y_lim[0]), float(y_lim[1])

    def sy(yv: float) -> float:
        return margin_top + (1.0 - (yv - y_lo) / (y_hi - y_lo)) * plot_h

    axis = "#222"
    grid = "#D9D9D9"
    text = "#111"

    y_ticks = _nice_ticks(y_lo, y_hi, n=6)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="18" fill="{text}">{_escape(title)}</text>')

    # Grid + y ticks
    for yt in y_ticks:
        ypix = sy(yt)
        parts.append(f'<line x1="{margin_left}" y1="{ypix:.2f}" x2="{width-margin_right}" y2="{ypix:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left-10}" y="{ypix+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="{text}">{yt:g}</text>')

    # Axes
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="{axis}" stroke-width="2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="{axis}" stroke-width="2"/>')

    # Bar geometry
    group_w = plot_w / k
    bar_w = group_w * 0.8 / m
    group_pad = group_w * 0.2

    for ci in range(k):
        gx0 = margin_left + ci * group_w + group_pad / 2
        # x tick label
        cx = margin_left + (ci + 0.5) * group_w
        parts.append(
            f'<text x="{cx:.2f}" y="{height-margin_bottom+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="{text}">{_escape(str(categories[ci]))}</text>'
        )
        for si, (label, vals, color) in enumerate(series):
            v = float(vals[ci])
            x0 = gx0 + si * bar_w
            y0 = sy(v)
            h = (height - margin_bottom) - y0
            parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')

    # Overlay line (e.g., true probabilities)
    if overlay_line is not None:
        olabel, ovals, ocolor = overlay_line
        pts = []
        for ci in range(k):
            cx = margin_left + (ci + 0.5) * group_w
            pts.append(f"{cx:.2f},{sy(float(ovals[ci])):.2f}")
        parts.append(f'<polyline fill="none" stroke="{ocolor}" stroke-width="2" stroke-dasharray="6,4" points="{" ".join(pts)}"/>')

    # Axis labels
    parts.append(f'<text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14" fill="{text}">{_escape(x_label)}</text>')
    parts.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" fill="{text}" transform="rotate(-90 24 {height/2:.1f})">{_escape(y_label)}</text>'
    )

    # Legend
    legend_items = [(label, color) for (label, _, color) in series]
    if overlay_line is not None:
        legend_items.append((overlay_line[0], overlay_line[2]))

    legend_x = margin_left + 10
    legend_y = margin_top + 10
    box_w = 360
    box_h = 18 * max(1, len(legend_items)) + 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{box_w}" height="{box_h}" fill="white" stroke="#BBB" stroke-width="1" opacity="0.95"/>')
    for i, (lbl, col) in enumerate(legend_items):
        yy = legend_y + 22 + i * 18
        parts.append(f'<rect x="{legend_x+10}" y="{yy-12}" width="18" height="10" fill="{col}"/>')
        parts.append(f'<text x="{legend_x+35}" y="{yy-3}" font-family="Arial" font-size="12" fill="{text}">{_escape(lbl)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)

