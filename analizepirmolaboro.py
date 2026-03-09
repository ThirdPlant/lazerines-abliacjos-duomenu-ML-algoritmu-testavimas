import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def load_measurements(file_path: str) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        header = f.read(8)

    # Real .xls files start with D0 CF 11 E0 A1 B1 1A E1.
    # This file is usually TSV text with .xls extension.
    if header.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raw = pd.read_excel(file_path, sheet_name=0, engine="xlrd")
    else:
        raw = pd.read_csv(file_path, sep="\t")

    raw = raw.rename(columns=lambda c: str(c).strip())
    df = raw.loc[:, ["No", "Time", "Value"]].copy()

    df["No"] = pd.to_numeric(df["No"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")

    df = df.dropna(subset=["No", "Value"]).reset_index(drop=True)

    if df["time_dt"].notna().all():
        # If a measurement passes midnight, keep elapsed time monotonic.
        day_jumps = (df["time_dt"].diff().dt.total_seconds() < 0).cumsum()
        df["time_dt"] = df["time_dt"] + pd.to_timedelta(day_jumps, unit="D")
        df["elapsed_s"] = (df["time_dt"] - df["time_dt"].iloc[0]).dt.total_seconds()
    else:
        # Fallback if time parsing fails: use measurement index.
        df["elapsed_s"] = df["No"] - df["No"].iloc[0]

    return df


def fit_tangent(x: np.ndarray, y: np.ndarray, x0: float, window_s: float = 35.0) -> tuple[float, float, float]:
    y0 = float(np.interp(x0, x, y))
    mask = np.abs(x - x0) <= window_s
    idx = np.where(mask)[0]

    if idx.size < 2:
        idx = np.argsort(np.abs(x - x0))[:8]

    slope, _ = np.polyfit(x[idx], y[idx], 1)
    intercept = y0 - slope * x0
    return slope, intercept, y0


def intersect_lines(line_a: dict, line_b: dict) -> tuple[float, float] | None:
    if line_a["kind"] == "vertical" and line_b["kind"] == "vertical":
        return None

    if line_a["kind"] == "vertical":
        x_val = line_a["x"]
        y_val = line_b["m"] * x_val + line_b["c"]
        return float(x_val), float(y_val)

    if line_b["kind"] == "vertical":
        x_val = line_b["x"]
        y_val = line_a["m"] * x_val + line_a["c"]
        return float(x_val), float(y_val)

    m1, c1 = line_a["m"], line_a["c"]
    m2, c2 = line_b["m"], line_b["c"]
    if np.isclose(m1, m2):
        return None

    x_val = (c2 - c1) / (m1 - m2)
    y_val = m1 * x_val + c1
    return float(x_val), float(y_val)


file_path = "silumos_eksperimentas2.xls"
df = load_measurements(file_path)

# Use unique x values for interpolation/regression.
series = (
    df.groupby("elapsed_s", as_index=False)["Value"]
    .mean()
    .sort_values("elapsed_s")
)
x = series["elapsed_s"].to_numpy(dtype=float)
y = series["Value"].to_numpy(dtype=float)

x_min, x_max = float(x.min()), float(x.max())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["elapsed_s"], df["Value"], lw=1.2, alpha=0.75, label=r"$T(t), ^{\circ}C$")

x_draw = np.linspace(max(x_min, 540.0), min(x_max, 980.0), 350)

# Tangents at x = 600 and x = 700.
m_600, c_600, _ = fit_tangent(x, y, 600.0)
m_700, c_700, _ = fit_tangent(x, y, 700.0)
y_line_600 = m_600 * x_draw + c_600
y_line_700 = m_700 * x_draw + c_700
ax.plot(x_draw, y_line_600, "--", lw=1.4)
ax.plot(x_draw, y_line_700, "--", lw=1.4)

# Horizontal tangent at x = 870.
_, _, y_870 = fit_tangent(x, y, 870.0)
ax.plot(x_draw, np.full_like(x_draw, y_870), "--", lw=1.4)

# Vertical reference lines.
ax.axvline(815, linestyle="--", lw=1.4)
ax.axvline(923, linestyle="--", lw=1.4)
ax.axvline(590, linestyle="-", lw=1.6, color="black")

# Intersections of dashed lines + x=590 line.
reference_lines = [
    {"name": "liestine_600", "kind": "line", "m": m_600, "c": c_600},
    {"name": "liestine_700", "kind": "line", "m": m_700, "c": c_700},
    {"name": "horizontali_870", "kind": "line", "m": 0.0, "c": y_870},
    {"name": "vertikali_815", "kind": "vertical", "x": 815.0},
    {"name": "vertikali_923", "kind": "vertical", "x": 923.0},
    {"name": "vertikali_590", "kind": "vertical", "x": 590.0},
]

intersections: list[tuple[float, float]] = []
for line_a, line_b in combinations(reference_lines, 2):
    point = intersect_lines(line_a, line_b)
    if point is None:
        continue
    intersections.append(point)

# Deduplicate near-identical intersections.
unique_intersections: list[tuple[float, float]] = []
for x_i, y_i in intersections:
    if not any(np.isclose(x_i, ux, atol=0.15) and np.isclose(y_i, uy, atol=0.15) for ux, uy in unique_intersections):
        unique_intersections.append((x_i, y_i))

for idx, (x_i, y_i) in enumerate(unique_intersections):
    ax.scatter(x_i, y_i, color="red", s=22, zorder=5)
    y_offset = 8 if idx % 2 == 0 else -14
    ax.annotate(f"({x_i:.1f}, {y_i:.1f})", (x_i, y_i), textcoords="offset points", xytext=(6, y_offset), fontsize=8)

y_candidates = list(y) + [p[1] for p in unique_intersections]
ax.set_xlim(x_min, x_max)
ax.set_ylim(min(y_candidates) - 10, max(y_candidates) + 10)

ax.set_xlabel("t, s")
ax.set_ylabel(r"$T, ^{\circ}C$")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
