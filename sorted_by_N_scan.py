import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CONFIG: choose Nscan here
# =========================
NSCAN_VALUE = 21   # <-- change this to the Nscan you want to display

df = pd.read_excel("surikiuoti_duomenys.xlsx")

# ---- Pareto front (maximize Rate1, maximize Gylis, minimize Ra) ----
def pareto_mask_maxmaxmin(df: pd.DataFrame, rate_col="Rate1", gylis_col="Gylis", ra_col="Ra") -> np.ndarray:
    """
    Returns a boolean mask: True for Pareto-optimal (non-dominated) rows.
    Objectives: max(rate_col), max(gylis_col), min(ra_col).
    A dominates B if it is >= in Rate1 and Gylis, <= in Ra, and strictly better in at least one objective.
    """
    obj = df[[rate_col, gylis_col, ra_col]].to_numpy(dtype=float)
    finite = np.isfinite(obj).all(axis=1)

    idx = np.where(finite)[0]
    obj = obj[finite]

    n = obj.shape[0]
    pareto = np.ones(n, dtype=bool)

    rate = obj[:, 0]
    gylis = obj[:, 1]
    ra = obj[:, 2]

    for i in range(n):
        if not pareto[i]:
            continue

        ge_rate = rate >= rate[i]
        ge_gylis = gylis >= gylis[i]
        le_ra = ra <= ra[i]
        better_any = (rate > rate[i]) | (gylis > gylis[i]) | (ra < ra[i])

        if np.any(ge_rate & ge_gylis & le_ra & better_any):
            pareto[i] = False

    full_mask = np.zeros(len(df), dtype=bool)
    full_mask[idx] = pareto
    return full_mask


# =========================
# Filter to one Nscan
# =========================
df = df[df["Nscan"] == NSCAN_VALUE].copy()

if df.empty:
    raise ValueError(f"No rows found with Nscan == {NSCAN_VALUE}")

# ---- Compute Pareto and groups (within this Nscan subset) ----
mask = pareto_mask_maxmaxmin(df, rate_col="Rate1", gylis_col="Gylis", ra_col="Ra")
df_plot = df.copy()

p1n1 = (df_plot["P"] == 1) & (df_plot["N"] == 1)

df_plot["Group"] = np.where(
    mask & p1n1, "P=1 & N=1 (Pareto)",
    np.where(mask, "Pareto", "Other")
)

# Hover fields
hover_cols = ["N", "P", "Nscan", "F0", "Rate1", "Gylis", "Ra"]
missing = [c for c in hover_cols if c not in df_plot.columns]
if missing:
    raise KeyError(f"Missing columns in df: {missing}")

fig = px.scatter_3d(
    df_plot,
    x="Rate1",
    y="Gylis",
    z="Ra",
    color="Group",
    color_discrete_map={
        "Pareto": "red",
        "Other": "blue",
        "P=1 & N=1 (Pareto)": "orange",
    },
    hover_data=hover_cols,
)

# Styling
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    title=f"Pareto (Nscan = {NSCAN_VALUE})",
    legend_title_text="",
    scene=dict(xaxis_title="Rate", yaxis_title="Gylis", zaxis_title="Ra"),
    margin=dict(l=0, r=0, t=40, b=0),
)

# Blue points lower opacity
fig.for_each_trace(
    lambda t: t.update(opacity=0.15) if t.name == "Other" else t.update(opacity=1.0)
)

#fig.write_html("pareto_3d.html", include_plotlyjs="cdn")
#fig.write_html(f"pareto_3d_Nscan_{NSCAN_VALUE}.html", include_plotlyjs="cdn")
fig.show()