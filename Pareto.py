import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import plotly.express as px


df = pd.read_excel("surikiuoti_duomenys.xlsx")

# ---- Pareto front (maximize Rate1, maximize Gylis, minimize Ra) ----
def pareto_mask_maxmaxmin(df: pd.DataFrame, rate_col="Rate1", gylis_col="Gylis", ra_col="Ra") -> np.ndarray:
    """
    Returns a boolean mask: True for Pareto-optimal (non-dominated) rows.
    Objectives: max(rate_col), max(gylis_col), min(ra_col).
    A dominates B if it is >= in Rate1 and Gylis, <= in Ra, and strictly better in at least one objective.
    """
    # Keep only rows with finite objective values
    obj = df[[rate_col, gylis_col, ra_col]].to_numpy(dtype=float)
    finite = np.isfinite(obj).all(axis=1)

    idx = np.where(finite)[0]
    obj = obj[finite]

    n = obj.shape[0]
    pareto = np.ones(n, dtype=bool)

    # For speed: split columns
    rate = obj[:, 0]
    gylis = obj[:, 1]
    ra = obj[:, 2]

    for i in range(n):
        if not pareto[i]:
            continue

        # Candidate i is dominated by any j that is:
        # rate[j] >= rate[i], gylis[j] >= gylis[i], ra[j] <= ra[i]
        # and at least one strict:
        # rate[j] > rate[i] or gylis[j] > gylis[i] or ra[j] < ra[i]
        ge_rate = rate >= rate[i]
        ge_gylis = gylis >= gylis[i]
        le_ra = ra <= ra[i]
        better_any = (rate > rate[i]) | (gylis > gylis[i]) | (ra < ra[i])

        dominated_by_any = np.any(ge_rate & ge_gylis & le_ra & better_any)
        if dominated_by_any:
            pareto[i] = False
            continue

        # If i is Pareto, it may dominate others -> you can optionally prune (not required)
        # But we can still mark dominated points as not Pareto to speed up:
        dominates_others = ge_rate & ge_gylis & le_ra & better_any  # (j dominates i) mask form, not needed here

    # Build full-length mask
    full_mask = np.zeros(len(df), dtype=bool)
    full_mask[idx] = pareto
    return full_mask


# ---- Compute Pareto and groups ----
mask = pareto_mask_maxmaxmin(df, rate_col="Rate1", gylis_col="Gylis", ra_col="Ra")
df_plot = df.copy()

p1n1 = (df_plot["P"] == 1) & (df_plot["N"] == 1)

df_plot["Group"] = np.where(
    mask & p1n1, "P=1 & N=1 (Pareto)",
    np.where(mask, "Pareto", "Other")
)

# If you want the hover to show EXACTLY these fields, ensure they exist:
hover_cols = ["N", "P", "Nscan", "F0", "Rate1", "Gylis", "Ra"]
missing = [c for c in hover_cols if c not in df_plot.columns]
if missing:
    raise KeyError(f"Missing columns in df: {missing}")

fig = px.scatter_3d(
    df_plot,
    x="Rate1",   # maximize
    y="Gylis",   # maximize
    z="Ra",      # minimize (lower is better)
    color="Group",
    color_discrete_map={
        "Pareto": "red",
        "Other": "blue",
        "P=1 & N=1 (Pareto)": "orange",
    },
    hover_data=hover_cols,  # shows all parameters on hover
    opacity=0.9
)

# Make it feel smooth / less laggy
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    legend_title_text="",
    scene=dict(
        xaxis_title="Rate",
        yaxis_title="Gylis",
        zaxis_title="Ra"
    ),
    margin=dict(l=0, r=0, t=30, b=0)
)
# Make ONLY the blue ("Other") points more transparent
fig.for_each_trace(
    lambda t: t.update(opacity=0.15) if t.name == "Other" else t.update(opacity=1.0)
)


fig.write_html("pareto_3d.html", include_plotlyjs="cdn")
fig.show()