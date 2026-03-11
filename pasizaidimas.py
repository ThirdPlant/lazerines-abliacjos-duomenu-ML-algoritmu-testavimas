import matplotlib.pyplot as plt
import pandas as pd

FILE_PATH = "surikiuoti_duomenys_Nscan_3.xlsx"
N_VALUE = 1.0
REQUIRED_COLUMNS = ["N", "P", "F0", "Gylis"]


def main() -> None:
    df = pd.read_excel(FILE_PATH)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faile truksta stulpeliu: {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    df = df[df["N"].astype(float) == N_VALUE].copy()

    if df.empty:
        raise ValueError(f"Nerasta eiluciu su N == {N_VALUE}.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(
        df["F0"],
        df["Gylis"],
        c=df["P"],
        cmap="viridis",
        s=42,
        alpha=0.9,
        edgecolors="none",
    )
    ax.set_xlabel("F0")
    ax.set_ylabel("Gylis")
    ax.set_title(f"N = {N_VALUE:g}: Gylis vs F0 (spalva = P)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("P")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
