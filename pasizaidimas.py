import itertools

import matplotlib.pyplot as plt
import pandas as pd

INPUT_COLUMNS = ["N", "P", "F0"]
OUTPUT_COLUMNS = ["Gylis", "Ra"]
NSCAN_VALUE = 3


def main() -> None:
    df = pd.read_excel("surikiuoti_duomenys.xlsx")
    df = df[df["Nscan"] == NSCAN_VALUE].copy()
    df = df.dropna(subset=INPUT_COLUMNS + OUTPUT_COLUMNS).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows left after filtering Nscan == {NSCAN_VALUE}.")

    input_pairs = list(itertools.combinations(INPUT_COLUMNS, 2))

    fig = plt.figure(figsize=(18, 10))
    plot_index = 1

    for output_col in OUTPUT_COLUMNS:
        for x_col, y_col in input_pairs:
            ax = fig.add_subplot(
                len(OUTPUT_COLUMNS),
                len(input_pairs),
                plot_index,
                projection="3d",
            )
            scatter = ax.scatter(
                df[x_col],
                df[y_col],
                df[output_col],
                c=df[output_col],
                cmap="viridis",
                s=20,
                alpha=0.85,
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(output_col)
            #ax.set_title(f"{output_col} vs {x_col}, {y_col}")
            fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08, label=output_col)
            plot_index += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
