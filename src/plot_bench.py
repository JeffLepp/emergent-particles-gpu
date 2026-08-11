"""
Plot the csvs written by the --bench modes.

    python plot_bench.py --csv gpu_grid=gpu_grid.csv --csv cpu_pygame=cpu_pygame.csv --out bench.png

Any csv with an "N" column plus the chosen --y column works, so this handles the
GPU csvs (compute_ms/frame_ms/fps) and the CPU ones (avg_ms/fps) the same way.
Saves to --out if given, otherwise opens a window.
"""
import csv
import argparse


def read_csv(path, ycol):
    xs, ys = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get(ycol)
            if not v:
                continue
            xs.append(int(row["N"]))
            ys.append(float(v))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, metavar="LABEL=PATH",
                    help="Series to plot, repeatable (e.g. --csv gpu_grid=gpu_grid.csv)")
    ap.add_argument("--y", default="fps", choices=["fps", "compute_ms", "frame_ms", "avg_ms"],
                    help="Column to plot on the y axis")
    ap.add_argument("--logx", action="store_true", help="Log-scale the x axis")
    ap.add_argument("--logy", action="store_true", help="Log-scale the y axis")
    ap.add_argument("--out", default="", help="Save a png here instead of opening a window")
    ap.add_argument("--budget", action="store_true",
                    help="Draw 60/30 FPS frame-budget lines (ms y-axes only)")
    args = ap.parse_args()

    # Saving a graph should not require Tk or a visible desktop session.
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    plotted = 0
    for spec in args.csv:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--csv wants LABEL=PATH, got {spec!r}")
        xs, ys = read_csv(path, args.y)
        if not xs:
            print(f"skipping {label}: no '{args.y}' data in {path}")
            continue
        plt.plot(xs, ys, marker="o", markersize=3, label=label)
        plotted += 1

    if not plotted:
        raise SystemExit(f"no csv had a '{args.y}' column")

    if args.budget and args.y.endswith("_ms"):
        for ms, name in ((1000 / 60, "60 FPS budget"), (1000 / 30, "30 FPS budget")):
            plt.axhline(ms, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            plt.annotate(name, (0.01, ms), xycoords=("axes fraction", "data"),
                         va="bottom", fontsize=8, color="gray")

    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")
    plt.xlabel("Particle count (N)")
    plt.ylabel(args.y)
    plt.title(f"Particle sim: {args.y} vs N")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
