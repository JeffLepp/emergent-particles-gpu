import csv
import argparse
import matplotlib.pyplot as plt
import math

def read_csv(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["N"]))
            ys.append(float(row["fps"]))
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, help="ModernGL CPU csv")
    ap.add_argument("--pygame", required=True, help="Pygame CPU csv")
    ap.add_argument("--out", default="", help="Optional png output path")
    args = ap.parse_args()

    x1, y1 = read_csv(args.cpu)
    x2, y2 = read_csv(args.pygame)

    plt.figure()
    plt.plot(x1, y1, marker="o", label="ModernGL (CPU physics + GPU render)")
    plt.plot(x2, y2, marker="o", label="Pygame (CPU physics + CPU draw)")
    plt.xlabel("Particle count (N)")
    plt.ylabel("Average fps (sqrt scale)")
    plt.title("CPU vs Pygame sweep benchmark")
    plt.legend()
    plt.grid(True)

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
