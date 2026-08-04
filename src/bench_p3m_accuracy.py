"""Measure P3M force error, speed, and divergence from brute force.

Unlike the visual benchmark sweep, every solver receives exactly the same
frozen snapshot.  Small snapshots are also checked against a float64 CPU
oracle.  The drift test then advances race-free brute-force and P3M simulations
from identical initial state and records both particle-wise and coarse visual
distribution error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import glfw
import moderngl
import numpy as np

from p3m_core import (
    PARTICLE_DTYPE,
    BruteForceSolver,
    GpuIntegrator,
    P3MForceSolver,
    make_particles,
)


SOFTENING = 0.02
FORCE_FALLOFF = 2.0
SAME_REPEL = 1.0
OTHER_ATTRACT = 1.001
WORLD_BOUNDS = 1.0
DT = 1.0 / 90.0
DRAG = 1.0
MAX_SPEED = 2.0


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def make_context():
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(32, 32, "P3M benchmark", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 4.3 context")
    glfw.make_context_current(window)
    return window, moderngl.create_context()


def timed_ms(ctx, callback, warmups=2, repeats=5):
    for _ in range(warmups):
        callback()
    values = []
    for _ in range(repeats):
        query = ctx.query(time=True)
        with query:
            callback()
        values.append(query.elapsed / 1e6)
    return float(np.median(values))


def cpu_force_oracle(particles, chunk=64):
    """Direct float64 force sum without allocating the full NxN matrix."""
    pos = particles["pos"].astype(np.float64)
    types = particles["type"].astype(np.int32)
    n = len(particles)
    result = np.empty((n, 2), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = pos[None, :, :] - pos[start:stop, None, :]
        geometric_r2 = np.einsum("ijk,ijk->ij", d, d)
        softened_r2 = geometric_r2 + SOFTENING
        kernel = softened_r2 ** (-(FORCE_FALLOFF + 0.5))
        same = types[start:stop, None] == types[None, :]
        coefficient = np.where(same, -SAME_REPEL, OTHER_ATTRACT)
        local_rows = np.arange(stop - start)
        coefficient[local_rows, np.arange(start, stop)] = 0.0
        result[start:stop] = np.sum(d * (kernel * coefficient)[:, :, None], axis=1)
    return result


def error_metrics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    delta_mag = np.linalg.norm(candidate - reference, axis=1)
    ref_rms = float(np.sqrt(np.mean(np.sum(reference * reference, axis=1))))
    rel_l2 = float(np.linalg.norm(candidate - reference) /
                   max(np.linalg.norm(reference), 1e-30))
    return {
        "relative_l2_pct": 100.0 * rel_l2,
        "p95_error_over_ref_rms_pct": 100.0 * float(np.percentile(delta_mag, 95)) / max(ref_rms, 1e-30),
        "max_error_over_ref_rms_pct": 100.0 * float(np.max(delta_mag)) / max(ref_rms, 1e-30),
    }


def occupancy_tv_percent(a, b, bins=32):
    scores = []
    for particle_type in (0, 1):
        pa = a["pos"][a["type"] == particle_type]
        pb = b["pos"][b["type"] == particle_type]
        ha, _, _ = np.histogram2d(pa[:, 0], pa[:, 1], bins=bins,
                                  range=[[-WORLD_BOUNDS, WORLD_BOUNDS]] * 2)
        hb, _, _ = np.histogram2d(pb[:, 0], pb[:, 1], bins=bins,
                                  range=[[-WORLD_BOUNDS, WORLD_BOUNDS]] * 2)
        ha /= max(ha.sum(), 1.0)
        hb /= max(hb.sum(), 1.0)
        scores.append(0.5 * np.abs(ha - hb).sum())
    return 100.0 * float(np.mean(scores))


def read_particles(buffer, n):
    return np.frombuffer(buffer.read(size=n * PARTICLE_DTYPE.itemsize),
                         dtype=PARTICLE_DTYPE).copy()


def run_accuracy(ctx, mesh_resolutions, n, seed):
    particles = make_particles(n, seed=seed)
    particle_buffer = ctx.buffer(particles.tobytes())
    cpu_reference = cpu_force_oracle(particles)
    brute = BruteForceSolver(ctx, n, SOFTENING, FORCE_FALLOFF,
                             SAME_REPEL, OTHER_ATTRACT)
    brute_ms = timed_ms(ctx, lambda: brute.compute(particle_buffer, n), repeats=3)
    brute.compute(particle_buffer, n)
    brute_force = brute.read_acceleration(n)
    rows = []
    brute_error = error_metrics(brute_force, cpu_reference)
    rows.append({
        "solver": "GPU brute force (float32)", "N": n, "mesh_res": 0,
        "compute_ms": f"{brute_ms:.4f}", "speedup_vs_brute": "1.000",
        **{key: f"{value:.6f}" for key, value in brute_error.items()},
    })
    for mesh_res in mesh_resolutions:
        solver = P3MForceSolver(ctx, n, mesh_res, WORLD_BOUNDS, SOFTENING,
                                FORCE_FALLOFF, SAME_REPEL, OTHER_ATTRACT)
        p3m_ms = timed_ms(ctx, lambda: solver.compute(particle_buffer, n))
        solver.compute(particle_buffer, n)
        p3m_force = solver.read_acceleration(n)
        metrics = error_metrics(p3m_force, cpu_reference)
        rows.append({
            "solver": f"P3M {mesh_res}x{mesh_res}", "N": n,
            "mesh_res": mesh_res, "compute_ms": f"{p3m_ms:.4f}",
            "speedup_vs_brute": f"{brute_ms / p3m_ms:.3f}",
            **{key: f"{value:.6f}" for key, value in metrics.items()},
        })
    return rows


def run_speed(ctx, mesh_res, counts, seed):
    capacity = max(counts)
    snapshot = make_particles(capacity, seed=seed, extent=WORLD_BOUNDS)
    particle_buffer = ctx.buffer(snapshot.tobytes())
    brute = BruteForceSolver(ctx, capacity, SOFTENING, FORCE_FALLOFF,
                             SAME_REPEL, OTHER_ATTRACT)
    p3m = P3MForceSolver(ctx, capacity, mesh_res, WORLD_BOUNDS, SOFTENING,
                         FORCE_FALLOFF, SAME_REPEL, OTHER_ATTRACT)
    rows = []
    for n in counts:
        brute_repeats = 3 if n <= 100_000 else 2
        brute_ms = timed_ms(ctx, lambda: brute.compute(particle_buffer, n),
                            warmups=1, repeats=brute_repeats)
        p3m_ms = timed_ms(ctx, lambda: p3m.compute(particle_buffer, n),
                          warmups=2, repeats=5)
        rows.append({
            "N": n, "mesh_res": mesh_res,
            "brute_compute_ms": f"{brute_ms:.4f}",
            "p3m_compute_ms": f"{p3m_ms:.4f}",
            "speedup": f"{brute_ms / p3m_ms:.3f}",
            "p3m_force_fps": f"{1000.0 / p3m_ms:.2f}",
        })
        print(f"speed N={n}: brute={brute_ms:.3f} ms, P3M={p3m_ms:.3f} ms, "
              f"speedup={brute_ms / p3m_ms:.1f}x")
    return rows


def run_drift(ctx, mesh_res, n, steps, seed):
    initial = make_particles(n, seed=seed)
    brute_a = ctx.buffer(initial.tobytes())
    brute_b = ctx.buffer(reserve=initial.nbytes)
    p3m_a = ctx.buffer(initial.tobytes())
    p3m_b = ctx.buffer(reserve=initial.nbytes)
    brute = BruteForceSolver(ctx, n, SOFTENING, FORCE_FALLOFF,
                             SAME_REPEL, OTHER_ATTRACT)
    p3m = P3MForceSolver(ctx, n, mesh_res, WORLD_BOUNDS, SOFTENING,
                         FORCE_FALLOFF, SAME_REPEL, OTHER_ATTRACT)
    integrator = GpuIntegrator(ctx, DT, DRAG, MAX_SPEED, WORLD_BOUNDS)
    checkpoints = sorted(set([0, 1, 10, 30, 100, steps]))
    checkpoints = [value for value in checkpoints if value <= steps]
    rows = []

    def record(step):
        brute_state = read_particles(brute_a, n)
        p3m_state = read_particles(p3m_a, n)
        pos_delta = p3m_state["pos"].astype(np.float64) - brute_state["pos"].astype(np.float64)
        vel_delta = p3m_state["vel"].astype(np.float64) - brute_state["vel"].astype(np.float64)
        pos_rmse = float(np.sqrt(np.mean(np.sum(pos_delta * pos_delta, axis=1))))
        vel_rmse = float(np.sqrt(np.mean(np.sum(vel_delta * vel_delta, axis=1))))
        centroid_error = float(np.linalg.norm(
            p3m_state["pos"].mean(axis=0) - brute_state["pos"].mean(axis=0)))
        rows.append({
            "step": step, "simulated_seconds": f"{step * DT:.6f}", "N": n,
            "mesh_res": mesh_res, "position_rmse": f"{pos_rmse:.8f}",
            "position_rmse_pct_world_width": f"{100.0 * pos_rmse / (2.0 * WORLD_BOUNDS):.6f}",
            "velocity_rmse": f"{vel_rmse:.8f}",
            "centroid_error": f"{centroid_error:.8f}",
            "occupancy_tv_pct": f"{occupancy_tv_percent(p3m_state, brute_state):.6f}",
        })

    record(0)
    for step in range(1, steps + 1):
        brute.compute(brute_a, n)
        integrator.step(brute_a, brute.acceleration, brute_b, n)
        brute_a, brute_b = brute_b, brute_a
        p3m.compute(p3m_a, n)
        integrator.step(p3m_a, p3m.acceleration, p3m_b, n)
        p3m_a, p3m_b = p3m_b, p3m_a
        if step in checkpoints:
            record(step)
            print(f"drift checkpoint {step}/{steps}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-res", nargs="+", type=int, default=[256, 512],
                        help="Power-of-two meshes used for the accuracy tradeoff")
    parser.add_argument("--reference-n", type=int, default=4096,
                        help="N used for the float64 CPU accuracy check")
    parser.add_argument("--speed-n", nargs="+", type=int,
                        default=[20_000, 50_000, 100_000, 200_000])
    parser.add_argument("--drift-n", type=int, default=2048)
    parser.add_argument("--drift-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    window, ctx = make_context()
    try:
        print(f"[GPU] {ctx.info['GL_RENDERER']} | {ctx.info['GL_VERSION']}")
        accuracy = run_accuracy(ctx, args.mesh_res, args.reference_n, args.seed)
        speed = run_speed(ctx, max(args.mesh_res), args.speed_n, args.seed + 1)
        drift = run_drift(ctx, max(args.mesh_res), args.drift_n,
                          args.drift_steps, args.seed + 2)
        write_csv(out_dir / "p3m_accuracy.csv", accuracy)
        write_csv(out_dir / "p3m_speed.csv", speed)
        write_csv(out_dir / "p3m_drift.csv", drift)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
