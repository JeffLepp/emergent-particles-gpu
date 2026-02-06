"""
CPU Two-Type Particles (Pygame)
--------------------------------
Same general rules + look as your ModernGL version, but rendered with pygame.draw.circle.

Keys:
- ESC = quit
- SPACE = add particles (random, no reset)
"""

import math
import numpy as np
import pygame
import argparse
import json
import csv
import time

# ----------------------------
# Configurations
# ----------------------------
W, H = 900, 700

N_PER_TYPE = 10
ADD_PER_SPACE = 10

DT = 1.0 / 60.0
SOFTENING = 0.10
DRAG = 0.95
MAX_SPEED = 1.5

SAME_REPEL = 0.6999
OTHER_ATTRACT = 0.7
FORCE_FALLOFF = 2.0

WORLD_BOUNDS = 1.0

# "point sprite" feel
POINT_SIZE = 2  # pixels radius-ish (pygame uses integer radius)
ALPHA_SOFT_EDGE = True  # set False for faster, hard circles

# CPU neighbor grid params
CELL_SIZE = 0.08
NEIGHBOR_RADIUS = 0.16
MAX_NEIGHBORS_PER_PARTICLE = 256  # cap work per particle (None = no cap)

# Colors (approx from your shader)
COLOR_A = (255, 64, 64)
COLOR_B = (64, 140, 255)
BG = (8, 8, 10)

# ----------------------------
# Benchmarking utilities
# ----------------------------
def load_bench_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="Run automated sweep benchmark and exit")
    ap.add_argument("--config", default="bench_config.json", help="Path to benchmark config json")
    return ap.parse_args()

def bench_write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "N", "avg_ms", "fps"])
        w.writeheader()
        w.writerows(rows)

# ----------------------------
# Helpers: world <-> screen
# ----------------------------
def world_to_screen(x, y):
    # world in [-1,1] => screen
    sx = int((x * 0.5 + 0.5) * W)
    sy = int(((-y) * 0.5 + 0.5) * H)
    return sx, sy

def clamp_speed(vx, vy, max_s):
    s2 = vx * vx + vy * vy
    m2 = max_s * max_s
    mask = s2 > m2
    if np.any(mask):
        s = np.sqrt(s2[mask])
        scale = (max_s / s)
        vx[mask] *= scale
        vy[mask] *= scale
    return vx, vy

def bounce_bounds(x, y, vx, vy, bounds):
    mask = x < -bounds
    if np.any(mask):
        x[mask] = -bounds
        vx[mask] *= -0.9
    mask = x > bounds
    if np.any(mask):
        x[mask] = bounds
        vx[mask] *= -0.9

    mask = y < -bounds
    if np.any(mask):
        y[mask] = -bounds
        vy[mask] *= -0.9
    mask = y > bounds
    if np.any(mask):
        y[mask] = bounds
        vy[mask] *= -0.9

    return x, y, vx, vy

# ----------------------------
# Physics: uniform grid neighbor search
# ----------------------------
def cpu_step_grid(x, y, vx, vy, t,
                  dt, soft, drag, max_speed,
                  same_repel, other_attract, force_falloff,
                  cell_size, neighbor_radius, max_neighbors):

    N = x.shape[0]
    inv_cell = 1.0 / cell_size

    # grid dims to cover [-WORLD_BOUNDS, WORLD_BOUNDS]
    grid_w = int(math.ceil((2.0 * WORLD_BOUNDS) * inv_cell)) + 2
    grid_h = grid_w

    gx = np.floor((x + WORLD_BOUNDS) * inv_cell).astype(np.int32)
    gy = np.floor((y + WORLD_BOUNDS) * inv_cell).astype(np.int32)
    gx = np.clip(gx, 0, grid_w - 1)
    gy = np.clip(gy, 0, grid_h - 1)

    cell_id = gx + gy * grid_w

    order = np.argsort(cell_id, kind="mergesort")
    cell_sorted = cell_id[order]

    counts = np.bincount(cell_sorted, minlength=grid_w * grid_h)
    starts = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=starts[1:])
    ends = starts + counts

    ax = np.zeros(N, dtype=np.float32)
    ay = np.zeros(N, dtype=np.float32)

    r2_max = neighbor_radius * neighbor_radius

    for i in range(N):
        ci = cell_id[i]
        cix = ci % grid_w
        ciy = ci // grid_w

        acc_x = 0.0
        acc_y = 0.0
        checked = 0

        for oy in (-1, 0, 1):
            ny = ciy + oy
            if ny < 0 or ny >= grid_h:
                continue
            for ox in (-1, 0, 1):
                nx = cix + ox
                if nx < 0 or nx >= grid_w:
                    continue

                cj = nx + ny * grid_w
                s = starts[cj]
                e = ends[cj]
                if s == e:
                    continue

                idxs = order[s:e]

                if max_neighbors is not None:
                    remaining = max_neighbors - checked
                    if remaining <= 0:
                        break
                    if idxs.size > remaining:
                        idxs = idxs[:remaining]

                dx = x[idxs] - x[i]
                dy = y[idxs] - y[i]

                # remove self
                if idxs.size:
                    mself = (idxs != i)
                    dx = dx[mself]
                    dy = dy[mself]
                    idxs = idxs[mself]
                    if idxs.size == 0:
                        continue

                r2 = dx * dx + dy * dy
                m = r2 <= r2_max
                if not np.any(m):
                    continue

                dx = dx[m]
                dy = dy[m]
                idxs2 = idxs[m]
                r2 = r2[m] + soft

                invr = 1.0 / np.sqrt(r2)
                dirx = dx * invr
                diry = dy * invr

                base = 1.0 / r2
                mag = base ** force_falloff

                same = (t[idxs2] == t[i])
                if np.any(same):
                    s_mag = same_repel * mag[same]
                    acc_x -= float(np.sum(dirx[same] * s_mag))
                    acc_y -= float(np.sum(diry[same] * s_mag))

                other = ~same
                if np.any(other):
                    o_mag = other_attract * mag[other]
                    acc_x += float(np.sum(dirx[other] * o_mag))
                    acc_y += float(np.sum(diry[other] * o_mag))

                checked += idxs2.size

            if max_neighbors is not None and checked >= max_neighbors:
                break

        ax[i] = acc_x
        ay[i] = acc_y

    vx += ax * dt
    vy += ay * dt
    vx *= drag
    vy *= drag
    vx, vy = clamp_speed(vx, vy, max_speed)

    x += vx * dt
    y += vy * dt
    x, y, vx, vy = bounce_bounds(x, y, vx, vy, WORLD_BOUNDS)

# ----------------------------
# Drawing: soft point sprite
# ----------------------------
_soft_cache = {}

def soft_circle_surface(radius, color):
    key = (radius, color)
    surf = _soft_cache.get(key)
    if surf is not None:
        return surf

    size = radius * 2 + 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2

    # radial alpha falloff
    for py in range(size):
        for px in range(size):
            dx = px - cx
            dy = py - cy
            r2 = dx*dx + dy*dy
            if r2 > radius*radius:
                continue
            r = math.sqrt(r2) / max(1e-6, radius)
            # similar “soft edge” feel
            a = int(255 * (1.0 - min(1.0, max(0.0, (r - 0.7) / 0.3))))
            s.set_at((px, py), (color[0], color[1], color[2], a))

    _soft_cache[key] = s
    return s

# ----------------------------
# Main
# ----------------------------
def main():
    
    args = parse_args()
    bench = args.bench
    cfg = load_bench_config(args.config) if bench else None

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CPU Two-Type Particles (Pygame)")
    clock = pygame.time.Clock()

    rng = np.random.default_rng(cfg["seed"] if bench else 1)

    # initialize
    N = 2 * N_PER_TYPE
    x = np.zeros(N, dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)
    vx = np.zeros(N, dtype=np.float32)
    vy = np.zeros(N, dtype=np.float32)
    t = np.zeros(N, dtype=np.int32)

    # A on left
    x[:N_PER_TYPE] = (-0.5 + rng.uniform(-0.3, 0.3, N_PER_TYPE)).astype(np.float32)
    y[:N_PER_TYPE] = (0.0 + rng.uniform(-0.3, 0.3, N_PER_TYPE)).astype(np.float32)
    vx[:N_PER_TYPE] = rng.uniform(-0.1, 0.1, N_PER_TYPE).astype(np.float32)
    vy[:N_PER_TYPE] = rng.uniform(-0.1, 0.1, N_PER_TYPE).astype(np.float32)
    t[:N_PER_TYPE] = 0

    # B on right
    x[N_PER_TYPE:] = (0.5 + rng.uniform(-0.3, 0.3, N_PER_TYPE)).astype(np.float32)
    y[N_PER_TYPE:] = (0.0 + rng.uniform(-0.3, 0.3, N_PER_TYPE)).astype(np.float32)
    vx[N_PER_TYPE:] = rng.uniform(-0.1, 0.1, N_PER_TYPE).astype(np.float32)
    vy[N_PER_TYPE:] = rng.uniform(-0.1, 0.1, N_PER_TYPE).astype(np.float32)
    t[N_PER_TYPE:] = 1

    running = True
    space_was_down = False

    ema_ms = None
    accum_print = 0.0

    # prebuild sprite surfaces
    sprite_a = soft_circle_surface(POINT_SIZE, COLOR_A) if ALPHA_SOFT_EDGE else None
    sprite_b = soft_circle_surface(POINT_SIZE, COLOR_B) if ALPHA_SOFT_EDGE else None

    if bench:
        start_n = int(cfg["start_n"])
        end_n   = int(cfg["end_n"])
        step_n  = int(cfg["step_n"])
        warm_s  = float(cfg["warmup_seconds"])
        samp_s  = float(cfg["sample_seconds"])
        out_csv = str(cfg.get("out_csv", "cpu_pygame.csv"))

        # Force initial N to exactly start_n
        cur_n = x.shape[0]
        if cur_n != start_n:
            if cur_n > start_n:
                x = x[:start_n]; y = y[:start_n]; vx = vx[:start_n]; vy = vy[:start_n]; t = t[:start_n]
            else:
                add = start_n - cur_n
                x = np.concatenate([x, rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)])
                y = np.concatenate([y, rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)])
                vx = np.concatenate([vx, rng.uniform(-0.1, 0.1, add).astype(np.float32)])
                vy = np.concatenate([vy, rng.uniform(-0.1, 0.1, add).astype(np.float32)])
                t = np.concatenate([t, rng.integers(0, 2, size=add, dtype=np.int32)])

        rows = []
        target = start_n

        def one_frame():
            t0 = time.perf_counter()

            cpu_step_grid(
                x, y, vx, vy, t,
                DT, SOFTENING, DRAG, MAX_SPEED,
                SAME_REPEL, OTHER_ATTRACT, FORCE_FALLOFF,
                CELL_SIZE, NEIGHBOR_RADIUS, MAX_NEIGHBORS_PER_PARTICLE,
            )

            screen.fill(BG)
            if ALPHA_SOFT_EDGE:
                ra = sprite_a.get_width() // 2
                rb = sprite_b.get_width() // 2
                for i in range(x.shape[0]):
                    sx, sy = world_to_screen(x[i], y[i])
                    if t[i] == 0:
                        screen.blit(sprite_a, (sx - ra, sy - ra))
                    else:
                        screen.blit(sprite_b, (sx - rb, sy - rb))
            else:
                for i in range(x.shape[0]):
                    sx, sy = world_to_screen(x[i], y[i])
                    col = COLOR_A if t[i] == 0 else COLOR_B
                    pygame.draw.circle(screen, col, (sx, sy), POINT_SIZE)

            pygame.display.flip()
            return (time.perf_counter() - t0) * 1000.0

        running = True
        while running and target <= end_n:
            # process quit/esc
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

            # grow to target
            cur_n = x.shape[0]
            if cur_n < target:
                add = target - cur_n
                x = np.concatenate([x, rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)])
                y = np.concatenate([y, rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)])
                vx = np.concatenate([vx, rng.uniform(-0.1, 0.1, add).astype(np.float32)])
                vy = np.concatenate([vy, rng.uniform(-0.1, 0.1, add).astype(np.float32)])
                t = np.concatenate([t, rng.integers(0, 2, size=add, dtype=np.int32)])

            # warmup
            warm_t0 = time.perf_counter()
            while running and (time.perf_counter() - warm_t0 < warm_s):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    running = False
                one_frame()

            # sample
            ms_sum = 0.0
            frames = 0
            samp_t0 = time.perf_counter()
            while running and (time.perf_counter() - samp_t0 < samp_s):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    running = False
                ms_sum += one_frame()
                frames += 1

            if frames > 0:
                avg_ms = ms_sum / frames
                fps = 1000.0 / max(1e-9, avg_ms)
                print(f"[BENCH][Pygame CPU] N={x.shape[0]} avg={avg_ms:.3f} ms ({fps:.1f} FPS)")
                rows.append({"engine": "pygame_cpu", "N": int(x.shape[0]), "avg_ms": avg_ms, "fps": fps})

            target += step_n

        bench_write_csv(out_csv, rows)
        pygame.quit()
        return


    while running:
        frame_dt = clock.tick(60) / 1000.0  # real time step (optional)
        # keep your original "DT" behavior for consistency:
        dt = DT

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # SPACE edge-trigger add particles
        space_down = keys[pygame.K_SPACE]
        if space_down and not space_was_down:
            add = ADD_PER_SPACE

            x_new = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)
            y_new = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, add).astype(np.float32)
            vx_new = rng.uniform(-0.1, 0.1, add).astype(np.float32)
            vy_new = rng.uniform(-0.1, 0.1, add).astype(np.float32)
            t_new = rng.integers(0, 2, size=add, dtype=np.int32)

            x = np.concatenate([x, x_new])
            y = np.concatenate([y, y_new])
            vx = np.concatenate([vx, vx_new])
            vy = np.concatenate([vy, vy_new])
            t = np.concatenate([t, t_new])

        space_was_down = space_down

        t0 = pygame.time.get_ticks()

        cpu_step_grid(
            x, y, vx, vy, t,
            dt, SOFTENING, DRAG, MAX_SPEED,
            SAME_REPEL, OTHER_ATTRACT, FORCE_FALLOFF,
            CELL_SIZE, NEIGHBOR_RADIUS, MAX_NEIGHBORS_PER_PARTICLE,
        )

        # Render
        screen.fill(BG)

        if ALPHA_SOFT_EDGE:
            # blit precomputed “soft sprites”
            ra = sprite_a.get_width() // 2
            rb = sprite_b.get_width() // 2
            for i in range(x.shape[0]):
                sx, sy = world_to_screen(x[i], y[i])
                if t[i] == 0:
                    screen.blit(sprite_a, (sx - ra, sy - ra))
                else:
                    screen.blit(sprite_b, (sx - rb, sy - rb))
        else:
            # faster hard circles
            for i in range(x.shape[0]):
                sx, sy = world_to_screen(x[i], y[i])
                col = COLOR_A if t[i] == 0 else COLOR_B
                pygame.draw.circle(screen, col, (sx, sy), POINT_SIZE)

        pygame.display.flip()

        dt_ms = (pygame.time.get_ticks() - t0)
        ema_ms = dt_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * dt_ms)

        accum_print += frame_dt
        if accum_print > 1.0:
            fps = 1000.0 / max(1e-6, float(ema_ms))
            print(f"CPU step+render: ~{ema_ms:.2f} ms/frame  (~{fps:.1f} FPS)  N={x.shape[0]:,}")
            accum_print = 0.0

    pygame.quit()

if __name__ == "__main__":
    main()
