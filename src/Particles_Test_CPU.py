"""
CPU Two-Type Particles (NumPy + ModernGL + GLFW)
---------------------------------------------------------------
Same visual + interaction rules as your GPU version, but physics is computed on CPU.

Notes:
- A truly apples-to-apples CPU version of your O(N^2) compute shader is also O(N^2) here,
  which becomes unusable above a few thousand particles.
- This CPU version uses a **uniform grid neighbor search** (approximate, local interactions)
  so it can still run at large N and give you meaningful performance comparisons.

Keys:
- ESC = quit
- SPACE = add 100 particles (random, no reset)
"""
import argparse
import json
import csv
import time
import numpy as np
import glfw
import moderngl

# ----------------------------
# Configurations
# ----------------------------
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

POINT_SIZE = 1.8

# CPU neighbor grid params (tune these for speed/behavior)
CELL_SIZE = 0.08         # bigger = more neighbors, slower, more like long-range
NEIGHBOR_RADIUS = 0.16   # only interact within this distance
MAX_NEIGHBORS_PER_PARTICLE = 256  # cap work per particle

# ----------------------------
# Benchmarking utilities (for --bench mode)
# ----------------------------
def load_bench_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="Run automated sweep benchmark and exit")
    ap.add_argument("--config", default="bench_config.json", help="Path to benchmark config json")
    return ap.parse_args()

def bench_append_particles(particles, k, rng, particle_dtype, WORLD_BOUNDS):
    new = np.zeros(k, dtype=particle_dtype)
    new["pos"] = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, (k, 2)).astype(np.float32)
    new["vel"] = rng.uniform(-0.1, 0.1, (k, 2)).astype(np.float32)
    new["type"] = rng.integers(0, 2, size=k, dtype=np.int32)
    new["pad0"] = 0
    return np.concatenate([particles, new], axis=0)

def bench_write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "N", "avg_ms", "fps"])
        w.writeheader()
        w.writerows(rows)

# ----------------------------
# GLSL: Vertex + Fragment (draw from SSBO)
# ----------------------------
VERT_SRC = r"""
#version 430

struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};

layout(std430, binding = 0) buffer Particles {
    Particle p[];
};

uniform float uPointSize;
out vec3 vColor;

void main() {
    int idx = gl_VertexID;
    vec2 pos = p[idx].pos;
    int  t   = p[idx].type;

    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = uPointSize;

    vColor = (t == 0) ? vec3(1.0, 0.25, 0.25) : vec3(0.25, 0.55, 1.0);
}
"""

FRAG_SRC = r"""
#version 430

in vec3 vColor;
out vec4 fColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    float alpha = 1.0 - smoothstep(0.5, 1.0, r2);
    fColor = vec4(vColor, alpha);
}
"""


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
    # x
    mask = x < -bounds
    if np.any(mask):
        x[mask] = -bounds
        vx[mask] *= -0.9
    mask = x > bounds
    if np.any(mask):
        x[mask] = bounds
        vx[mask] *= -0.9
    # y
    mask = y < -bounds
    if np.any(mask):
        y[mask] = -bounds
        vy[mask] *= -0.9
    mask = y > bounds
    if np.any(mask):
        y[mask] = bounds
        vy[mask] *= -0.9
    return x, y, vx, vy


def cpu_step_grid(x, y, vx, vy, t, dt, soft, drag, max_speed,
                  same_repel, other_attract, force_falloff,
                  cell_size, neighbor_radius, max_neighbors):
    """
    Approximate CPU physics:
    - Put particles into a uniform grid
    - For each particle, only check particles in its 3x3 neighboring cells
    - Only interact within neighbor_radius
    - Optionally cap neighbor checks per particle
    """

    N = x.shape[0]
    inv_cell = 1.0 / cell_size

    # Map positions [-bounds, bounds] into grid coordinates
    gx = np.floor((x + WORLD_BOUNDS) * inv_cell).astype(np.int32)
    gy = np.floor((y + WORLD_BOUNDS) * inv_cell).astype(np.int32)

    # Grid dimensions (cover [-bounds, bounds])
    grid_w = int(np.ceil((2.0 * WORLD_BOUNDS) * inv_cell)) + 2
    grid_h = grid_w

    # Clamp to grid
    gx = np.clip(gx, 0, grid_w - 1)
    gy = np.clip(gy, 0, grid_h - 1)

    cell_id = gx + gy * grid_w

    # Sort by cell id so each cell becomes a contiguous range
    order = np.argsort(cell_id, kind="mergesort")
    cell_sorted = cell_id[order]

    # Find cell ranges in the sorted list
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

                # skip self if present
                if idxs.size and np.any(idxs == i):
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
                    acc_x -= np.sum(dirx[same] * s_mag)
                    acc_y -= np.sum(diry[same] * s_mag)

                other = ~same
                if np.any(other):
                    o_mag = other_attract * mag[other]
                    acc_x += np.sum(dirx[other] * o_mag)
                    acc_y += np.sum(diry[other] * o_mag)

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
    return x, y, vx, vy


def main():

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    
    args = parse_args()
    bench = args.bench
    cfg = load_bench_config(args.config) if bench else None

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(900, 700, "CPU Two-Type Particles (Grid Neighbors)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    glfw.swap_interval(0)   # Unbound/bound FPS here -> unbounded for benchmark best to keep on

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # Build initial particles on CPU (only once)
    N = 2 * N_PER_TYPE

    particle_dtype = np.dtype([
        ("pos",  np.float32, (2,)),
        ("vel",  np.float32, (2,)),
        ("type", np.int32),
        ("pad0", np.int32),
    ])

    particles = np.zeros(N, dtype=particle_dtype)
    rng = np.random.default_rng(cfg["seed"] if bench else 1)


    # Type A on left
    particles["pos"][:N_PER_TYPE] = (np.array([-0.5, 0.0], np.float32)
                                    + rng.uniform(-0.3, 0.3, (N_PER_TYPE, 2)).astype(np.float32))
    particles["vel"][:N_PER_TYPE] = rng.uniform(-0.1, 0.1, (N_PER_TYPE, 2)).astype(np.float32)
    particles["type"][:N_PER_TYPE] = 0

    # Type B on right
    particles["pos"][N_PER_TYPE:] = (np.array([+0.5, 0.0], np.float32)
                                    + rng.uniform(-0.3, 0.3, (N_PER_TYPE, 2)).astype(np.float32))
    particles["vel"][N_PER_TYPE:] = rng.uniform(-0.1, 0.1, (N_PER_TYPE, 2)).astype(np.float32)
    particles["type"][N_PER_TYPE:] = 1

    # Convenience views for CPU stepping
    x = particles["pos"][:, 0]
    y = particles["pos"][:, 1]
    vx = particles["vel"][:, 0]
    vy = particles["vel"][:, 1]
    t = particles["type"]

    # Upload as SSBO (we'll update it every frame from CPU)
    ssbo = ctx.buffer(particles.tobytes())
    ssbo.bind_to_storage_buffer(binding=0)

    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    vao = ctx.vertex_array(prog, [])  # gl_VertexID reads from SSBO
    prog["uPointSize"].value = POINT_SIZE

    # Simple perf stats
    ema_ms = None
    last_print = time.perf_counter()

    space_was_down = False
    
    if bench:
        # Re-init particle count to exactly start_n (total particles)
        start_n = int(cfg["start_n"])
        end_n   = int(cfg["end_n"])
        step_n  = int(cfg["step_n"])
        warm_s  = float(cfg["warmup_seconds"])
        samp_s  = float(cfg["sample_seconds"])
        out_csv = str(cfg.get("out_csv", "cpu_moderngl.csv"))

        # Force initial N to match start_n
        # (your current init creates N = 2*N_PER_TYPE)
        if N != start_n:
            if N > start_n:
                particles = particles[:start_n].copy()
            else:
                particles = particles.copy()
                particles = bench_append_particles(particles, start_n - N, rng, particle_dtype, WORLD_BOUNDS)

            x = particles["pos"][:, 0]
            y = particles["pos"][:, 1]
            vx = particles["vel"][:, 0]
            vy = particles["vel"][:, 1]
            t = particles["type"]
            N = particles.shape[0]

            # rebuild SSBO to exact size
            ssbo.release()
            ssbo = ctx.buffer(particles.tobytes())
            ssbo.bind_to_storage_buffer(binding=0)

        rows = []
        target = start_n

        def one_frame():
            nonlocal ssbo, particles, N
            t0 = time.perf_counter()

            cpu_step_grid(
                x, y, vx, vy, t,
                DT, SOFTENING, DRAG, MAX_SPEED,
                SAME_REPEL, OTHER_ATTRACT, FORCE_FALLOFF,
                CELL_SIZE, NEIGHBOR_RADIUS, MAX_NEIGHBORS_PER_PARTICLE,
            )

            # upload
            if ssbo.size != particles.nbytes:
                ssbo.release()
                ssbo = ctx.buffer(particles.tobytes())
                ssbo.bind_to_storage_buffer(binding=0)
            else:
                ssbo.write(particles.tobytes())

            fb_w, fb_h = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, fb_w, fb_h)
            ctx.clear(0.03, 0.03, 0.04, 1.0)
            vao.render(mode=moderngl.POINTS, vertices=N)
            glfw.swap_buffers(window)

            return (time.perf_counter() - t0) * 1000.0

        while target <= end_n and (not glfw.window_should_close(window)):
            # grow to target
            if N < target:
                particles = bench_append_particles(particles, target - N, rng, particle_dtype, WORLD_BOUNDS)

                x = particles["pos"][:, 0]
                y = particles["pos"][:, 1]
                vx = particles["vel"][:, 0]
                vy = particles["vel"][:, 1]
                t = particles["type"]
                N = particles.shape[0]

            # warmup
            warm_t0 = time.perf_counter()
            while time.perf_counter() - warm_t0 < warm_s:
                glfw.poll_events()
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(window, True)
                    break
                one_frame()

            # sample
            ms_sum = 0.0
            frames = 0
            samp_t0 = time.perf_counter()
            while time.perf_counter() - samp_t0 < samp_s:
                glfw.poll_events()
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(window, True)
                    break
                ms_sum += one_frame()
                frames += 1

            if frames > 0:
                avg_ms = ms_sum / frames
                fps = 1000.0 / max(1e-9, avg_ms)
                print(f"[BENCH][ModernGL CPU] N={N} avg={avg_ms:.3f} ms ({fps:.1f} FPS)")
                rows.append({"engine": "moderngl_cpu", "N": N, "avg_ms": avg_ms, "fps": fps})

            target += step_n

        bench_write_csv(out_csv, rows)
        glfw.terminate()
        return


    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # SPACE edge-trigger: add 100 particles (random) without resetting sim
        space_down = (glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS)
        if space_down and not space_was_down:
            new = np.zeros(ADD_PER_SPACE, dtype=particle_dtype)
            new["pos"] = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, (ADD_PER_SPACE, 2)).astype(np.float32)
            new["vel"] = rng.uniform(-0.1, 0.1, (ADD_PER_SPACE, 2)).astype(np.float32)
            new["type"] = rng.integers(0, 2, size=ADD_PER_SPACE, dtype=np.int32)
            new["pad0"] = 0

            particles = np.concatenate([particles, new], axis=0)

            # Rebuild views so CPU step sees the new arrays
            x = particles["pos"][:, 0]
            y = particles["pos"][:, 1]
            vx = particles["vel"][:, 0]
            vy = particles["vel"][:, 1]
            t = particles["type"]

            N = particles.shape[0]
        space_was_down = space_down

        t0 = time.perf_counter()

        # CPU physics
        cpu_step_grid(
            x, y, vx, vy, t,
            DT, SOFTENING, DRAG, MAX_SPEED,
            SAME_REPEL, OTHER_ATTRACT, FORCE_FALLOFF,
            CELL_SIZE, NEIGHBOR_RADIUS, MAX_NEIGHBORS_PER_PARTICLE,
        )

        # Upload updated particle buffer to GPU for drawing
        # If N grew, reallocate the SSBO (only when space is pressed)
        if ssbo.size != particles.nbytes:
            ssbo.release()
            ssbo = ctx.buffer(particles.tobytes())
            ssbo.bind_to_storage_buffer(binding=0)
        else:
            ssbo.write(particles.tobytes())

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)

        ctx.clear(0.03, 0.03, 0.04, 1.0)
        vao.render(mode=moderngl.POINTS, vertices=N)
        glfw.swap_buffers(window)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        ema_ms = dt_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * dt_ms)

        now = time.perf_counter()
        if now - last_print > 1.0:
            fps = 1000.0 / max(1e-6, ema_ms)
            print(f"CPU step+upload+render: ~{ema_ms:.2f} ms/frame  (~{fps:.1f} FPS)  N={N:,}")
            last_print = now

    glfw.terminate()


if __name__ == "__main__":
    main()
