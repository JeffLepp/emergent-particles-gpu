"""
GPU Particles (ModernGL + GLFW) - Neighbor-limited via Uniform Grid
-------------------------------------------------------------------
All physics runs on GPU.

Note for user, your program will run with the rules below (assuming you don't change the shader code, but feel free):
- Builds a uniform grid each frame (GPU)
- Each particle only interacts with nearby cells (GPU)
"""

import numpy as np
import glfw
import moderngl
import time
import argparse
import csv
import json

# ------------------------------------------------
# Configurations (You can tune these as you want 
#                   but its on a cool setting now)
# ------------------------------------------------

N_PER_TYPE = 2000               # ie. 100 means 100 red and 100 blue particles at start
ADD_PER_SPACE = 1000            # how many to add when space is pressed (up to EXTRA_CAPACITY)
EXTRA_CAPACITY = 1000000        # pre-allocate this many particles max

SAME_REPEL = 1 # 1.001          # "strength" of same-type repulsion
OTHER_ATTRACT = 1 # 1.00        # "strength" of dif-type attraction
               
SOFTENING = 0.02                # to avoid singularities at close range
DRAG = 1 #.98                   # friction (1 = no drag, 0 = stop immediately)
MAX_SPEED = 2 # .8              # cap on velocity 
PARTICLE_SIZE = 1.0             # size of each particle

FORCE_FALLOFF = 2 # .6          # 0 = inverse-square, 1 = inverse-linear, 2 = no falloff, etc.
WORLD_BOUNDS = 1.0   
DT = 1.0 / 90.0                                             

VSYNC = 1                      # 1 = demo (capped to monitor), 0 = benchmarking (uncapped)

# Neighbor grid params
GRID_RES = 256                 # grid is GRID_RES x GRID_RES    
NEIGHBOR_RADIUS = 0.18         # world units                    
MAX_NEIGHBORS = 2056           # safety cap per particle        

# ----------------------------
# GLSL: Grid clear (set head[] = -1)
# ----------------------------
GRID_CLEAR_SRC = r"""
#version 430
layout(std430, binding = 1) buffer GridHead { int head[]; };

uniform int uCellCount;

layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uCellCount)) return;
    head[idx] = -1;
}
"""

# ----------------------------
# GLSL: Grid build (linked cell list)
# - head[cell] = first particle index
# - nextIdx[i] = next particle in same cell (or -1)
# ----------------------------
GRID_BUILD_SRC = r"""
#version 430

struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};

layout(std430, binding = 0) buffer Particles { Particle p[]; };
layout(std430, binding = 1) buffer GridHead  { int head[]; };
layout(std430, binding = 2) buffer NextIdx   { int nextIdx[]; };

uniform int   uN;
uniform int   uGridRes;
uniform float uBounds;

layout(local_size_x = 256) in;

int cellIndex(vec2 pos) {
    vec2 p01 = (pos + vec2(uBounds)) / (2.0 * uBounds); // [-B,B] -> [0,1]
    ivec2 c  = ivec2(clamp(floor(p01 * float(uGridRes)), 0.0, float(uGridRes - 1)));
    return c.x + c.y * uGridRes;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    int c = cellIndex(p[i].pos);

    // push-front into cell list
    int old = atomicExchange(head[c], int(i));
    nextIdx[i] = old;
}
"""

# ----------------------------
# GLSL: Physics compute (neighbors only)
# ----------------------------
COMPUTE_SRC = r"""
#version 430

struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};

layout(std430, binding = 0) buffer Particles { Particle p[]; };
layout(std430, binding = 1) buffer GridHead  { int head[]; };
layout(std430, binding = 2) buffer NextIdx   { int nextIdx[]; };

uniform int   uN;
uniform float uDT;
uniform float uSoft;
uniform float uDrag;
uniform float uMaxSpeed;

uniform float uSameRepel;
uniform float uOtherAttract;
uniform float uForceFalloff;
uniform float uBounds;

uniform int   uGridRes;
uniform float uNeighborRadius;
uniform int   uMaxNeighbors;

layout(local_size_x = 256) in;

vec2 clampSpeed(vec2 v, float maxS) {
    float s2 = dot(v, v);
    float m2 = maxS * maxS;
    if (s2 > m2) {
        float s = sqrt(s2);
        return v * (maxS / s);
    }
    return v;
}

ivec2 cellCoord(vec2 pos, int gridRes, float bounds) {
    vec2 p01 = (pos + vec2(bounds)) / (2.0 * bounds);
    ivec2 c  = ivec2(clamp(floor(p01 * float(gridRes)), 0.0, float(gridRes - 1)));
    return c;
}

int cellIndexFromCoord(ivec2 c, int gridRes) {
    return c.x + c.y * gridRes;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec2 pos_i = p[i].pos;
    vec2 vel_i = p[i].vel;
    int  t_i   = p[i].type;

    vec2 acc = vec2(0.0);

    int gridRes = uGridRes;
    float cellSize = (2.0 * uBounds) / float(gridRes);
    int rCells = int(ceil(uNeighborRadius / cellSize));

    ivec2 ci = cellCoord(pos_i, gridRes, uBounds);

    int neighborsChecked = 0;
    float r2max = uNeighborRadius * uNeighborRadius;

    for (int oy = -rCells; oy <= rCells; oy++) {
        for (int ox = -rCells; ox <= rCells; ox++) {
            ivec2 cc = ci + ivec2(ox, oy);
            if (cc.x < 0 || cc.y < 0 || cc.x >= gridRes || cc.y >= gridRes) continue;

            int cell = cellIndexFromCoord(cc, gridRes);
            int j = head[cell];

            while (j != -1) {
                if (j != int(i)) {
                    vec2 d = p[j].pos - pos_i;
                    float r2 = dot(d, d) + uSoft;

                    if (r2 < r2max) {
                        float invr = inversesqrt(r2);
                        vec2 dir = d * invr;

                        float base = 1.0 / r2;
                        float mag  = pow(base, uForceFalloff);

                        int t_j = p[j].type;

                        if (t_j == t_i) {
                            acc -= dir * (uSameRepel * mag);
                        } else {
                            acc += dir * (uOtherAttract * mag);
                        }
                    }
                }

                j = nextIdx[j];

                neighborsChecked++;
                if (neighborsChecked >= uMaxNeighbors) break;
            }

            if (neighborsChecked >= uMaxNeighbors) break;
        }
        if (neighborsChecked >= uMaxNeighbors) break;
    }

    vel_i += acc * uDT;
    vel_i *= uDrag;
    vel_i = clampSpeed(vel_i, uMaxSpeed);
    pos_i += vel_i * uDT;

    // Bounce boundaries
    if (pos_i.x < -uBounds) { pos_i.x = -uBounds; vel_i.x *= -0.9; }
    if (pos_i.x >  uBounds) { pos_i.x =  uBounds; vel_i.x *= -0.9; }
    if (pos_i.y < -uBounds) { pos_i.y = -uBounds; vel_i.y *= -0.9; }
    if (pos_i.y >  uBounds) { pos_i.y =  uBounds; vel_i.y *= -0.9; }

    p[i].pos = pos_i;
    p[i].vel = vel_i;
}
"""

# ----------------------------
# GLSL: Vertex + Fragment
# ----------------------------
VERT_SRC = r"""
#version 430

struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};

layout(std430, binding = 0) buffer Particles { Particle p[]; };

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

# ----------------------------
# Benchmark mode (--bench): sweep N, write csv, exit
# ----------------------------
ENGINE = "gpu_grid"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="Run automated sweep benchmark and exit")
    ap.add_argument("--config", default="bench_config_gpu.json", help="Path to benchmark config json")
    ap.add_argument("--out", default="", help="Override out_csv from the config")
    return ap.parse_args()

def write_bench_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "N", "compute_ms", "frame_ms", "fps"])
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    bench = args.bench
    cfg = None
    if bench:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(900, 700, "GPU Two-Type Particles (Neighbor Grid)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    vsync = 0 if bench else VSYNC
    glfw.swap_interval(vsync)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    print(f"[GPU] {ctx.info['GL_RENDERER']} | {ctx.info['GL_VERSION']} | vsync={vsync}")

    # ----------------------------
    # Particle init (CPU once)
    # ----------------------------
    N0 = 2 * N_PER_TYPE
    CAPACITY = N0 + EXTRA_CAPACITY
    active_N = N0

    particle_dtype = np.dtype([
        ("pos",  np.float32, (2,)),
        ("vel",  np.float32, (2,)),
        ("type", np.int32),
        ("pad0", np.int32),
    ])
    stride = particle_dtype.itemsize

    particles_cpu = np.zeros(CAPACITY, dtype=particle_dtype)
    rng = np.random.default_rng(cfg["seed"] if bench else 1)

    # A
    for i in range(N_PER_TYPE):
        particles_cpu["pos"][i]  = np.array([0.0, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles_cpu["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles_cpu["type"][i] = 0

    # B
    for i in range(N_PER_TYPE, N0):
        particles_cpu["pos"][i]  = np.array([0.0, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles_cpu["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles_cpu["type"][i] = 1

    # SSBO: particles (binding=0) - allocate full capacity
    ssbo_particles = ctx.buffer(particles_cpu.tobytes())
    ssbo_particles.bind_to_storage_buffer(binding=0)

    # SSBO: grid head (binding=1)
    cell_count = GRID_RES * GRID_RES
    grid_head = np.full(cell_count, -1, dtype=np.int32)
    ssbo_head = ctx.buffer(grid_head.tobytes())
    ssbo_head.bind_to_storage_buffer(binding=1)

    # SSBO: next pointers (binding=2) - allocate full capacity
    next_idx_cpu = np.full(CAPACITY, -1, dtype=np.int32)
    ssbo_next = ctx.buffer(next_idx_cpu.tobytes())
    ssbo_next.bind_to_storage_buffer(binding=2)

    # ----------------------------
    # Shaders / Programs
    # ----------------------------
    cs_clear = ctx.compute_shader(GRID_CLEAR_SRC)
    cs_build = ctx.compute_shader(GRID_BUILD_SRC)
    cs_physics = ctx.compute_shader(COMPUTE_SRC)

    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    vao = ctx.vertex_array(prog, [])  # gl_VertexID fetches from SSBO

    # Uniforms: clear
    cs_clear["uCellCount"].value = cell_count

    # Uniforms: build
    cs_build["uN"].value = active_N
    cs_build["uGridRes"].value = GRID_RES
    cs_build["uBounds"].value = WORLD_BOUNDS

    # Uniforms: physics
    cs_physics["uN"].value = active_N
    cs_physics["uDT"].value = DT
    cs_physics["uSoft"].value = SOFTENING
    cs_physics["uDrag"].value = DRAG
    cs_physics["uMaxSpeed"].value = MAX_SPEED
    cs_physics["uSameRepel"].value = SAME_REPEL
    cs_physics["uOtherAttract"].value = OTHER_ATTRACT
    cs_physics["uForceFalloff"].value = FORCE_FALLOFF
    cs_physics["uBounds"].value = WORLD_BOUNDS
    cs_physics["uGridRes"].value = GRID_RES
    cs_physics["uNeighborRadius"].value = NEIGHBOR_RADIUS
    cs_physics["uMaxNeighbors"].value = MAX_NEIGHBORS

    # Render uniforms
    prog["uPointSize"].value = PARTICLE_SIZE

    # ----------------------------
    # Append helper (writes only new range)
    # ----------------------------
    def append_particles(k: int):
        nonlocal active_N

        if active_N + k > CAPACITY:
            print(f"Out of capacity: active_N={active_N}, add={k}, CAPACITY={CAPACITY}. Increase EXTRA_CAPACITY.")
            return

        new = np.zeros(k, dtype=particle_dtype)
        new["pos"]  = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, (k, 2)).astype(np.float32)
        new["vel"]  = rng.uniform(-0.1, 0.1, (k, 2)).astype(np.float32)
        new["type"] = rng.integers(0, 2, size=k, dtype=np.int32)
        new["pad0"] = 0

        start = active_N
        end = active_N + k

        # (optionl) keep CPU shadow
        particles_cpu[start:end] = new

        ssbo_particles.write(new.tobytes(), offset=start * stride)
        ssbo_next.write(np.full(k, -1, dtype=np.int32).tobytes(), offset=start * 4)

        active_N = end
        cs_build["uN"].value = active_N
        cs_physics["uN"].value = active_N

    # ----------------------------
    # Timing + Input
    # ----------------------------
    ema_compute_ms = None
    ema_frame_ms = None
    last_print = time.perf_counter()
    query = ctx.query(time=True)

    space_was_down = False

    # Bench sweep state (mean over each sample window, not the EMA -- the EMA
    # lags badly once frames get slow and would flatter the tail of the curve)
    bench_rows = []
    bench_next_t = None
    bench_skip = 0
    acc_compute = acc_frame = 0.0
    acc_frames = 0
    if bench:
        active_N = 0                       # the sweep owns N, ignore N_PER_TYPE
        append_particles(int(cfg["start_n"]))
        bench_next_t = time.perf_counter() + float(cfg["warmup_seconds"])
        bench_skip = 10

    # ----------------------------
    # Main loop
    # ----------------------------
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        space_down = (glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS)
        if space_down and not space_was_down:
            append_particles(ADD_PER_SPACE)
        space_was_down = space_down

        frame_t0 = time.perf_counter()

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)

        groups_particles = (active_N + 256 - 1) // 256
        groups_cells = (cell_count + 256 - 1) // 256

        with query:
            cs_clear.run(group_x=groups_cells)
            cs_build.run(group_x=groups_particles)
            cs_physics.run(group_x=groups_particles)

        ctx.memory_barrier()

        ctx.clear(0.03, 0.03, 0.04, 1.0)
        vao.render(mode=moderngl.POINTS, vertices=active_N)
        glfw.swap_buffers(window)

        compute_ms = query.elapsed / 1e6
        frame_ms = (time.perf_counter() - frame_t0) * 1000.0

        if ema_compute_ms is None:
            ema_compute_ms = compute_ms
            ema_frame_ms = frame_ms
        else:
            ema_compute_ms = 0.9 * ema_compute_ms + 0.1 * compute_ms
            ema_frame_ms = 0.9 * ema_frame_ms + 0.1 * frame_ms

        if bench:
            if bench_skip > 0:
                # discard the frames right after an N change (buffer writes, warmup)
                bench_skip -= 1
                acc_compute = acc_frame = 0.0
                acc_frames = 0
            else:
                acc_compute += compute_ms
                acc_frame += frame_ms
                acc_frames += 1

        now = time.perf_counter()
        if bench:
            if now >= bench_next_t and acc_frames > 0:
                mean_compute = acc_compute / acc_frames
                mean_frame = acc_frame / acc_frames
                fps = 1000.0 / max(1e-6, mean_frame)
                bench_rows.append({
                    "engine": ENGINE, "N": active_N,
                    "compute_ms": f"{mean_compute:.4f}",
                    "frame_ms": f"{mean_frame:.4f}",
                    "fps": f"{fps:.2f}",
                })
                print(f"[BENCH][{ENGINE}] N={active_N} compute={mean_compute:.3f} ms "
                      f"frame={mean_frame:.3f} ms ({fps:.1f} FPS) over {acc_frames} frames")

                if mean_frame > float(cfg["abort_ms"]) or active_N >= int(cfg["end_n"]):
                    break
                prev_n = active_N
                # linear at small N, geometric above it, so one config covers
                # the CPU's range (20..400) and the GPU's (..300k) in ~37 samples
                mul = float(cfg.get("step_mul", 1.0))
                append_particles(max(int(cfg["step_n"]), int(active_N * mul) - active_N))
                if active_N == prev_n:
                    break          # hit CAPACITY, nothing more to sweep
                bench_next_t = now + float(cfg["sample_seconds"])
                bench_skip = 10
        elif now - last_print > 1.0:
            fps = 1000.0 / max(1e-6, ema_frame_ms)
            print(f"GPU compute: ~{ema_compute_ms:.3f} ms | frame: ~{ema_frame_ms:.3f} ms (~{fps:.1f} FPS) | N={active_N}")
            last_print = now

    if bench:
        out_path = args.out or str(cfg.get("out_csv", ENGINE + ".csv"))
        write_bench_csv(out_path, bench_rows)
        print(f"[BENCH] wrote {out_path} ({len(bench_rows)} rows)")

    glfw.terminate()



if __name__ == "__main__":
    main()
