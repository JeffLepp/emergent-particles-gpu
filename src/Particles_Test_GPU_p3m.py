"""Interactive GPU particles using P3M (mesh far field + direct near field).

This is the accuracy-oriented accelerated counterpart to
Particles_Test_GPU_bruteforce.py.  It keeps the full all-pairs force law: the
smooth far field is evaluated by a GPU FFT convolution and very close pairs
receive a direct correction.  Particle integration uses ping-pong buffers.
"""

import argparse
import csv
import json
import time

import glfw
import moderngl
import numpy as np

from p3m_core import GpuIntegrator, PARTICLE_DTYPE, P3MForceSolver


N_PER_TYPE = 2000
ADD_PER_SPACE = 1000
EXTRA_CAPACITY = 1_000_000

SAME_REPEL = 1.0
OTHER_ATTRACT = 1.001
SOFTENING = 0.02
DRAG = 1.0
MAX_SPEED = 2.0
PARTICLE_SIZE = 1.0
FORCE_FALLOFF = 2.0
WORLD_BOUNDS = 1.0
DT = 1.0 / 90.0
VSYNC = 1
DEFAULT_MESH_RES = 512


VERT_SRC = r"""
#version 430
struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};
layout(std430, binding = 0) readonly buffer Particles { Particle p[]; };
uniform float uPointSize;
out vec3 vColor;
void main() {
    Particle pt = p[gl_VertexID];
    gl_Position = vec4(pt.pos, 0.0, 1.0);
    gl_PointSize = uPointSize;
    vColor = (pt.type == 0) ? vec3(1.0, 0.2, 0.2) : vec3(0.2, 0.5, 1.0);
}
"""


FRAG_SRC = r"""
#version 430
in vec3 vColor;
out vec4 fColor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    if (dot(p, p) > 1.0) discard;
    fColor = vec4(vColor, 1.0);
}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--config", default="bench_config_gpu.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--mesh-res", type=int, default=DEFAULT_MESH_RES,
                        help="Power-of-two physical mesh resolution")
    return parser.parse_args()


def write_bench_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "engine", "N", "compute_ms", "frame_ms", "fps"
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    bench = args.bench
    cfg = {
        "start_n": 20, "end_n": 300_000, "step_n": 10, "step_mul": 1.3,
        "warmup_seconds": 1.0, "sample_seconds": 1.5,
        "abort_ms": 250.0, "seed": 1, "out_csv": "gpu_p3m.csv",
    }
    if bench:
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg.update(json.load(handle))

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(900, 700, "GPU Particles - P3M", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 4.3 window")
    glfw.make_context_current(window)
    glfw.swap_interval(0 if bench else VSYNC)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    print(f"[GPU] {ctx.info['GL_RENDERER']} | {ctx.info['GL_VERSION']} | "
          f"P3M mesh={args.mesh_res}x{args.mesh_res}")

    n0 = 2 * N_PER_TYPE
    capacity = n0 + EXTRA_CAPACITY
    active_n = n0
    rng = np.random.default_rng(cfg["seed"] if bench else 1)
    particles_cpu = np.zeros(capacity, dtype=PARTICLE_DTYPE)
    particles_cpu["pos"][:n0] = rng.uniform(-0.3, 0.3, (n0, 2)).astype(np.float32)
    particles_cpu["vel"][:n0] = rng.uniform(-0.1, 0.1, (n0, 2)).astype(np.float32)
    particles_cpu["type"][:N_PER_TYPE] = 0
    particles_cpu["type"][N_PER_TYPE:n0] = 1

    particles_current = ctx.buffer(particles_cpu.tobytes())
    particles_next = ctx.buffer(reserve=particles_cpu.nbytes)
    solver = P3MForceSolver(
        ctx, capacity, args.mesh_res, WORLD_BOUNDS, SOFTENING,
        FORCE_FALLOFF, SAME_REPEL, OTHER_ATTRACT,
    )
    integrator = GpuIntegrator(ctx, DT, DRAG, MAX_SPEED, WORLD_BOUNDS)
    program = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    program["uPointSize"].value = PARTICLE_SIZE
    vao = ctx.vertex_array(program, [])

    def append_particles(count):
        nonlocal active_n
        count = min(int(count), capacity - active_n)
        if count <= 0:
            print("Particle capacity reached")
            return
        new = np.zeros(count, dtype=PARTICLE_DTYPE)
        new["pos"] = rng.uniform(-WORLD_BOUNDS, WORLD_BOUNDS, (count, 2)).astype(np.float32)
        new["vel"] = rng.uniform(-0.1, 0.1, (count, 2)).astype(np.float32)
        new["type"] = rng.integers(0, 2, count, dtype=np.int32)
        particles_current.write(new.tobytes(), offset=active_n * PARTICLE_DTYPE.itemsize)
        active_n += count

    bench_rows = []
    bench_next_t = None
    bench_skip = 0
    acc_compute = acc_frame = 0.0
    acc_frames = 0
    if bench:
        active_n = 0
        append_particles(int(cfg["start_n"]))
        bench_next_t = time.perf_counter() + float(cfg["warmup_seconds"])
        bench_skip = 10

    query = ctx.query(time=True)
    ema_compute_ms = ema_frame_ms = None
    last_print = time.perf_counter()
    space_was_down = False

    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        space_down = glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
        if space_down and not space_was_down:
            append_particles(ADD_PER_SPACE)
        space_was_down = space_down

        frame_start = time.perf_counter()
        width, height = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, width, height)
        with query:
            solver.compute(particles_current, active_n)
            integrator.step(particles_current, solver.acceleration,
                            particles_next, active_n)
        particles_current, particles_next = particles_next, particles_current

        particles_current.bind_to_storage_buffer(0)
        ctx.clear(0.03, 0.03, 0.04, 1.0)
        vao.render(mode=moderngl.POINTS, vertices=active_n)
        glfw.swap_buffers(window)
        compute_ms = query.elapsed / 1e6
        frame_ms = (time.perf_counter() - frame_start) * 1000.0

        if ema_compute_ms is None:
            ema_compute_ms, ema_frame_ms = compute_ms, frame_ms
        else:
            ema_compute_ms = 0.9 * ema_compute_ms + 0.1 * compute_ms
            ema_frame_ms = 0.9 * ema_frame_ms + 0.1 * frame_ms

        if bench:
            if bench_skip:
                bench_skip -= 1
                acc_compute = acc_frame = 0.0
                acc_frames = 0
            else:
                acc_compute += compute_ms
                acc_frame += frame_ms
                acc_frames += 1

        now = time.perf_counter()
        if bench and now >= bench_next_t and acc_frames:
            mean_compute = acc_compute / acc_frames
            mean_frame = acc_frame / acc_frames
            fps = 1000.0 / max(mean_frame, 1e-6)
            bench_rows.append({
                "engine": f"gpu_p3m_{args.mesh_res}", "N": active_n,
                "compute_ms": f"{mean_compute:.4f}",
                "frame_ms": f"{mean_frame:.4f}", "fps": f"{fps:.2f}",
            })
            print(f"[BENCH][P3M] N={active_n} compute={mean_compute:.3f} ms "
                  f"frame={mean_frame:.3f} ms ({fps:.1f} FPS) over {acc_frames} frames")
            if mean_frame > float(cfg["abort_ms"]) or active_n >= int(cfg["end_n"]):
                break
            previous = active_n
            multiplier = float(cfg.get("step_mul", 1.0))
            append_particles(max(int(cfg["step_n"]), int(active_n * multiplier) - active_n))
            if active_n == previous:
                break
            bench_next_t = now + float(cfg["sample_seconds"])
            bench_skip = 10
        elif not bench and now - last_print > 1.0:
            fps = 1000.0 / max(ema_frame_ms, 1e-6)
            print(f"P3M compute: ~{ema_compute_ms:.3f} ms | frame: "
                  f"~{ema_frame_ms:.3f} ms (~{fps:.1f} FPS) | N={active_n}")
            last_print = now

    if bench:
        output = args.out or str(cfg.get("out_csv", "gpu_p3m.csv"))
        write_bench_csv(output, bench_rows)
        print(f"[BENCH] wrote {output} ({len(bench_rows)} rows)")
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
