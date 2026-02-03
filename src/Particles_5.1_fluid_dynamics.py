"""
GPU Particles (ModernGL + GLFW) - Neighbor-limited via Uniform Grid
-------------------------------------------------------------------
Two types: A(0) and B(1)
- Same type repels
- Opposite type attracts

All physics runs on GPU.

Key upgrade vs O(N^2):
- Builds a uniform grid each frame (GPU)
- Each particle only interacts with nearby cells (GPU)
"""

import numpy as np
import glfw
import moderngl

# ----------------------------
# Configurations
# ----------------------------
N_PER_TYPE = 25000
DT = 1.0 / 60.0
SOFTENING = 0.01
DRAG = .99
MAX_SPEED = 0.8

SAME_REPEL = 1.01
OTHER_ATTRACT = 1.51
FORCE_FALLOFF = 0.5
WORLD_BOUNDS = .95

# Neighbor grid params
GRID_RES = 256                 # grid is GRID_RES x GRID_RES
NEIGHBOR_RADIUS = 0.18         # world units (tune this)
MAX_NEIGHBORS = 100            # safety cap per particle (prevents worst-case blowups)

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

layout(std430, binding = 0) readonly buffer ParticlesIn  { Particle pin[]; };
layout(std430, binding = 3) writeonly buffer ParticlesOut { Particle pout[]; };
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

    int c = cellIndex(pin[i].pos);

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

layout(std430, binding = 0) readonly buffer ParticlesIn   { Particle pin[]; };
layout(std430, binding = 3) writeonly buffer ParticlesOut { Particle pout[]; };
layout(std430, binding = 1) buffer GridHead  { int head[]; };
layout(std430, binding = 2) buffer NextIdx   { int nextIdx[]; };

uniform int   uN;
uniform float uDT;
uniform float uSoft;
uniform float uDrag;
uniform float uMaxSpeed;

uniform float uSameRepel;
uniform float uOtherAttract;
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

    vec2 pos_i = pin[i].pos;
    vec2 vel_i = pin[i].vel;
    int  t_i   = pin[i].type;

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
                    vec2 d = pin[j].pos - pos_i;
                    float r2 = dot(d, d) + uSoft;

                    if (r2 < r2max) {
                        neighborsChecked++;
                        if (neighborsChecked >= uMaxNeighbors) break;

                        float invr = inversesqrt(r2);
                        vec2 dir = d * invr;

                        float base = 1.0 / r2;
                        float r = sqrt(r2);
                        float q = 1.0 - (r / uNeighborRadius);  // q in (0..1)
                        float mag = q * q;                      // smooth falloff

                        acc -= dir * (uSameRepel * mag);

                        int t_j = pin[j].type;
                        if (t_j == t_i) {
                            acc -= dir * (uSameRepel * mag);
                        } else {
                            acc += dir * (uOtherAttract * mag);
                        }
                    }
                }

                j = nextIdx[j];
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

    // write to output buffer (ping-pong)
    pout[i].pos  = pos_i;
    pout[i].vel  = vel_i;
    pout[i].type = t_i;
    pout[i].pad0 = 0;
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

layout(std430, binding = 0) readonly buffer ParticlesIn  { Particle pin[]; };

uniform float uPointSize;
out vec3 vColor;

void main() {
    int idx = gl_VertexID;
    vec2 pos = pin[idx].pos;
    int  t   = pin[idx].type;

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

def assign_initial_velocities(
    particles,
    mode="random",
    speed=0.15,
    swirl_center=(0.0, 0.0),
    radial_bias=1.0,
    seed=42,
):
    """
    Assign initial velocities to particles (CPU-side, one-time).

    Parameters
    ----------
    particles : structured numpy array
        Must have fields ["pos", "vel", "type"]
    mode : str
        "random"  -> isotropic random velocities
        "radial"  -> outward/inward from center
        "swirl"   -> tangential swirl around center
        "split"   -> opposite flow per type (A vs B)
    speed : float
        Base velocity magnitude
    swirl_center : tuple(float, float)
        Center used for radial / swirl modes
    radial_bias : float
        +1 = outward, -1 = inward (radial mode)
    seed : int
        RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    pos = particles["pos"]
    vel = particles["vel"]
    typ = particles["type"]

    if mode == "random":
        angles = rng.uniform(0, 2 * np.pi, len(particles))
        vel[:, 0] = np.cos(angles) * speed
        vel[:, 1] = np.sin(angles) * speed

    elif mode == "radial":
        center = np.array(swirl_center, dtype=np.float32)
        d = pos - center
        norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        vel[:] = radial_bias * speed * d / norm

    elif mode == "swirl":
        center = np.array(swirl_center, dtype=np.float32)
        d = pos - center
        norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        tangent = np.stack([-d[:, 1], d[:, 0]], axis=1)
        vel[:] = speed * tangent / norm

    elif mode == "split":
        angles = rng.uniform(0, 2 * np.pi, len(particles))
        base = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        vel[:] = speed * base
        vel[typ == 1] *= -1.0  # B flows opposite of A

    else:
        raise ValueError(f"Unknown velocity mode: {mode}")


def main():
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
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)

    # Particle init (CPU once)
    N = 2 * N_PER_TYPE
    particle_dtype = np.dtype([
        ("pos",  np.float32, (2,)),
        ("vel",  np.float32, (2,)),
        ("type", np.int32),
        ("pad0", np.int32),
    ])

    particles = np.zeros(N, dtype=particle_dtype)
    rng = np.random.default_rng(1)


    # A on left
    for i in range(N_PER_TYPE):
        particles["pos"][i]  = np.array([-0.0, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles["type"][i] = 0

    # B on right
    for i in range(N_PER_TYPE, N):
        particles["pos"][i]  = np.array([+0.0, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles["type"][i] = 1

    assign_initial_velocities(
        particles,
        mode="random",      # try: "random", "radial", "swirl", "split"
        speed=0.25,
        swirl_center=(0.0, 0.0),
        seed=42,
    )

    # Two SSBOs for ping-pong
    ssbo_a = ctx.buffer(particles.tobytes())   # current (read)
    ssbo_b = ctx.buffer(reserve=particles.nbytes)  # next (write)

    # initial bindings for this frame
    ssbo_a.bind_to_storage_buffer(binding=0)  # ParticlesIn
    ssbo_b.bind_to_storage_buffer(binding=3)  # ParticlesOut

    # SSBO: grid head (binding=1) and next pointers (binding=2)
    cell_count = GRID_RES * GRID_RES
    grid_head = np.full(cell_count, -1, dtype=np.int32)
    ssbo_head = ctx.buffer(grid_head.tobytes())
    ssbo_head.bind_to_storage_buffer(binding=1)

    next_idx = np.full(N, -1, dtype=np.int32)
    ssbo_next = ctx.buffer(next_idx.tobytes())
    ssbo_next.bind_to_storage_buffer(binding=2)

    # Shaders
    cs_clear = ctx.compute_shader(GRID_CLEAR_SRC)
    cs_build = ctx.compute_shader(GRID_BUILD_SRC)
    cs_physics = ctx.compute_shader(COMPUTE_SRC)

    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    vao = ctx.vertex_array(prog, [])  # gl_VertexID fetches from SSBO

    # Uniforms: clear
    cs_clear["uCellCount"].value = cell_count

    # Uniforms: build
    cs_build["uN"].value = N
    cs_build["uGridRes"].value = GRID_RES
    cs_build["uBounds"].value = WORLD_BOUNDS

    # Uniforms: physics
    cs_physics["uN"].value = N
    cs_physics["uDT"].value = DT
    cs_physics["uSoft"].value = SOFTENING
    cs_physics["uDrag"].value = DRAG
    cs_physics["uMaxSpeed"].value = MAX_SPEED
    cs_physics["uSameRepel"].value = SAME_REPEL
    cs_physics["uOtherAttract"].value = OTHER_ATTRACT
    cs_physics["uBounds"].value = WORLD_BOUNDS
    cs_physics["uGridRes"].value = GRID_RES
    cs_physics["uNeighborRadius"].value = NEIGHBOR_RADIUS
    cs_physics["uMaxNeighbors"].value = MAX_NEIGHBORS

    # Render uniforms
    prog["uPointSize"].value = 12.0

    def dispatch(shader, count):
        groups_x = (count + 256 - 1) // 256
        shader.run(group_x=groups_x)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)

        # 1) clear grid heads to -1
        dispatch(cs_clear, cell_count)
        ctx.memory_barrier()  # make sure clear is visible

        # 2) build grid linked lists
        dispatch(cs_build, N)
        ctx.memory_barrier()  # make sure head/next are visible

        # 3) physics (neighbors only)
        # swap ping-pong buffers
        # 3) physics (neighbors only) - run compute
        dispatch(cs_physics, N)
        ctx.memory_barrier()

        # 4) swap ping-pong buffers AFTER physics
        ssbo_a, ssbo_b = ssbo_b, ssbo_a
        ssbo_a.bind_to_storage_buffer(binding=0)  # new input for next frame + rendering
        ssbo_b.bind_to_storage_buffer(binding=3)  # new output for next frame
        ctx.memory_barrier()


        # render
        ctx.clear(0.03, 0.03, 0.04, 1.0)
        vao.render(mode=moderngl.POINTS, vertices=N)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
