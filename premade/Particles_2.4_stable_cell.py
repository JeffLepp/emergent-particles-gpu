"""
GPU Particles (ModernGL + GLFW)
---------------------------------------------------------------
Two types: A(0) and B(1)
- Same type repels
- Opposite type attracts

Physics runs on GPU (compute shader).
"""

# TODO: I want Particle struct to contain a weight
#       From this we can dynamically allocate configs

#       Mouse interaction (L: grav R: repl)

# Handled by the CPU
import numpy as np
import glfw
import moderngl

# ----------------------------
# Configurations
# ----------------------------
N_PER_TYPE = 5000
DT = 1.0 / 60.0     # How much time passes per frame
SOFTENING = 0.10    # Prevents divide by zero
DRAG = 0.99         # Lower is more dampening
MAX_SPEED = 1.5

SAME_REPEL = 1
OTHER_ATTRACT = 1.002
FORCE_FALLOFF = .85
WORLD_BOUNDS = 1.0

# ----------------------------
# GLSL: Compute shader (OpenGL 4.3)
# ----------------------------
COMPUTE_SRC = r"""
#version 430

// Define the struct OUTSIDE the SSBO block (driver-friendly)
struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};

layout(std430, binding = 0) buffer Particles {
    Particle p[];
};

uniform int   uN;
uniform float uDT;
uniform float uSoft;
uniform float uDrag;
uniform float uMaxSpeed;

uniform float uSameRepel;
uniform float uOtherAttract;
uniform float uForceFalloff;
uniform float uBounds;

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

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec2 pos_i = p[i].pos;
    vec2 vel_i = p[i].vel;
    int  t_i   = p[i].type;

    vec2 acc = vec2(0.0);

    // O(N^2) forces
    for (int j = 0; j < uN; j++) {
        if (j == int(i)) continue;

        vec2 d = p[j].pos - pos_i;
        float r2 = dot(d, d) + uSoft;
        float invr = inversesqrt(r2);
        vec2 dir = d * invr;

        float base = 1.0 / r2;
        float mag  = pow(base, uForceFalloff);

        int t_j = p[j].type;

        if (t_j == t_i) {
            // same type repels
            acc -= dir * (uSameRepel * mag);
        } else {
            // other type attracts
            acc += dir * (uOtherAttract * mag);
        }
    }

    vel_i += acc * uDT;
    vel_i *= uDrag;
    vel_i = clampSpeed(vel_i, uMaxSpeed);
    pos_i += vel_i * uDT;

    // bounce bounds
    if (pos_i.x < -uBounds) { pos_i.x = -uBounds; vel_i.x *= -0.9; }
    if (pos_i.x >  uBounds) { pos_i.x =  uBounds; vel_i.x *= -0.9; }
    if (pos_i.y < -uBounds) { pos_i.y = -uBounds; vel_i.y *= -0.9; }
    if (pos_i.y >  uBounds) { pos_i.y =  uBounds; vel_i.y *= -0.9; }

    p[i].pos = pos_i;
    p[i].vel = vel_i;
}
"""

# ----------------------------
# GLSL: Vertex + Fragment (use #version 430 for SSBO access)
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


def main():
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    # Request OpenGL 4.3 core profile (compute shaders)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(900, 700, "GPU Two-Type Particles", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

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
    rng = np.random.default_rng(1)

    # Type A on left
    for i in range(N_PER_TYPE):
        particles["pos"][i]  = np.array([-0.5, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles["type"][i] = 0

    # Type B on right
    for i in range(N_PER_TYPE, N):
        particles["pos"][i]  = np.array([+0.5, 0.0], np.float32) + rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        particles["vel"][i]  = rng.uniform(-0.1, 0.1, 2).astype(np.float32)
        particles["type"][i] = 1

    # Upload as SSBO
    ssbo = ctx.buffer(particles.tobytes())
    ssbo.bind_to_storage_buffer(binding=0)

    compute = ctx.compute_shader(COMPUTE_SRC)
    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    vao = ctx.vertex_array(prog, [])  # gl_VertexID reads from SSBO

    # uniforms
    compute["uN"].value = N
    compute["uDT"].value = DT
    compute["uSoft"].value = SOFTENING
    compute["uDrag"].value = DRAG
    compute["uMaxSpeed"].value = MAX_SPEED
    compute["uSameRepel"].value = SAME_REPEL
    compute["uOtherAttract"].value = OTHER_ATTRACT
    compute["uForceFalloff"].value = FORCE_FALLOFF
    compute["uBounds"].value = WORLD_BOUNDS

    prog["uPointSize"].value = 1.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)

        # run compute
        groups_x = (N + 256 - 1) // 256
        compute.run(group_x=groups_x)

        # ensure compute writes visible to rendering
        ctx.memory_barrier()

        # render
        ctx.clear(0.03, 0.03, 0.04, 1.0)
        vao.render(mode=moderngl.POINTS, vertices=N)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
