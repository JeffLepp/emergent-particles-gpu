"""Reusable GPU force solvers for the particle benchmarks.

The P3M solver splits the exact pair force into two pieces:

* a smooth far field evaluated as a free-space mesh convolution with FFTs;
* a compact near field evaluated directly from a linked-cell list.

The split is exact before mesh discretization.  Increasing ``mesh_res`` makes
the remaining interpolation/FFT error smaller.  Inputs are read-only and
accelerations are written to a separate buffer, so both this solver and the
brute-force reference represent one synchronous time step.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np


PARTICLE_DTYPE = np.dtype([
    ("pos", np.float32, (2,)),
    ("vel", np.float32, (2,)),
    ("type", np.int32),
    ("pad0", np.int32),
])


PARTICLE_STRUCT = r"""
struct Particle {
    vec2 pos;
    vec2 vel;
    int type;
    int pad0;
};
"""


CLEAR_MESH_SRC = r"""
#version 430
layout(std430, binding = 1) buffer MeshDensity { vec2 density[]; };
uniform int uCount;
layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(uCount)) density[i] = vec2(0.0);
}
"""


CLEAR_HEAD_SRC = r"""
#version 430
layout(std430, binding = 2) buffer NearHead { int head[]; };
uniform int uCount;
layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(uCount)) head[i] = -1;
}
"""


BUILD_SRC = r"""
#version 430
#extension GL_NV_shader_atomic_float : require
""" + PARTICLE_STRUCT + r"""
layout(std430, binding = 0) readonly buffer Particles { Particle p[]; };
layout(std430, binding = 1) buffer MeshDensity { float density[]; };
layout(std430, binding = 2) buffer NearHead { int head[]; };
layout(std430, binding = 3) buffer NearNext { int nextIdx[]; };

uniform int uN;
uniform int uMeshRes;
uniform int uPadRes;
uniform int uNearGridRes;
uniform float uBounds;
layout(local_size_x = 256) in;

void addDensity(int node, float mass, float charge) {
    atomicAdd(density[2 * node], mass);
    atomicAdd(density[2 * node + 1], charge);
}

void main() {
    uint ui = gl_GlobalInvocationID.x;
    if (ui >= uint(uN)) return;
    int i = int(ui);
    vec2 pos = p[i].pos;

    // Cloud-in-cell deposition onto the physical (unpadded) mesh.
    vec2 g = clamp((pos + vec2(uBounds)) *
                   (float(uMeshRes - 1) / (2.0 * uBounds)),
                   vec2(0.0), vec2(float(uMeshRes - 1)));
    ivec2 b = ivec2(floor(g));
    ivec2 n = min(b + ivec2(1), ivec2(uMeshRes - 1));
    vec2 f = g - vec2(b);
    float q = (p[i].type == 0) ? 1.0 : -1.0;

    int i00 = b.x + b.y * uPadRes;
    int i10 = n.x + b.y * uPadRes;
    int i01 = b.x + n.y * uPadRes;
    int i11 = n.x + n.y * uPadRes;
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;
    addDensity(i00, w00, q * w00);
    addDensity(i10, w10, q * w10);
    addDensity(i01, w01, q * w01);
    addDensity(i11, w11, q * w11);

    // A coarse cell list is sufficient for the compact direct correction.
    vec2 p01 = clamp((pos + vec2(uBounds)) / (2.0 * uBounds),
                     vec2(0.0), vec2(0.99999994));
    ivec2 c = ivec2(floor(p01 * float(uNearGridRes)));
    int cell = c.x + c.y * uNearGridRes;
    int old = atomicExchange(head[cell], i);
    nextIdx[i] = old;
}
"""


MESH_TO_COMPLEX_SRC = r"""
#version 430
layout(std430, binding = 1) readonly buffer MeshDensity { vec2 density[]; };
layout(std430, binding = 4) writeonly buffer Spectrum { vec2 z[]; };
uniform int uCount;
layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(uCount)) z[i] = density[i];
}
"""


FFT_BIT_REVERSE_SRC = r"""
#version 430
layout(std430, binding = 8) buffer FFTData { vec2 z[]; };
uniform int uSize;
uniform int uBits;
uniform int uAxis;
layout(local_size_x = 256) in;

void main() {
    uint flatIndex = gl_GlobalInvocationID.x;
    uint count = uint(uSize * uSize);
    if (flatIndex >= count) return;
    uint line = flatIndex / uint(uSize);
    uint c = flatIndex - line * uint(uSize);
    uint r = bitfieldReverse(c) >> uint(32 - uBits);
    if (r <= c) return;
    uint a;
    uint b;
    if (uAxis == 0) {
        a = line * uint(uSize) + c;
        b = line * uint(uSize) + r;
    } else {
        a = c * uint(uSize) + line;
        b = r * uint(uSize) + line;
    }
    vec2 t = z[a];
    z[a] = z[b];
    z[b] = t;
}
"""


FFT_STAGE_SRC = r"""
#version 430
layout(std430, binding = 8) buffer FFTData { vec2 z[]; };
uniform int uSize;
uniform int uStage;
uniform int uAxis;
uniform int uInverse;
layout(local_size_x = 256) in;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y,
                a.x * b.y + a.y * b.x);
}

void main() {
    uint flatIndex = gl_GlobalInvocationID.x;
    uint butterflies = uint(uSize * uSize / 2);
    if (flatIndex >= butterflies) return;
    uint perLine = uint(uSize / 2);
    uint line = flatIndex / perLine;
    uint pair = flatIndex - line * perLine;
    uint halfStage = uint(uStage / 2);
    uint group = pair / halfStage;
    uint j = pair - group * halfStage;
    uint c0 = group * uint(uStage) + j;
    uint c1 = c0 + halfStage;
    uint i0;
    uint i1;
    if (uAxis == 0) {
        i0 = line * uint(uSize) + c0;
        i1 = line * uint(uSize) + c1;
    } else {
        i0 = c0 * uint(uSize) + line;
        i1 = c1 * uint(uSize) + line;
    }

    float signValue = (uInverse != 0) ? 1.0 : -1.0;
    float angle = signValue * 6.283185307179586 * float(j) / float(uStage);
    vec2 w = vec2(cos(angle), sin(angle));
    vec2 a = z[i0];
    vec2 b = cmul(z[i1], w);
    z[i0] = a + b;
    z[i1] = a - b;
}
"""


SPECTRUM_MULTIPLY_SRC = r"""
#version 430
layout(std430, binding = 4) readonly buffer Spectrum { vec2 sourceZ[]; };
layout(std430, binding = 6) readonly buffer Kernel { vec2 kernelZ[]; };
layout(std430, binding = 5) writeonly buffer Work { vec2 resultZ[]; };
uniform int uCount;
layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount)) return;
    vec2 a = sourceZ[i];
    vec2 b = kernelZ[i];
    resultZ[i] = vec2(a.x * b.x - a.y * b.y,
                      a.x * b.y + a.y * b.x);
}
"""


GATHER_SRC = r"""
#version 430
""" + PARTICLE_STRUCT + r"""
layout(std430, binding = 0) readonly buffer Particles { Particle p[]; };
layout(std430, binding = 2) readonly buffer NearHead { int head[]; };
layout(std430, binding = 3) readonly buffer NearNext { int nextIdx[]; };
layout(std430, binding = 5) readonly buffer FieldX { vec2 fieldX[]; };
layout(std430, binding = 9) readonly buffer FieldY { vec2 fieldY[]; };
layout(std430, binding = 7) writeonly buffer Accelerations { vec2 acceleration[]; };

uniform int uN;
uniform int uMeshRes;
uniform int uPadRes;
uniform int uNearGridRes;
uniform float uNearRadius;
uniform float uBounds;
uniform float uSoft;
uniform float uFalloff;
uniform float uSameRepel;
uniform float uOtherAttract;
uniform float uFFTScale;
layout(local_size_x = 256) in;

vec2 sampleComplexField(vec2 pos, bool useY) {
    vec2 g = clamp((pos + vec2(uBounds)) *
                   (float(uMeshRes - 1) / (2.0 * uBounds)),
                   vec2(0.0), vec2(float(uMeshRes - 1)));
    ivec2 b = ivec2(floor(g));
    ivec2 n = min(b + ivec2(1), ivec2(uMeshRes - 1));
    vec2 f = g - vec2(b);
    int i00 = b.x + b.y * uPadRes;
    int i10 = n.x + b.y * uPadRes;
    int i01 = b.x + n.y * uPadRes;
    int i11 = n.x + n.y * uPadRes;
    vec2 a00 = useY ? fieldY[i00] : fieldX[i00];
    vec2 a10 = useY ? fieldY[i10] : fieldX[i10];
    vec2 a01 = useY ? fieldY[i01] : fieldX[i01];
    vec2 a11 = useY ? fieldY[i11] : fieldX[i11];
    return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y) * uFFTScale;
}

float smootherstep(float x) {
    x = clamp(x, 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

void main() {
    uint ui = gl_GlobalInvocationID.x;
    if (ui >= uint(uN)) return;
    int i = int(ui);
    vec2 pos_i = p[i].pos;
    int type_i = p[i].type;
    float q_i = (type_i == 0) ? 1.0 : -1.0;

    // One complex FFT carries both total density (real) and signed type
    // density (imaginary).  These two fields exactly represent the 2x2
    // same/other interaction matrix.
    vec2 fx = sampleComplexField(pos_i, false);
    vec2 fy = sampleComplexField(pos_i, true);
    float c0 = 0.5 * (uOtherAttract - uSameRepel);
    float c1 = -0.5 * (uSameRepel + uOtherAttract);
    vec2 acc = vec2(c0 * fx.x + c1 * q_i * fx.y,
                    c0 * fy.x + c1 * q_i * fy.y);

    // Add K_exact - K_far inside the compact split radius.
    vec2 p01 = clamp((pos_i + vec2(uBounds)) / (2.0 * uBounds),
                     vec2(0.0), vec2(0.99999994));
    ivec2 ci = ivec2(floor(p01 * float(uNearGridRes)));
    float cellSize = (2.0 * uBounds) / float(uNearGridRes);
    int rCells = int(ceil(uNearRadius / cellSize));
    float near2 = uNearRadius * uNearRadius;

    for (int oy = -rCells; oy <= rCells; ++oy) {
        for (int ox = -rCells; ox <= rCells; ++ox) {
            ivec2 cc = ci + ivec2(ox, oy);
            if (cc.x < 0 || cc.y < 0 ||
                cc.x >= uNearGridRes || cc.y >= uNearGridRes) continue;
            int j = head[cc.x + cc.y * uNearGridRes];
            while (j != -1) {
                if (j != i) {
                    vec2 d = p[j].pos - pos_i;
                    float geometricR2 = dot(d, d);
                    if (geometricR2 < near2) {
                        float softenedR2 = geometricR2 + uSoft;
                        float invr = inversesqrt(softenedR2);
                        float mag = pow(1.0 / softenedR2, uFalloff);
                        float farWeight = smootherstep(sqrt(geometricR2) / uNearRadius);
                        float coeff = (p[j].type == type_i) ? -uSameRepel : uOtherAttract;
                        acc += coeff * d * invr * mag * (1.0 - farWeight);
                    }
                }
                j = nextIdx[j];
            }
        }
    }
    acceleration[i] = acc;
}
"""


BRUTE_FORCE_SRC = r"""
#version 430
""" + PARTICLE_STRUCT + r"""
layout(std430, binding = 0) readonly buffer Particles { Particle p[]; };
layout(std430, binding = 7) writeonly buffer Accelerations { vec2 acceleration[]; };
uniform int uN;
uniform float uSoft;
uniform float uFalloff;
uniform float uSameRepel;
uniform float uOtherAttract;
layout(local_size_x = 256) in;
void main() {
    uint ui = gl_GlobalInvocationID.x;
    if (ui >= uint(uN)) return;
    int i = int(ui);
    vec2 pos_i = p[i].pos;
    int type_i = p[i].type;
    vec2 acc = vec2(0.0);
    for (int j = 0; j < uN; ++j) {
        if (j == i) continue;
        vec2 d = p[j].pos - pos_i;
        float r2 = dot(d, d) + uSoft;
        float invr = inversesqrt(r2);
        float mag = pow(1.0 / r2, uFalloff);
        float coeff = (p[j].type == type_i) ? -uSameRepel : uOtherAttract;
        acc += coeff * d * invr * mag;
    }
    acceleration[i] = acc;
}
"""


INTEGRATE_SRC = r"""
#version 430
""" + PARTICLE_STRUCT + r"""
layout(std430, binding = 0) readonly buffer InputParticles { Particle sourceP[]; };
layout(std430, binding = 7) readonly buffer Accelerations { vec2 acceleration[]; };
layout(std430, binding = 10) writeonly buffer OutputParticles { Particle targetP[]; };
uniform int uN;
uniform float uDT;
uniform float uDrag;
uniform float uMaxSpeed;
uniform float uBounds;
layout(local_size_x = 256) in;

vec2 clampSpeed(vec2 v, float maxS) {
    float s2 = dot(v, v);
    float m2 = maxS * maxS;
    if (s2 > m2) return v * (maxS * inversesqrt(s2));
    return v;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    Particle outP = sourceP[i];
    outP.vel += acceleration[i] * uDT;
    outP.vel *= uDrag;
    outP.vel = clampSpeed(outP.vel, uMaxSpeed);
    outP.pos += outP.vel * uDT;
    if (outP.pos.x < -uBounds) { outP.pos.x = -uBounds; outP.vel.x *= -0.9; }
    if (outP.pos.x >  uBounds) { outP.pos.x =  uBounds; outP.vel.x *= -0.9; }
    if (outP.pos.y < -uBounds) { outP.pos.y = -uBounds; outP.vel.y *= -0.9; }
    if (outP.pos.y >  uBounds) { outP.pos.y =  uBounds; outP.vel.y *= -0.9; }
    targetP[i] = outP;
}
"""


class _GpuSolverBase:
    def __init__(self, ctx: moderngl.Context, capacity: int):
        self.ctx = ctx
        self.capacity = int(capacity)
        self.acceleration = ctx.buffer(reserve=self.capacity * 8)

    def read_acceleration(self, n: int) -> np.ndarray:
        raw = self.acceleration.read(size=int(n) * 8)
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


class BruteForceSolver(_GpuSolverBase):
    """Race-free, force-only all-pairs reference on the GPU."""

    def __init__(self, ctx, capacity, softening=0.02, force_falloff=2.0,
                 same_repel=1.0, other_attract=1.001):
        super().__init__(ctx, capacity)
        self.shader = ctx.compute_shader(BRUTE_FORCE_SRC)
        self.shader["uSoft"].value = float(softening)
        self.shader["uFalloff"].value = float(force_falloff)
        self.shader["uSameRepel"].value = float(same_repel)
        self.shader["uOtherAttract"].value = float(other_attract)

    def compute(self, particles: moderngl.Buffer, n: int) -> None:
        particles.bind_to_storage_buffer(0)
        self.acceleration.bind_to_storage_buffer(7)
        self.shader["uN"].value = int(n)
        self.shader.run(group_x=(int(n) + 255) // 256)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)


class P3MForceSolver(_GpuSolverBase):
    """GPU particle-mesh force with an exact compact near correction."""

    def __init__(self, ctx, capacity, mesh_res=256, bounds=1.0,
                 softening=0.02, force_falloff=2.0, same_repel=1.0,
                 other_attract=1.001, near_cells=0.5):
        super().__init__(ctx, capacity)
        if mesh_res < 16 or mesh_res & (mesh_res - 1):
            raise ValueError("mesh_res must be a power of two >= 16")
        if "GL_NV_shader_atomic_float" not in ctx.extensions:
            raise RuntimeError("P3M currently requires GL_NV_shader_atomic_float")

        self.mesh_res = int(mesh_res)
        self.pad_res = 2 * self.mesh_res
        self.pad_count = self.pad_res * self.pad_res
        self.bits = int(math.log2(self.pad_res))
        self.bounds = float(bounds)
        self.softening = float(softening)
        self.force_falloff = float(force_falloff)
        self.same_repel = float(same_repel)
        self.other_attract = float(other_attract)
        self.mesh_spacing = 2.0 * self.bounds / float(self.mesh_res - 1)
        self.near_radius = float(near_cells) * self.mesh_spacing
        self.near_grid_res = max(1, int(math.floor(2.0 * self.bounds / self.near_radius)))
        self.near_cell_count = self.near_grid_res * self.near_grid_res

        self.density = ctx.buffer(reserve=self.pad_count * 8)
        self.near_head = ctx.buffer(reserve=self.near_cell_count * 4)
        self.near_next = ctx.buffer(reserve=self.capacity * 4)
        self.spectrum = ctx.buffer(reserve=self.pad_count * 8)
        self.field_x = ctx.buffer(reserve=self.pad_count * 8)
        self.field_y = ctx.buffer(reserve=self.pad_count * 8)
        kernel_x, kernel_y = self._make_kernel_spectra()
        self.kernel_x = ctx.buffer(kernel_x)
        self.kernel_y = ctx.buffer(kernel_y)

        self.clear_mesh = ctx.compute_shader(CLEAR_MESH_SRC)
        self.clear_head = ctx.compute_shader(CLEAR_HEAD_SRC)
        self.build = ctx.compute_shader(BUILD_SRC)
        self.mesh_to_complex = ctx.compute_shader(MESH_TO_COMPLEX_SRC)
        self.bit_reverse = ctx.compute_shader(FFT_BIT_REVERSE_SRC)
        self.fft_stage = ctx.compute_shader(FFT_STAGE_SRC)
        self.multiply = ctx.compute_shader(SPECTRUM_MULTIPLY_SRC)
        self.gather = ctx.compute_shader(GATHER_SRC)

        self.clear_mesh["uCount"].value = self.pad_count
        self.clear_head["uCount"].value = self.near_cell_count
        for name, value in (
            ("uMeshRes", self.mesh_res),
            ("uPadRes", self.pad_res),
            ("uNearGridRes", self.near_grid_res),
        ):
            self.build[name].value = value
            self.gather[name].value = value
        self.build["uBounds"].value = self.bounds
        for name, value in (
            ("uNearRadius", self.near_radius),
            ("uBounds", self.bounds),
            ("uSoft", self.softening),
            ("uFalloff", self.force_falloff),
            ("uSameRepel", self.same_repel),
            ("uOtherAttract", self.other_attract),
            ("uFFTScale", 1.0 / float(self.pad_count)),
        ):
            self.gather[name].value = value
        self.mesh_to_complex["uCount"].value = self.pad_count
        self.multiply["uCount"].value = self.pad_count
        self.bit_reverse["uSize"].value = self.pad_res
        self.bit_reverse["uBits"].value = self.bits
        self.fft_stage["uSize"].value = self.pad_res

    @staticmethod
    def _smootherstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x**3 * (x * (x * 6.0 - 15.0) + 10.0)

    def _make_kernel_spectra(self):
        p = self.pad_res
        r = self.mesh_res
        coord = np.arange(p, dtype=np.float64)
        coord = np.where(coord < r, coord, coord - p) * self.mesh_spacing
        yy, xx = np.meshgrid(coord, coord, indexing="ij")
        geometric_r2 = xx * xx + yy * yy
        softened_r2 = geometric_r2 + self.softening
        weight = softened_r2 ** (-(self.force_falloff + 0.5))
        radius = np.sqrt(geometric_r2)
        far_weight = self._smootherstep(radius / self.near_radius)

        # Convolution is indexed by target-source, while the original shader
        # uses d=source-target, hence the minus sign.
        kernel_x = -xx * weight * far_weight
        kernel_y = -yy * weight * far_weight
        kernel_x[0, 0] = 0.0
        kernel_y[0, 0] = 0.0
        kx = np.fft.fft2(kernel_x).astype(np.complex64)
        ky = np.fft.fft2(kernel_y).astype(np.complex64)
        packed_x = np.stack((kx.real, kx.imag), axis=-1).astype(np.float32)
        packed_y = np.stack((ky.real, ky.imag), axis=-1).astype(np.float32)
        return packed_x.tobytes(), packed_y.tobytes()

    def _barrier(self):
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

    def _fft(self, data: moderngl.Buffer, inverse: bool) -> None:
        data.bind_to_storage_buffer(8)
        groups_values = (self.pad_count + 255) // 256
        groups_butterflies = (self.pad_count // 2 + 255) // 256
        self.fft_stage["uInverse"].value = 1 if inverse else 0
        for axis in (0, 1):
            self.bit_reverse["uAxis"].value = axis
            self.bit_reverse.run(group_x=groups_values)
            self._barrier()
            self.fft_stage["uAxis"].value = axis
            stage = 2
            while stage <= self.pad_res:
                self.fft_stage["uStage"].value = stage
                self.fft_stage.run(group_x=groups_butterflies)
                self._barrier()
                stage *= 2

    def compute(self, particles: moderngl.Buffer, n: int) -> None:
        n = int(n)
        particles.bind_to_storage_buffer(0)
        self.density.bind_to_storage_buffer(1)
        self.near_head.bind_to_storage_buffer(2)
        self.near_next.bind_to_storage_buffer(3)
        self.spectrum.bind_to_storage_buffer(4)
        self.acceleration.bind_to_storage_buffer(7)

        self.clear_mesh.run(group_x=(self.pad_count + 255) // 256)
        self.clear_head.run(group_x=(self.near_cell_count + 255) // 256)
        self._barrier()

        self.build["uN"].value = n
        self.build.run(group_x=(n + 255) // 256)
        self._barrier()

        self.mesh_to_complex.run(group_x=(self.pad_count + 255) // 256)
        self._barrier()
        self._fft(self.spectrum, inverse=False)

        self.kernel_x.bind_to_storage_buffer(6)
        self.field_x.bind_to_storage_buffer(5)
        self.multiply.run(group_x=(self.pad_count + 255) // 256)
        self._barrier()
        self._fft(self.field_x, inverse=True)

        self.kernel_y.bind_to_storage_buffer(6)
        self.field_y.bind_to_storage_buffer(5)
        self.multiply.run(group_x=(self.pad_count + 255) // 256)
        self._barrier()
        self._fft(self.field_y, inverse=True)

        self.field_x.bind_to_storage_buffer(5)
        self.field_y.bind_to_storage_buffer(9)
        self.gather["uN"].value = n
        self.gather.run(group_x=(n + 255) // 256)
        self._barrier()


class GpuIntegrator:
    """Shared ping-pong integrator used by both force solvers."""

    def __init__(self, ctx, dt=1.0 / 90.0, drag=1.0, max_speed=2.0,
                 bounds=1.0):
        self.ctx = ctx
        self.shader = ctx.compute_shader(INTEGRATE_SRC)
        self.shader["uDT"].value = float(dt)
        self.shader["uDrag"].value = float(drag)
        self.shader["uMaxSpeed"].value = float(max_speed)
        self.shader["uBounds"].value = float(bounds)

    def step(self, source, acceleration, target, n):
        source.bind_to_storage_buffer(0)
        acceleration.bind_to_storage_buffer(7)
        target.bind_to_storage_buffer(10)
        self.shader["uN"].value = int(n)
        self.shader.run(group_x=(int(n) + 255) // 256)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)


def make_particles(n: int, seed: int = 1, extent: float = 0.3) -> np.ndarray:
    """Make a balanced, shuffled two-type snapshot used by all comparisons."""
    rng = np.random.default_rng(seed)
    result = np.zeros(int(n), dtype=PARTICLE_DTYPE)
    result["pos"] = rng.uniform(-extent, extent, (n, 2)).astype(np.float32)
    result["vel"] = rng.uniform(-0.1, 0.1, (n, 2)).astype(np.float32)
    result["type"] = np.arange(n, dtype=np.int32) & 1
    rng.shuffle(result)
    return result
