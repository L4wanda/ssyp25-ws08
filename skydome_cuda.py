from datetime import datetime
from functools import lru_cache

import suncalc
from geopy.geocoders import Nominatim

import math
from math import ceil
import numpy as np
from numba import cuda, float32


EARTH_RADIUS: float32 = 6_360_000.0
ATMOSPHERE_RADIUS: float32 = 6_420_000.0

# Scattering constants (Rayleigh, Mie)
BETA_R = np.array([3.8e-6, 13.5e-6, 33.1e-6], dtype=np.float32)
BETA_M = np.array([3.0e-6, 3.0e-6, 3.0e-6], dtype=np.float32)
HM: float32 = 1_200.0
L0: float32 = 2.0

num_samples_light = 8
num_samples = 16


# ---------------------------- device functions ----------------------------- #

@cuda.jit(device=True, inline=True)
def _len_vec(v0: float32, v1: float32, v2: float32):
    return math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)


@cuda.jit(device=True, inline=True)
def _sphere_intersect(o0, o1, o2, v0, v1, v2, radius):
    b = 2.0 * (o0 * v0 + o1 * v1 + o2 * v2)
    c = (o0 * o0 + o1 * o1 + o2 * o2) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return -1.0
    sqrt_disc = math.sqrt(disc)
    t = (-b + sqrt_disc) * 0.5
    if t < 0.0:
        return -1.0
    return t


@cuda.jit(device=True, inline=True)
def _phase_rayleigh(mu):
    return (3.0 / (16.0 * math.pi)) * (1.0 + mu * mu)


@cuda.jit(device=True, inline=True)
def _phase_mie(mu):
    g = 0.76
    return (3.0 / (8.0 * math.pi)) * (1 - g * g) * (1.0 + mu * mu) / (
        (2 + g * g) * math.pow(1 + g * g - 2 * g * mu, 1.5)
    )


@cuda.jit(device=True)
def _transmittance(p1x, p1y, p1z, p2x, p2y, p2z, scale_height):
    acc = 0.0
    dx = (p2x - p1x) / num_samples_light
    dy = (p2y - p1y) / num_samples_light
    dz = (p2z - p1z) / num_samples_light
    for k in range(num_samples_light):
        sx = p1x + dx * k
        sy = p1y + dy * k
        sz = p1z + dz * k
        height = _len_vec(sx, sy, sz) - EARTH_RADIUS
        acc += math.exp(-height / scale_height)
    seg_len = _len_vec(dx, dy, dz)
    return acc * seg_len


@cuda.jit(device=True)
def _single_scatter(
    beta_r_ptr,
    beta_m_ptr,
    use_mie: bool,
    o0, o1, o2,
    v0, v1, v2,
    l0, l1, l2,
    scale_height,
    beta_idx,
):

    t1 = _sphere_intersect(o0, o1, o2, v0, v1, v2, ATMOSPHERE_RADIUS)
    if t1 < 0.0:
        return 0.0
    pax = o0 + v0 * t1
    pay = o1 + v1 * t1
    paz = o2 + v2 * t1

    step_x = (pax - o0) / num_samples
    step_y = (pay - o1) / num_samples
    step_z = (paz - o2) / num_samples

    acc = 0.0
    trans_integral = 0.0

    for i in range(num_samples - 1):
        sx = o0 + step_x * i
        sy = o1 + step_y * i
        sz = o2 + step_z * i
        # intersection towards the sun
        t_sun = _sphere_intersect(sx, sy, sz, l0, l1, l2, ATMOSPHERE_RADIUS)
        if t_sun < 0.0:
            continue
        psx = sx + l0 * t_sun
        psy = sy + l1 * t_sun
        psz = sz + l2 * t_sun

        height = _len_vec(sx, sy, sz) - EARTH_RADIUS
        trans_integral += math.exp(-height / scale_height) * _len_vec(
            pax - sx, pay - sy, paz - sz
        ) / num_samples

        scatter_coeff = beta_m_ptr[beta_idx] if use_mie else beta_r_ptr[beta_idx]
        acc += (
            math.exp(-(trans_integral + _transmittance(sx, sy, sz, psx, psy, psz, scale_height)) * scatter_coeff)
            * scatter_coeff
            * _len_vec(step_x, step_y, step_z)
        )
    return acc


# ------------------------------- GPU kernel -------------------------------- #

@cuda.jit
def render_kernel(
    out_img,
    theta0,
    phi0,
    use_mie: bool,
    width: int,
    height: int,
    scale_height_rayleigh
):
    j, i = cuda.grid(2)
    if j >= height or i >= width:
        return

    y = 2.0 * (j + 0.5) / height - 1.0
    x = 2.0 * (i + 0.5) / width - 1.0
    r = x * x + y * y
    if r > 1.0:
        out_img[j, i, 0] = 0.0
        out_img[j, i, 1] = 0.0
        out_img[j, i, 2] = 0.0
        return

    phi = math.atan2(y, x)
    theta = math.acos(1.0 - r)

    v0 = math.sin(theta) * math.cos(phi)
    v1 = math.cos(theta)
    v2 = math.sin(theta) * math.sin(phi)

    l0 = math.cos(theta0) * math.cos(phi0)
    l1 = math.sin(theta0)
    l2 = math.cos(theta0) * math.sin(phi0)

    red = _single_scatter(BETA_R, BETA_M, use_mie, 0.0, EARTH_RADIUS, 0.0,
                          v0, v1, v2, l0, l1, l2, HM if use_mie else scale_height_rayleigh, 0)
    green = _single_scatter(BETA_R, BETA_M, use_mie, 0.0, EARTH_RADIUS, 0.0,
                            v0, v1, v2, l0, l1, l2, HM if use_mie else scale_height_rayleigh, 1)
    blue = _single_scatter(BETA_R, BETA_M, use_mie, 0.0, EARTH_RADIUS, 0.0,
                           v0, v1, v2, l0, l1, l2, HM if use_mie else scale_height_rayleigh, 2)

    mu = v0 * l0 + v1 * l1 + v2 * l2  # dot(V, L)
    phase = _phase_mie(mu) if use_mie else _phase_rayleigh(mu)

    out_img[j, i, 0] = red * phase * L0
    out_img[j, i, 1] = green * phase * L0
    out_img[j, i, 2] = blue * phase * L0


class SkydomeRenderer:

    def __init__(self, use_mie: bool, latitude, longitude, width, height, scale_height_rayleigh):
        self.width = width
        self.height = height

        self.threads = (16, 16)
        self.blocks = (ceil(self.height / self.threads[0]), ceil(self.width / self.threads[1]))
        self.use_mie = use_mie
        self.d_img = cuda.device_array((self.height, self.width, 3), dtype=np.float32)
        self.latitude = latitude
        self.longitude = longitude
        self.scale_height_rayleigh = scale_height_rayleigh


    # @lru_cache(maxsize=128)
    def sun_pos(self, time):
        hours = time // 60
        minutes = time % 60
        date = datetime(2025, 7, 26, hours, minutes, 0)
        res = suncalc.get_position(date, self.longitude, self.latitude)
        return float(res["altitude"]), float(res["azimuth"])

    def render(self, hour_utc: int) -> np.ndarray:
        theta0, phi0 = self.sun_pos(hour_utc)
        render_kernel[self.blocks, self.threads](
            self.d_img, theta0, phi0, self.use_mie, self.width, self.height, self.scale_height_rayleigh
        )
        return self.d_img.copy_to_host()
