"""Pygame animation: 4 controllers running simultaneously with scenario presets.

Controls:
    LEFT/RIGHT  Switch platform (Unicycle / AUV / Drone)
    5-8         Scenario presets (see below)
    1-4         Trajectory: Lemniscate / Multi-Freq / Chirp / Composite (unicycle only)
    SPACE       Pause/Resume
    R           Restart simulation
    H           Toggle help overlay
    Q/ESC       Quit

Scenarios:
    5  Nominal       All defaults -- everyone performs well
    6  Wind          Strong wind disturbance -- LQR has no integral action
    7  Heavy+Fast    Mass 3.5x -- NN Frozen degrades, ACE adapts best
    8  Noisy shift   Mass 2.5x + sensor noise + wind -- ACE rejects noise

Controllers:
    LQR Expert      Nominal-gain LQR, fresh state (no delay)
    NN Frozen       Pre-trained NN, no online updates
    NN ER           Uniform replay (Experience Replay baseline)
    NN ACE (ours)   Active Continual Learning with Evidential Uncertainty
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, math, copy
import numpy as np
import torch

try:
    import pygame
except ImportError:
    sys.exit("Install pygame:  pip install pygame")

from config import Config
from envs.unicycle_env import UnicycleEnv
from envs.auv3d_env import AUV3DEnv
from envs.drone3d_env import Drone3DEnv
from models.evidential_net import EvidentialNet, evidential_loss, epistemic_score
from controllers.lqr_controller import LQRController
from controllers.lqr_3d import LQR3D
from controllers.nn_controller import NNController, compute_desired_next_state
from training.train_online import UniformCLAdapter, ACEAdapter
from utils.trajectories import (
    generate_trajectory, lemniscate_2d, lemniscate_3d,
)
from utils.exciting_trajectories import generate_exciting_trajectory

# ======================================================================
# Layout & Colors
# ======================================================================

PANEL_W = 280; SIM_W = 1200; SIM_H = 900; TOTAL_W = SIM_W + PANEL_W

BG          = (25, 25, 35)
PANEL_BG    = (18, 18, 28)
GRID        = (45, 45, 55)
REF         = (120, 120, 130)
TARGET      = (255, 220, 50)
TEXT        = (210, 210, 210)
TEXT_DIM    = (130, 130, 140)
SLIDER_BG   = (50, 50, 65)
SLIDER_FG   = (90, 90, 110)
SLIDER_KNOB = (200, 200, 220)
SLIDER_ACT  = (100, 180, 255)
SECTION_COL = (80, 160, 255)
WARN_COL    = (255, 80, 80)
GOOD_COL    = (80, 220, 80)

# Visual tuning — short translucent trails, small translucent robots.
MAX_TRAIL_LEN = 60         # very short tail (was effectively unbounded at 2000)
TRAIL_ALPHA   = 110        # 0..255 — lower = more transparent
ROBOT_ALPHA   = 170
ROBOT_SIZE    = 0.10       # in metres — small silhouette for all platforms
HALO_ALPHA    = 80
SHADOW_ALPHA  = 70          # ground shadow opacity for 3-D platforms
# Pseudo-3-D z to screen-pixel mapping (per platform). Larger means
# height variation reads more strongly on screen. Tuned so a typical
# z swing is visible without dominating the frame.
Z_VERTICAL_PPM = {"unicycle": 0.0, "auv": 80.0, "drone": 200.0}

CONTROLLERS = [
    {"name": "LQR Expert", "color": (0, 230, 0),   "shape": "triangle", "short": "LQR"},
    {"name": "NN Frozen",  "color": (50, 130, 255), "shape": "diamond",  "short": "Frozen"},
    {"name": "NN ER",      "color": (180, 80, 255), "shape": "square",   "short": "ER"},
    {"name": "NN ACE",     "color": (255, 60, 120), "shape": "pentagon", "short": "ACE"},
]

TRAJ_MAP    = {1: 'lemniscate', 2: 'multi_freq', 3: 'chirp', 4: 'composite'}
TRAJ_LABELS = {1: 'Lemniscate', 2: 'Multi-Freq', 3: 'Chirp', 4: 'Composite'}

# ======================================================================
# Platform configuration
# ======================================================================

PLATFORMS       = ["unicycle", "auv", "drone"]
PLATFORM_TITLES = {
    "unicycle": "Unicycle (2-D)",
    "auv":      "BlueROV2 AUV (3-D)",
    "drone":    "Quadrotor (3-D)",
}
# Per-platform: dt, model checkpoint, pixels-per-metre for display,
# LQR kind, total simulation length, position dimensionality.
PLATFORM_CFG = {
    "unicycle": dict(dt=0.02, ckpt="pretrained/unicycle.pth",
                     ppm=60,  lqr_kind=None,  total_time=90.0,
                     pos_dim=2),
    "auv":      dict(dt=0.05, ckpt="pretrained/auv.pth",
                     ppm=200, lqr_kind="auv", total_time=90.0,
                     pos_dim=3),
    "drone":    dict(dt=0.02, ckpt="pretrained/drone.pth",
                     ppm=480, lqr_kind="drone", total_time=60.0,
                     pos_dim=3),
}

# ======================================================================
# Scenarios
# ======================================================================

SCENARIOS = {
    5: {"name": "Nominal",
        "desc": "All defaults. Everyone tracks well. Baseline reference.",
        "params": {"mass": 1.0, "friction": 1.0, "wind_x": 0.0, "wind_y": 0.0,
                   "noise": 0.0, "state_noise": 0.0, "traj_speed": 0.5}},
    6: {"name": "Wind",
        "desc": "Strong wind disturbance.  LQR has no integral action so its tracking error grows; NN methods adapt the feed-forward.",
        "params": {"mass": 1.0, "friction": 1.0, "wind_x": 1.0, "wind_y": 0.5,
                   "noise": 0.0, "state_noise": 0.0, "traj_speed": 0.7}},
    7: {"name": "Heavy+Fast",
        "desc": "Mass 3.5x + fast traj. NN Frozen undershoots (wrong torque scale). ER/ACE adapt.",
        "params": {"mass": 3.5, "friction": 1.5, "wind_x": 2.0, "wind_y": 0.0,
                   "noise": 0.0, "state_noise": 0.0, "traj_speed": 0.8}},
    8: {"name": "Noisy Shift",
        "desc": "Mass 2.5x + sensor noise + wind. ER corrupted by noise. ACE rejects noisy samples.",
        "params": {"mass": 2.5, "friction": 1.5, "wind_x": 3.0, "wind_y": 2.0,
                   "noise": 0.5, "state_noise": 3.0, "traj_speed": 0.6}},
}

# ======================================================================
# Slider
# ======================================================================

class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, default, fmt="{:.2f}",
                 color=SLIDER_ACT, snap_to_default=False):
        self.rect = pygame.Rect(x, y, w, 18)
        self.label, self.vmin, self.vmax = label, vmin, vmax
        self.value, self.default = default, default
        self.fmt, self.color = fmt, color
        self.snap_to_default = snap_to_default
        self.dragging = False
        self.knob_r = 7
        self.label_y = y - 15

    @property
    def norm(self):
        return (self.value - self.vmin) / (self.vmax - self.vmin + 1e-12)

    @norm.setter
    def norm(self, n):
        n = max(0.0, min(1.0, n))
        self.value = self.vmin + n * (self.vmax - self.vmin)
        if self.snap_to_default and abs(self.value - self.default) < (self.vmax - self.vmin) * 0.03:
            self.value = self.default

    @property
    def knob_x(self):
        return self.rect.x + int(self.norm * self.rect.w)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            kx, ky = self.knob_x, self.rect.y + self.rect.h // 2
            if abs(ev.pos[0]-kx) < self.knob_r+4 and abs(ev.pos[1]-ky) < self.knob_r+4:
                self.dragging = True; return True
            if self.rect.collidepoint(ev.pos):
                self.norm = (ev.pos[0] - self.rect.x) / self.rect.w
                self.dragging = True; return True
        elif ev.type == pygame.MOUSEBUTTONUP: self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self.norm = (ev.pos[0] - self.rect.x) / self.rect.w; return True
        return False

    def reset(self): self.value = self.default

    def draw(self, surf, font):
        is_def = abs(self.value - self.default) < 1e-6
        surf.blit(font.render(f"{self.label}: {self.fmt.format(self.value)}", True,
                  TEXT_DIM if is_def else self.color), (self.rect.x, self.label_y))
        cy = self.rect.y + self.rect.h // 2
        pygame.draw.rect(surf, SLIDER_BG, (self.rect.x, cy-3, self.rect.w, 6), border_radius=3)
        fw = int(self.norm * self.rect.w)
        if fw > 0:
            pygame.draw.rect(surf, (self.color if not is_def else SLIDER_FG),
                             (self.rect.x, cy-3, fw, 6), border_radius=3)
        def_x = self.rect.x + int((self.default-self.vmin)/(self.vmax-self.vmin+1e-12)*self.rect.w)
        pygame.draw.line(surf, TEXT_DIM, (def_x, cy-6), (def_x, cy+6), 1)
        kx = self.knob_x
        pygame.draw.circle(surf, SLIDER_ACT if self.dragging else SLIDER_KNOB, (kx,cy), self.knob_r)
        pygame.draw.circle(surf, (255,255,255), (kx,cy), self.knob_r, 2)

# ======================================================================
# Drawing helpers
# ======================================================================

def w2s(x, y, cx, cy, ppm):
    return int(cx + x * ppm), int(cy - y * ppm)

def w2s_3d(x, y, z, cx, cy, ppm, platform):
    """Project (x, y, z) world coordinate to screen, lifting by z via
    a per-platform vertical mapping so height reads visually."""
    sx, sy = w2s(x, y, cx, cy, ppm)
    sy -= int(z * Z_VERTICAL_PPM.get(platform, 0.0))
    return sx, sy

def _draw_shadow(surf, x, y, cx, cy, ppm, platform, size_px):
    """Faint ground shadow at z=0 (only meaningful for 3-D platforms)."""
    if Z_VERTICAL_PPM.get(platform, 0.0) <= 0.0:
        return
    sx, sy = w2s(x, y, cx, cy, ppm)
    rx = max(2, int(size_px * 0.9))
    ry = max(1, int(size_px * 0.32))
    rect = pygame.Rect(sx - rx, sy - ry, 2 * rx, 2 * ry)
    pygame.draw.ellipse(surf, (10, 10, 14, SHADOW_ALPHA), rect)

def _draw_quadcopter(surf, sx, sy, th, s, fill, edge):
    """Top-down quadcopter: 4 rotor disks on an X-arm body."""
    ct, st = math.cos(th), math.sin(th)
    def rot(bx, by):
        return sx + int(bx * ct - by * st), sy - int(bx * st + by * ct)
    arm = s
    rotor_r = max(2, int(s * 0.55))
    arms = [rot(arm, arm), rot(-arm, -arm), rot(arm, -arm), rot(-arm, arm)]
    pygame.draw.line(surf, edge, arms[0], arms[1], 2)
    pygame.draw.line(surf, edge, arms[2], arms[3], 2)
    for ax, ay in arms:
        pygame.draw.circle(surf, fill, (ax, ay), rotor_r)
        pygame.draw.circle(surf, edge, (ax, ay), rotor_r, 1)
    body_r = max(2, int(s * 0.35))
    pygame.draw.circle(surf, fill, (sx, sy), body_r)
    pygame.draw.circle(surf, edge, (sx, sy), body_r, 1)
    nose = rot(s * 1.5, 0)
    pygame.draw.line(surf, edge, (sx, sy), nose, 1)

def _draw_cylinder(surf, sx, sy, th, s, fill, edge):
    """Side-on cylinder for an AUV body: rectangle hull + two end caps."""
    ct, st = math.cos(th), math.sin(th)
    def rot(bx, by):
        return sx + int(bx * ct - by * st), sy - int(bx * st + by * ct)
    half_l = s * 1.4
    half_w = s * 0.55
    pts = [rot(half_l, half_w), rot(half_l, -half_w),
           rot(-half_l, -half_w), rot(-half_l, half_w)]
    pygame.draw.polygon(surf, fill, pts)
    pygame.draw.polygon(surf, edge, pts, 1)
    cap_r = max(2, int(half_w))
    front = rot(half_l, 0)
    back = rot(-half_l, 0)
    pygame.draw.circle(surf, fill, front, cap_r)
    pygame.draw.circle(surf, edge, front, cap_r, 1)
    pygame.draw.circle(surf, fill, back, cap_r)
    pygame.draw.circle(surf, edge, back, cap_r, 1)
    fin = rot(-half_l * 0.9, half_w * 1.6)
    pygame.draw.line(surf, edge, back, fin, 1)

def draw_robot(surf, x, y, z, th, cx, cy, ppm, color, shape="triangle",
               platform="unicycle", size=ROBOT_SIZE, alpha=ROBOT_ALPHA):
    """Draw a robot marker with translucent fill on an SRCALPHA `surf`.

    For 3-D platforms the marker is lifted by `z` and a faint ground
    shadow is drawn at z=0.  Drone uses a quadcopter glyph and AUV uses
    a cylinder regardless of the controller's shape, with the controller
    colour distinguishing them; on the unicycle the per-controller
    shape is preserved.
    """
    s = max(3, int(size * ppm))
    fill = (color[0], color[1], color[2], alpha)
    edge = (255, 255, 255, min(255, alpha + 60))
    if platform in ("auv", "drone"):
        _draw_shadow(surf, x, y, cx, cy, ppm, platform, s)
    sx, sy = w2s_3d(x, y, z, cx, cy, ppm, platform)
    if platform == "drone":
        _draw_quadcopter(surf, sx, sy, th, s, fill, edge)
        return
    if platform == "auv":
        _draw_cylinder(surf, sx, sy, th, s, fill, edge)
        return
    ct, st = math.cos(th), math.sin(th)
    def rot(bx, by): return sx+int(bx*ct-by*st), sy-int(bx*st+by*ct)
    if shape == "triangle":
        pts = [rot(s,0), rot(-s*0.6,s*0.55), rot(-s*0.6,-s*0.55)]
        pygame.draw.polygon(surf, fill, pts); pygame.draw.polygon(surf, edge, pts, 1)
    elif shape == "diamond":
        pts = [rot(s,0), rot(0,s*0.55), rot(-s,0), rot(0,-s*0.55)]
        pygame.draw.polygon(surf, fill, pts); pygame.draw.polygon(surf, edge, pts, 1)
    elif shape == "square":
        hs = int(s*0.55)
        pts = [rot(hs,hs), rot(hs,-hs), rot(-hs,-hs), rot(-hs,hs)]
        pygame.draw.polygon(surf, fill, pts); pygame.draw.polygon(surf, edge, pts, 1)
    elif shape == "circle":
        pygame.draw.circle(surf, fill, (sx,sy), s)
        pygame.draw.circle(surf, edge, (sx,sy), s, 1)
    elif shape == "pentagon":
        pts = [rot(s, 0),
               rot(s*0.31, s*0.95),
               rot(-s*0.81, s*0.59),
               rot(-s*0.81, -s*0.59),
               rot(s*0.31, -s*0.95)]
        pygame.draw.polygon(surf, fill, pts); pygame.draw.polygon(surf, edge, pts, 1)
    hx, hy = rot(s*1.3, 0)
    pygame.draw.line(surf, edge, (sx,sy), (hx,hy), 1)

def draw_grid(surf, cx, cy, ppm, W, H):
    # Spacing in metres adapts to ppm so the grid never gets too dense.
    spacing = 1.0 if ppm <= 100 else (0.5 if ppm <= 250 else 0.1)
    rng_m = max(W, H) / ppm + spacing
    n = int(rng_m / spacing) + 1
    for i in range(-n, n+1):
        v = i * spacing
        sx, _ = w2s(v, 0, cx, cy, ppm)
        if 0 <= sx <= W: pygame.draw.line(surf, GRID, (sx,0), (sx,H), 1)
        _, sy = w2s(0, v, cx, cy, ppm)
        if 0 <= sy <= H: pygame.draw.line(surf, GRID, (0,sy), (W,sy), 1)

def draw_wind_arrow(surf, cx, cy, wx, wy, scale=8.0):
    mag = math.sqrt(wx**2 + wy**2)
    if mag < 0.05: return
    ax, ay = int(cx + wx*scale), int(cy - wy*scale)
    pygame.draw.line(surf, (100,200,255), (cx,cy), (ax,ay), 3)
    angle = math.atan2(-wy, wx)
    for da in [2.5, -2.5]:
        hx = ax - int(10*math.cos(angle+da))
        hy = ay + int(10*math.sin(angle+da))
        pygame.draw.line(surf, (100,200,255), (ax,ay), (hx,hy), 2)

# ======================================================================
# Extended Unicycle Environment (with wind / noise / sensor noise)
# ======================================================================

class ExtendedEnv(UnicycleEnv):
    """UnicycleEnv + wind, noise, sensor noise."""
    def __init__(self, dt=0.02):
        super().__init__(dt)
        self.wind_x = 0.0; self.wind_y = 0.0
        self.action_noise_std = 0.0; self.state_noise_std = 0.0
        self._mass_mult = 1.0; self._friction_mult = 1.0
        self._time = 0.0
        self._nominal_mass = self.MASS + self.INERTIA_Y / (self.WHEEL_RADIUS**2)
        self._nominal_Iz = self.INERTIA_Z
        self._nominal_friction = self.FRICTION
        self.last_actual_torque = np.zeros(2)

    @property
    def effective_mass(self):
        return self.MASS * self._mass_mult + self.INERTIA_Y / (self.WHEEL_RADIUS**2)
    @property
    def effective_Iz(self):
        return self.INERTIA_Z * self._mass_mult
    @property
    def effective_friction(self):
        return self.FRICTION * self._friction_mult

    def step(self, action):
        action = np.array(action, dtype=np.float64)
        if self.action_noise_std > 0:
            action = action + np.random.randn(2) * self.action_noise_std * np.array([3.0, 1.5])
        action[0] = np.clip(action[0], -20.0, 20.0)
        action[1] = np.clip(action[1], -10.0, 10.0)
        x, y, theta, v, omega = self.state
        m_eff, I_z, fric, dt = self.effective_mass, self.effective_Iz, self.effective_friction, self.dt

        gx = self.wind_x * (1.0 + 0.5*np.sin(2*np.pi*0.3*self._time)) * 1.5
        gy = self.wind_y * (1.0 + 0.5*np.cos(2*np.pi*0.2*self._time+0.7)) * 1.5
        wb_x =  np.cos(theta)*gx + np.sin(theta)*gy
        wb_y = (-np.sin(theta)*gx + np.cos(theta)*gy) * 0.5

        td = action[0] + wb_x
        ts = action[1] + wb_y
        td = np.clip(td, -25, 25); ts = np.clip(ts, -12, 12)

        self.last_actual_torque = np.array([td, ts])

        v_dot = (td - fric*v) / m_eff
        o_dot = (ts - fric*omega) / I_z

        self.state = np.array([x + v*np.cos(theta)*dt, y + v*np.sin(theta)*dt,
                               theta + omega*dt, v + v_dot*dt, omega + o_dot*dt])
        self._time += dt
        return self.state.copy()

    def get_state(self):
        s = self.state.copy()
        if self.state_noise_std > 0:
            s += np.random.randn(5) * self.state_noise_std * np.array([0.02, 0.02, 0.03, 0.05, 0.05])
        return s

    def get_observation(self):
        return self.get_state()

    def get_true_state(self):
        return self.state.copy()


class NominalEnvWrapper:
    """Returns nominal dynamics params for unicycle LQR (no privileged knowledge)."""
    def __init__(self, real_env, delay_steps=0):
        self._real = real_env
        self._m = real_env._nominal_mass
        self._Iz = real_env._nominal_Iz
        self._fric = real_env._nominal_friction
        self._delay = delay_steps
        self._state_history = []

    def __getattr__(self, name):
        if name in ('_real', '_m', '_Iz', '_fric', '_delay', '_state_history'):
            raise AttributeError(name)
        return getattr(self._real, name)

    @property
    def effective_mass(self): return self._m
    @property
    def effective_Iz(self): return self._Iz
    @property
    def effective_friction(self): return self._fric

    def get_delayed_state(self, current_state):
        self._state_history.append(current_state.copy())
        if len(self._state_history) > self._delay + 5:
            self._state_history = self._state_history[-(self._delay+5):]
        if len(self._state_history) > self._delay:
            return self._state_history[-self._delay - 1].copy()
        return current_state.copy()

    def reset_delay(self):
        self._state_history = []

# ======================================================================
# Platform abstraction: factories + per-platform helpers
# ======================================================================

def _attach_action_noise(env):
    """Monkey-patch env.step so it reads `env._act_noise_sigma` live each step."""
    orig_step = env.step
    env._act_noise_sigma = None
    def noisy_step(action):
        a = np.asarray(action, dtype=np.float64).copy()
        sig = getattr(env, "_act_noise_sigma", None)
        if sig is not None and np.any(np.asarray(sig) > 0):
            a = a + np.random.randn(*a.shape) * np.asarray(sig)
        return orig_step(a)
    env.step = noisy_step

def make_env(platform, dt):
    if platform == "unicycle":
        return ExtendedEnv(dt=dt)
    if platform == "auv":
        env = AUV3DEnv(dt=dt); _attach_action_noise(env); return env
    if platform == "drone":
        env = Drone3DEnv(dt=dt); _attach_action_noise(env); return env
    raise ValueError(platform)

def make_lqr(platform, env, dt):
    if platform == "unicycle":
        wrap = NominalEnvWrapper(env, delay_steps=0)
        return LQRController(wrap, dt), wrap
    return LQR3D(env, kind=PLATFORM_CFG[platform]["lqr_kind"]), None

def step_lqr(platform, ctrl, nominal_wrap, env, state, positions, k, n_steps):
    if platform == "unicycle":
        delayed = nominal_wrap.get_delayed_state(state)
        action, _ = ctrl.compute(delayed, positions, k)
    else:
        sp = positions[min(k+3, n_steps-1)]
        action = ctrl.compute(state, setpoint=sp)
    env.step(action)
    return action

def state_xy(state):
    """Top-down (x, y) — works for all platforms."""
    return state[:2]

def state_xyz(state, platform):
    """Return (x, y, z) for any platform; z is 0 on the unicycle."""
    if platform == "unicycle":
        return float(state[0]), float(state[1]), 0.0
    return float(state[0]), float(state[1]), float(state[2])

def state_heading(platform, state):
    if platform == "unicycle": return float(state[2])
    if platform == "auv":      return float(state[3])
    if platform == "drone":    return float(state[5])
    return 0.0

def make_init_state(platform, positions):
    p0 = positions[0]
    p2 = positions[min(2, len(positions)-1)]
    if platform == "unicycle":
        ddx = p2[0] - p0[0]; ddy = p2[1] - p0[1]
        return np.array([p0[0], p0[1], np.arctan2(ddy, ddx), 0.0, 0.0])
    psi0 = float(np.arctan2(p2[1] - p0[1], p2[0] - p0[0]))
    if platform == "auv":
        return np.array([p0[0], p0[1], p0[2], psi0,
                         0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([p0[0], p0[1], p0[2],
                     0.0, 0.0, psi0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0], dtype=np.float64)

def apply_perturbations(platform, env, sliders):
    m  = sliders["mass"].value
    f  = sliders["friction"].value
    wx = sliders["wind_x"].value
    wy = sliders["wind_y"].value
    sn = sliders["state_noise"].value
    an = sliders["noise"].value
    if platform == "unicycle":
        env._mass_mult = m; env._friction_mult = f
        env.wind_x = wx; env.wind_y = wy
        env.action_noise_std = an; env.state_noise_std = sn
        return
    if platform == "auv":
        env.set_perturbation(mass_mult=m, drag_mult=f)
        env.set_noise(std_pos=sn*0.05, std_vel=sn*0.05)
        env._act_noise_sigma = np.full(4, an*2.0) if an > 0 else None
        return
    if platform == "drone":
        env.set_perturbation(mass_mult=m, drag_mult=f,
                             wind_x=wx, wind_y=wy, wind_z=0.0)
        env.set_noise(std_pos=sn*0.02, std_vel=sn*0.03, std_ori=sn*0.02)
        env._act_noise_sigma = (an * np.array([0.3, 0.10, 0.10, 0.05])
                                if an > 0 else None)

# ======================================================================
# Model loading
# ======================================================================

def load_model(platform):
    path = PLATFORM_CFG[platform]["ckpt"]
    if not os.path.exists(path):
        print(f"  [!] No model at {path}. Run: python run.py train")
        return None
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model_state" in ckpt): return None
    use_ln = bool(ckpt.get("use_layernorm", False))
    model = EvidentialNet(ckpt["input_dim"], ckpt["output_dim"],
                          hidden_dims=ckpt.get("hidden_dims", [256,256,256,256]),
                          use_layernorm=use_ln)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.input_mean  = ckpt.get("input_mean",  torch.zeros(ckpt["input_dim"]))
    model.input_std   = ckpt.get("input_std",   torch.ones(ckpt["input_dim"]))
    model.target_mean = ckpt.get("target_mean", torch.zeros(ckpt["output_dim"]))
    model.target_std  = ckpt.get("target_std",  torch.ones(ckpt["output_dim"]))
    model.feature_mode = ckpt.get("feature_mode", "full")
    model.eval()
    print(f"  [OK] Loaded {path} (feature_mode={model.feature_mode})")
    return model

# ======================================================================
# Trajectory
# ======================================================================

def make_trajectory(platform, traj_key, total_time):
    cfg_p = PLATFORM_CFG[platform]
    dt = cfg_p["dt"]
    n_steps = int(total_time / dt)
    if platform == "unicycle":
        name = TRAJ_MAP.get(traj_key, 'lemniscate')
        if name == 'lemniscate':
            fn = lambda t: lemniscate_2d(t, scale=4.0, speed=0.5)
            positions, _ = generate_trajectory(fn, dt, n_steps, r_bar=4)
        else:
            positions, _ = generate_exciting_trajectory(name, dt, n_steps, 4,
                                                        scale=4.0, speed=0.5)
        return positions, n_steps, dt
    if platform == "auv":
        fn = lambda t: lemniscate_3d(t, scale_xy=1.2, scale_z=0.3,
                                     z_offset=-3.0, speed=0.08)
    else:  # drone
        fn = lambda t: lemniscate_3d(t, scale_xy=0.5, scale_z=0.2,
                                     z_offset=3.0, speed=0.12)
    positions, _ = generate_trajectory(fn, dt, n_steps, r_bar=4)
    return positions, n_steps, dt

# ======================================================================
# Per-controller state
# ======================================================================

class ControllerState:
    def __init__(self, idx, nn_model, dt, device, positions, platform):
        self.idx = idx; self.info = CONTROLLERS[idx]
        self.platform = platform
        self.env = make_env(platform, dt)
        self.env.reset(state=make_init_state(platform, positions))
        self.trail = []; self.errors = []; self.epist_log = []; self.ep_val = 0.0
        self.lqr = None; self.nn_ctrl = None; self.adapter = None
        self.nominal_wrap = None
        feature_mode = getattr(nn_model, "feature_mode", "full") if nn_model else "full"
        name = self.info["name"]
        feedback_env = (NominalEnvWrapper(self.env, delay_steps=0)
                        if platform == "unicycle" else self.env)
        if name == "LQR Expert":
            self.lqr, self.nominal_wrap = make_lqr(platform, self.env, dt)
        elif name == "NN Frozen":
            self.nn_ctrl = NNController(nn_model, device=device,
                                        feature_mode=feature_mode,
                                        feedback_gain=1.0,
                                        feedback_env=feedback_env,
                                        feedback_dt=dt) if nn_model else None
        elif name == "NN ER":
            self.adapter = UniformCLAdapter(
                nn_model, self.env, dt, device='cpu',
                buffer_capacity=1000, batch_size=32,
                lr=3e-4, lambda_reg=0.001,
                update_every=5, min_buffer_size=64,
                feature_mode=feature_mode,
                update_threshold=0.0,
                feedback_gain=1.0,
                feedback_env=feedback_env) if nn_model else None
        elif name == "NN ACE":
            # iter-37 paper config: tempered loss + relative_eps +
            # sustained-bias detector + anchor reset.  Same six core
            # hyperparameters as iter-29 plus four flags from
            # scripts/run_iter37_validation.py.
            self.adapter = ACEAdapter(
                nn_model, self.env, dt, device='cpu',
                buffer_capacity=1000, batch_size=32,
                lr=1e-4, lambda_eta=0.10, lambda_kappa=0.02,
                lambda_floor=0.10,
                anchor_strength=5e-3,
                update_every=5, min_buffer_size=64,
                feature_mode=feature_mode,
                feedback_gain=1.0, feedback_env=feedback_env,
                loss_form="tempered",
                relative_eps=True,
                shift_detection=True,
                shift_threshold=2.0,
                shift_anchor_reset=True) if nn_model else None
        self.epistemic_history = []; self.aleatoric_history = []
        self.last_aleatoric = 0.0
        self.lambda_history = []

    def apply_params(self, sliders):
        apply_perturbations(self.platform, self.env, sliders)
        if self.adapter:
            if hasattr(self.adapter, "lambda_reg"):
                self.adapter.lambda_reg = sliders["lambda_reg"].value
            if hasattr(self.adapter, "lambda_0"):
                self.adapter.lambda_0 = sliders["lambda_reg"].value

    def step_sim(self, positions, step, dt, n_steps):
        info = {}
        if step >= n_steps: return False
        if self.platform == "unicycle":
            state = self.env.get_state()
        else:
            state = self.env.get_observation()
        name = self.info["name"]
        if name == "LQR Expert" and self.lqr:
            step_lqr(self.platform, self.lqr, self.nominal_wrap,
                     self.env, state, positions, step, n_steps)
            self.ep_val = 0.0
        elif name == "NN Frozen" and self.nn_ctrl:
            action, info = self.nn_ctrl.compute(self.env, state, positions, step, dt)
            self.env.step(action); self.ep_val = info["epistemic"]
        elif name in ("NN ER", "NN ACE") and self.adapter:
            action, info = self.adapter.step(self.env, state, positions, step)
            self.ep_val = info["epistemic"]
        else:
            return False
        true_state = (self.env.get_true_state() if self.platform == "unicycle"
                      else self.env.get_state())
        xyz = state_xyz(true_state, self.platform)
        self.trail.append(xyz)
        if len(self.trail) > MAX_TRAIL_LEN:
            self.trail.pop(0)
        ref = positions[min(step, n_steps-1)]
        d = PLATFORM_CFG[self.platform]["pos_dim"]
        self.errors.append(float(np.linalg.norm(true_state[:d] - ref[:d])))
        self.epist_log.append(self.ep_val)
        al = info.get("aleatoric", 0.0)
        self.last_aleatoric = al
        self.aleatoric_history.append(al)
        if "lambda_schedule" in info:
            self.lambda_history.append(info["lambda_schedule"])
        else:
            self.lambda_history.append(1.0 if name == "NN ACE" else 0.0)
        return True

# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", default="lemniscate",
                        choices=["lemniscate", "multi_freq", "chirp", "composite"])
    parser.add_argument("--platform", default="unicycle",
                        choices=PLATFORMS,
                        help="Initial platform (LEFT/RIGHT to switch at runtime).")
    args = parser.parse_args()
    traj_key = {v:k for k,v in TRAJ_MAP.items()}.get(args.traj, 1)

    cfg = Config()
    cfg.nn.hidden_dims = [256,256,256,256]

    print("Loading models for all platforms ...")
    nn_models = {p: load_model(p) for p in PLATFORMS}

    current_platform = args.platform
    cx, cy = SIM_W//2, SIM_H//2
    device = cfg.device

    pygame.init()
    screen = pygame.display.set_mode((TOTAL_W, SIM_H))
    pygame.display.set_caption("ACE Multi-Platform Demo (LEFT/RIGHT to switch)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)
    font_big = pygame.font.SysFont("monospace", 18, bold=True)
    font_sec = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm = pygame.font.SysFont("monospace", 11)

    px, pw = SIM_W+15, PANEL_W-30; sy = 90; sliders = {}
    def add_s(key, label, vmin, vmax, default, fmt="{:.2f}", color=SLIDER_ACT, snap=True):
        nonlocal sy
        sliders[key] = Slider(px, sy, pw, label, vmin, vmax, default, fmt, color, snap); sy += 44
    add_s("mass",       "Mass",        0.5,4.0, 1.0, "{:.1f}x",  (255,100,100))
    add_s("friction",   "Friction",    0.1,5.0, 1.0, "{:.1f}x",  (255,160,60))
    sy += 8
    add_s("wind_x",     "Wind X",     -5.0,5.0, 0.0, "{:+.1f}N", (100,200,255))
    add_s("wind_y",     "Wind Y",     -5.0,5.0, 0.0, "{:+.1f}N", (100,200,255))
    add_s("noise",      "Act Noise",   0.0,3.0, 0.0, "{:.2f}",   (200,100,255))
    add_s("state_noise","Sensor Noise", 0.0,5.0, 0.0, "{:.1f}",  (255,100,200))
    sy += 8
    add_s("speed",      "Sim Speed",   1.0,20.0,1.0, "{:.0f}x",  (100,255,150))
    add_s("traj_speed", "Traj Speed",  0.2,1.5, 0.5, "{:.2f}",   (255,220,100))
    add_s("lambda_reg", "Lambda Reg",  0.001,0.1,0.01,"{:.3f}",  (255,200,50))

    positions = None; ref_pts = []; ctrl_states = []
    step = 0; paused = False; show_help = True; active_scenario = None
    n_steps = 0; dt = 0.02; ppm = 60

    def rebuild_traj():
        nonlocal positions, ref_pts, n_steps, dt, ppm
        cfg.traj.speed = sliders["traj_speed"].value
        ppm = PLATFORM_CFG[current_platform]["ppm"]
        positions, n_steps, dt = make_trajectory(
            current_platform, traj_key,
            PLATFORM_CFG[current_platform]["total_time"])
        z_scale = Z_VERTICAL_PPM.get(current_platform, 0.0)
        ref_pts = [w2s_3d(p[0], p[1],
                          (p[2] if (z_scale > 0 and len(p) > 2) else 0.0),
                          cx, cy, ppm, current_platform)
                   for p in positions[:n_steps]]

    def restart(new_traj=None, new_platform=None):
        nonlocal positions, ref_pts, ctrl_states, step, n_steps, traj_key, current_platform
        if new_platform is not None: current_platform = new_platform
        if new_traj is not None: traj_key = new_traj
        rebuild_traj()
        nn_model = nn_models.get(current_platform)
        ctrl_states = []
        for i in range(len(CONTROLLERS)):
            if i >= 1 and nn_model is None: continue
            ctrl_states.append(ControllerState(i, nn_model, dt, device,
                                               positions, current_platform))
        step = 0
        pygame.display.set_caption(
            f"ACE Demo — {PLATFORM_TITLES[current_platform]}  (LEFT/RIGHT to switch)")

    def apply_scenario(key):
        nonlocal active_scenario
        if key not in SCENARIOS: return
        active_scenario = key
        for sn, sv in SCENARIOS[key]["params"].items():
            if sn in sliders: sliders[sn].value = sv
        restart()

    def cycle_platform(direction):
        i = (PLATFORMS.index(current_platform) + direction) % len(PLATFORMS)
        restart(new_platform=PLATFORMS[i])

    restart(); running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False; continue
            slider_handled = False
            for s in sliders.values():
                if s.handle_event(ev): slider_handled = True
            if slider_handled: active_scenario = None; continue
            if ev.type == pygame.KEYDOWN:
                if show_help and ev.key not in (pygame.K_q, pygame.K_ESCAPE, pygame.K_h):
                    show_help = False; continue
                if ev.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif ev.key == pygame.K_SPACE: paused = not paused
                elif ev.key == pygame.K_r: restart()
                elif ev.key == pygame.K_h: show_help = not show_help
                elif ev.key == pygame.K_LEFT:  cycle_platform(-1)
                elif ev.key == pygame.K_RIGHT: cycle_platform(+1)
                elif ev.key == pygame.K_1 and current_platform == "unicycle": restart(1)
                elif ev.key == pygame.K_2 and current_platform == "unicycle": restart(2)
                elif ev.key == pygame.K_3 and current_platform == "unicycle": restart(3)
                elif ev.key == pygame.K_4 and current_platform == "unicycle": restart(4)
                elif ev.key == pygame.K_5: apply_scenario(5)
                elif ev.key == pygame.K_6: apply_scenario(6)
                elif ev.key == pygame.K_7: apply_scenario(7)
                elif ev.key == pygame.K_8: apply_scenario(8)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                for s in sliders.values():
                    kx, ky = s.knob_x, s.rect.y + s.rect.h//2
                    if abs(ev.pos[0]-kx)<15 and abs(ev.pos[1]-ky)<15:
                        s.reset(); active_scenario = None

        for cs in ctrl_states: cs.apply_params(sliders)
        speed_mult = max(1, int(sliders["speed"].value))

        if not paused and step < n_steps:
            for _ in range(speed_mult):
                if step >= n_steps: break
                for cs in ctrl_states: cs.step_sim(positions, step, dt, n_steps)
                step += 1

        # === DRAW ===
        screen.fill(BG)
        screen.set_clip(pygame.Rect(0,0,SIM_W,SIM_H))
        draw_grid(screen, cx, cy, ppm, SIM_W, SIM_H)

        # Reference trajectory rendered with alpha for less visual noise.
        if len(ref_pts) > 2:
            ref_overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
            ref_rgba = (REF[0], REF[1], REF[2], 120)
            for i in range(0, len(ref_pts)-1, 3):
                p1, p2 = ref_pts[i], ref_pts[min(i+1, len(ref_pts)-1)]
                if 0<=p1[0]<=SIM_W and 0<=p2[0]<=SIM_W:
                    pygame.draw.line(ref_overlay, ref_rgba, p1, p2, 2)
            screen.blit(ref_overlay, (0, 0))

        # Target waypoint
        if step < n_steps:
            ri = min(step+3, len(positions)-1)
            pz = (positions[ri][2] if len(positions[ri]) > 2 else 0.0)
            tp = w2s_3d(positions[ri][0], positions[ri][1], pz,
                        cx, cy, ppm, current_platform)
            if 0<=tp[0]<=SIM_W: pygame.draw.circle(screen, TARGET, tp, 6, 2)

        # Translucent overlay for trails + robot bodies + halos.
        overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        for cs in ctrl_states:
            c, sh = cs.info["color"], cs.info["shape"]
            pts_s = [w2s_3d(tx, ty, tz, cx, cy, ppm, cs.platform)
                     for tx, ty, tz in cs.trail]
            vis = [(px_,py_) for px_,py_ in pts_s if -50<px_<SIM_W+50 and -50<py_<SIM_H+50]

            if len(vis) > 1:
                use_epi = (cs.info["name"] != "LQR Expert"
                           and len(cs.epistemic_history) > 0)
                if use_epi:
                    n_vis = len(vis)
                    ep_start = max(0, len(cs.epistemic_history) - n_vis)
                    for i in range(min(n_vis - 1, len(cs.epistemic_history) - ep_start - 1)):
                        ep_seg = cs.epistemic_history[ep_start + i]
                        ratio = min(ep_seg / 0.3, 1.0)
                        seg_col = (int(255*ratio), int(255*(1-ratio)), 50, TRAIL_ALPHA)
                        pygame.draw.line(overlay, seg_col, vis[i], vis[i+1], 2)
                else:
                    rgba = (c[0], c[1], c[2], TRAIL_ALPHA)
                    pygame.draw.lines(overlay, rgba, False, vis, 2)

            true_state = (cs.env.get_true_state() if cs.platform == "unicycle"
                          else cs.env.get_state())
            wx_, wy_, wz_ = state_xyz(true_state, cs.platform)
            heading = state_heading(cs.platform, true_state)
            draw_robot(overlay, wx_, wy_, wz_, heading, cx, cy, ppm,
                       c, sh, platform=cs.platform)

            if cs.info["name"] != "LQR Expert":
                sx, sy_h = w2s_3d(wx_, wy_, wz_, cx, cy, ppm, cs.platform)
                ep_now = cs.ep_val
                ratio = min(ep_now / 0.3, 1.0)
                halo = (int(255*ratio), int(255*(1-ratio)), 50, HALO_ALPHA)
                halo_r = int(8 + 14 * ratio)
                pygame.draw.circle(overlay, halo, (sx, sy_h), halo_r, 1)
        screen.blit(overlay, (0, 0))

        # Wind arrow (no exogenous wind on AUV — drag perturbation only)
        wx, wy = sliders["wind_x"].value, sliders["wind_y"].value
        if (abs(wx)>0.05 or abs(wy)>0.05) and current_platform != "auv":
            wcx, wcy = 80, SIM_H-80
            pygame.draw.circle(screen, (40,40,55), (wcx,wcy), 30, 1)
            draw_wind_arrow(screen, wcx, wcy, wx, wy, scale=5)
            screen.blit(font.render("WIND", True, (100,200,255)), (wcx-16, wcy+34))

        # Platform banner
        plat_label = PLATFORM_TITLES[current_platform]
        screen.blit(font_big.render(plat_label, True, (220, 220, 240)), (10, 6))
        screen.blit(font_sm.render("LEFT / RIGHT  switch platform",
                                   True, (140, 140, 160)), (10, 26))

        screen.set_clip(None)

        # === PANEL ===
        pygame.draw.rect(screen, PANEL_BG, (SIM_W,0,PANEL_W,SIM_H))
        pygame.draw.line(screen, (60,60,75), (SIM_W,0), (SIM_W,SIM_H), 2)

        py = 10
        if active_scenario and active_scenario in SCENARIOS:
            sc = SCENARIOS[active_scenario]
            screen.blit(font_big.render(f"S{active_scenario}: {sc['name']}", True, SECTION_COL), (px,py))
            py += 20
            desc = sc["desc"]
            while desc:
                chunk = desc[:36]
                if len(desc) > 36:
                    sp = chunk.rfind(' ')
                    if sp > 10: chunk = desc[:sp]; desc = desc[sp+1:]
                    else: desc = desc[36:]
                else: desc = ""
                screen.blit(font_sm.render(chunk, True, TEXT_DIM), (px,py)); py += 14
        else:
            screen.blit(font_big.render("Custom", True, SECTION_COL), (px,py)); py += 18

        for s in sliders.values(): s.draw(screen, font)

        sy2 = sliders["lambda_reg"].rect.y + 48
        pygame.draw.line(screen, (60,60,75), (SIM_W+10,sy2), (SIM_W+PANEL_W-10,sy2), 1); sy2 += 6

        t_s = step * dt
        screen.blit(font_sec.render(f"Time: {t_s:.1f}s / {n_steps*dt:.0f}s", True, TEXT), (px,sy2)); sy2 += 18
        if paused:
            screen.blit(font_big.render("PAUSED", True, (255,255,100)), (px,sy2)); sy2 += 22

        prog = step / max(n_steps,1)
        pygame.draw.rect(screen, SLIDER_BG, (px,sy2,pw,6), border_radius=3)
        pygame.draw.rect(screen, SECTION_COL, (px,sy2,int(pw*prog),6), border_radius=3); sy2 += 14

        screen.blit(font_sec.render("-- Tracking Error --", True, (80,80,100)), (px,sy2)); sy2 += 18
        for cs in ctrl_states:
            c, nm = cs.info["color"], cs.info["short"]
            mean_err = np.mean(cs.errors[-500:]) if cs.errors else 0.0
            mx, my = px, sy2+1; sh = cs.info["shape"]
            if sh == "triangle":   pygame.draw.polygon(screen, c, [(mx+5,my),(mx,my+8),(mx+10,my+8)])
            elif sh == "diamond":  pygame.draw.polygon(screen, c, [(mx+5,my),(mx+10,my+5),(mx+5,my+10),(mx,my+5)])
            elif sh == "square":   pygame.draw.rect(screen, c, (mx,my,10,10))
            elif sh == "circle":   pygame.draw.circle(screen, c, (mx+5,my+5), 5)
            elif sh == "pentagon": pygame.draw.polygon(screen, c,
                                       [(mx+5,my),(mx+10,my+4),(mx+8,my+10),(mx+2,my+10),(mx,my+4)])
            ec = WARN_COL if mean_err > 0.3 else (GOOD_COL if mean_err < 0.1 else TEXT)
            screen.blit(font.render(f"{nm:6s} {mean_err:.3f}m", True, ec), (mx+14,sy2))
            bx = px+120; bw = int(min(mean_err/1.0,1.0)*(pw-120))
            pygame.draw.rect(screen, SLIDER_BG, (bx,sy2+2,pw-120,8), border_radius=2)
            if bw > 0: pygame.draw.rect(screen, c, (bx,sy2+2,min(bw,pw-120),8), border_radius=2)
            sy2 += 16

        sy2 += 4
        screen.blit(font_sec.render("-- Epistemic Unc --", True, (80,80,100)), (px,sy2)); sy2 += 18
        for cs in ctrl_states:
            if cs.info["name"] == "LQR Expert": continue
            c, nm, ep = cs.info["color"], cs.info["short"], cs.ep_val
            ec = WARN_COL if ep > 0.1 else GOOD_COL
            screen.blit(font.render(f"{nm:6s} {ep:.4f}", True, ec), (px+14,sy2))
            bx = px+120; bw = int(min(ep/0.3,1.0)*(pw-120))
            pygame.draw.rect(screen, SLIDER_BG, (bx,sy2+2,pw-120,8), border_radius=2)
            if bw > 0: pygame.draw.rect(screen, c, (bx,sy2+2,min(bw,pw-120),8), border_radius=2)
            sy2 += 16

        for cs in ctrl_states:
            if cs.adapter is None: continue
            nm, c = cs.info["short"], cs.info["color"]
            bl, bc, up = len(cs.adapter.buffer), cs.adapter.buffer.capacity, cs.adapter.update_count
            screen.blit(font_sm.render(f"{nm}: buf={bl}/{bc} upd={up}", True, TEXT_DIM), (px,sy2)); sy2 += 12

        sy2 += 4
        for cs in ctrl_states:
            if cs.info["name"] != "NN ACE": continue
            if len(cs.lambda_history) < 2: continue
            c = cs.info["color"]
            screen.blit(font_sec.render("-- ACE λ_t vs Time --", True, (80,80,100)), (px,sy2)); sy2 += 16
            plot_w, plot_h = pw, 45
            plot_x, plot_y = px, sy2
            pygame.draw.rect(screen, SLIDER_BG, (plot_x, plot_y, plot_w, plot_h), border_radius=3)
            for y_ref in [0.25, 0.5, 0.75]:
                y_line = plot_y + plot_h - int(y_ref * (plot_h - 4)) - 2
                pygame.draw.line(screen, (60,60,70), (plot_x, y_line), (plot_x+plot_w, y_line), 1)
            hist = cs.lambda_history[-400:] if len(cs.lambda_history) > 400 else cs.lambda_history
            n_h = len(hist)
            if n_h > 1:
                pts = []
                for i, lam in enumerate(hist):
                    x = plot_x + int((i / max(n_h-1, 1)) * plot_w)
                    y = plot_y + plot_h - int(lam * (plot_h - 4)) - 2
                    pts.append((x, y))
                if len(pts) > 1:
                    pygame.draw.lines(screen, c, False, pts, 2)
            lam_now = cs.lambda_history[-1]
            screen.blit(font_sm.render(f"λ={lam_now:.3f}", True, c), (plot_x + plot_w - 60, plot_y - 12))
            sy2 += plot_h + 4
            break

        sy2 += 2
        screen.blit(font_sm.render("Trail: green=confident  red=uncertain", True, (150,150,160)), (px,sy2)); sy2 += 12

        sy2 += 6; perturbs = []
        if abs(sliders["mass"].value-1.0)>0.05: perturbs.append(f"M={sliders['mass'].value:.1f}x")
        if abs(sliders["friction"].value-1.0)>0.05: perturbs.append(f"F={sliders['friction'].value:.1f}x")
        if abs(wx)>0.1 or abs(wy)>0.1: perturbs.append(f"W=({wx:+.0f},{wy:+.0f})")
        if sliders["noise"].value>0.01: perturbs.append(f"An={sliders['noise'].value:.1f}")
        if sliders["state_noise"].value>0.01: perturbs.append(f"Sn={sliders['state_noise'].value:.1f}")
        if perturbs:
            screen.blit(font_sm.render("Active: "+" ".join(perturbs), True, WARN_COL), (px,sy2))

        bar_bg = pygame.Surface((SIM_W,24)); bar_bg.set_alpha(200); bar_bg.fill((15,15,25))
        screen.blit(bar_bg, (0,SIM_H-24))
        screen.blit(font.render(
            "LEFT/RIGHT=platform  5-8=scenarios  1-4=traj  SPACE=pause  R=restart  H=help  Q=quit",
            True, (90,90,110)), (10, SIM_H-20))
        scx = SIM_W - 380
        for sk in [5,6,7,8]:
            tc = SECTION_COL if sk == active_scenario else (70,70,80)
            screen.blit(font_sm.render(f"{sk}:{SCENARIOS[sk]['name'][:9]}", True, tc), (scx,SIM_H-19))
            scx += 95

        if show_help:
            hl = [
                "=== ACE MULTI-PLATFORM DEMO ===", "",
                "  LEFT / RIGHT  Switch platform: Unicycle <-> AUV <-> Drone",
                "  5  Nominal      All defaults, all track well",
                "  6  Wind         Wind disturbance -> NN methods adapt",
                "  7  Heavy+Fast   Mass 3.5x + fast -> Frozen bad, ACE wins",
                "  8  Noisy shift  Mass 2.5x + sensor noise -> ACE rejects noise",
                "",
                "  1-4       Trajectory select (unicycle only)",
                "  SPACE     Pause / Resume",
                "  R         Restart",
                "  H         Toggle help",
                "  Q / ESC   Quit",
                "",
                "=== CONTROLLERS ===", "",
                "  Green tri   LQR  (nominal gains, can't adapt)",
                "  Blue dia    Frozen (offline only)",
                "  Purple sq   ER   (uniform replay baseline)",
                "  Pink pent   ACE  (smooth adaptive lambda + epistemic priority, OURS)",
                "",
                "=== PLATFORMS ===", "",
                "  Unicycle  2-D ground robot, 5 states / 2 actions",
                "  AUV       BlueROV2, 8 states / 4 actions (top-down view)",
                "  Drone     12-state quadrotor, T + 3 torques (top-down view)",
                "",
                "  Drag sliders or right-click to reset.",
                "  Press any key to close.",
            ]
            ow, oh = 580, len(hl)*16+30
            ox, oy = (SIM_W-ow)//2, (SIM_H-oh)//2
            ov = pygame.Surface((ow,oh)); ov.set_alpha(240); ov.fill((12,12,22))
            screen.blit(ov, (ox,oy))
            pygame.draw.rect(screen, SECTION_COL, (ox,oy,ow,oh), 2)
            for i, ln in enumerate(hl):
                if ln.startswith("==="): c = SECTION_COL
                elif "Green" in ln: c = CONTROLLERS[0]["color"]
                elif "Blue" in ln:  c = CONTROLLERS[1]["color"]
                elif "Purple" in ln: c = CONTROLLERS[2]["color"]
                elif "Pink" in ln or "ACE" in ln: c = CONTROLLERS[3]["color"]
                elif "->" in ln: c = (180,180,200)
                else: c = (210,210,210)
                screen.blit(font.render(ln, True, c), (ox+15, oy+12+i*16))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
