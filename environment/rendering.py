"""
Rendering for SafeBuild Inspector.

Two renderers:
  * OpenGLRenderer   - real-time 3D view (pygame window + PyOpenGL). Terrain is
                       drawn as extruded 3D columns, buildings as 3D blocks, the
                       inspection drone as a hovering pyramid with a scan beam.
  * SoftwareRenderer - 2.5D isometric image drawn with Pillow. Needs no display,
                       so it works in Jupyter/Colab and for recording GIF/MP4.

make_renderer() picks OpenGL for "human" mode and falls back automatically.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment.custom_env import ACTION_NAMES

STATUS_COLOURS = {
    "unknown": (235, 235, 235),
    "inspected": (250, 210, 60),
    "flagged": (220, 45, 45),
    "cleared": (40, 170, 80),
}
RIVER = (40, 110, 215)
DRONE = (140, 60, 220)


def _terrain_colour(env, y, x):
    if env.river[y, x]:
        return RIVER
    e = float(env.elevation[y, x])
    base = np.array([110 + 70 * e, 160 - 30 * e, 80 - 20 * e])          # green -> brown with height
    if env.flood_risk[y, x] > 0.5:                                        # flood plain: blue tint
        base = 0.55 * base + 0.45 * np.array([120, 170, 230])
    if env.slope[y, x] > 0.6:                                             # steep slope: orange tint
        base = 0.55 * base + 0.45 * np.array([215, 120, 50])
    return tuple(int(c) for c in np.clip(base, 0, 255))


def _building_colour(b):
    if b["status"] == "unknown":
        return STATUS_COLOURS["inspected"] if b["inspected"] else STATUS_COLOURS["unknown"]
    return STATUS_COLOURS[b["status"]]


def _hud_lines(env):
    info = env._info()
    act = ACTION_NAMES.get(env.last_action, "-") if env.last_action is not None else "-"
    return [
        "SafeBuild Inspector  |  flood & landslide risk survey",
        f"step {env.steps}/{env.max_steps}   battery {info['battery']*100:5.1f}%   "
        f"undecided {info['remaining']}/{env.n_buildings}",
        f"flags ok {info['correct_flags']}  false alarms {info['false_alarms']}  "
        f"clears ok {info['correct_clears']}  missed {info['missed_unsafe']}",
        f"action: {act}   event: {env.last_event}",
    ]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
class SoftwareRenderer:
    """Isometric 2.5D renderer that returns an RGB numpy array."""

    def __init__(self, env, width=900, height=700):
        self.w, self.h = width, height
        g = env.grid_size
        self.tw = int(min(width / (g + 1), 2 * (height - 220) / (g + 1)))
        self.th = self.tw // 2
        self.hs = self.th * 1.2          # elevation scale (pixels)
        self.ox = width // 2
        self.oy = 150
        self.font = _font(15)
        self.small = _font(13)

    def _iso(self, y, x, z=0.0):
        sx = self.ox + (x - y) * self.tw / 2
        sy = self.oy + (x + y) * self.th / 2 - z
        return sx, sy

    def _block(self, d, y, x, z0, z1, top, inset=0.0):
        """Draw an iso box covering cell (y,x) from height z0 to z1."""
        a, b = inset, 1 - inset
        p = lambda yy, xx, z: self._iso(y + yy, x + xx, z)
        top_poly = [p(a, a, z1), p(a, b, z1), p(b, b, z1), p(b, a, z1)]
        left = [p(b, a, z1), p(b, b, z1), p(b, b, z0), p(b, a, z0)]
        right = [p(a, b, z1), p(b, b, z1), p(b, b, z0), p(a, b, z0)]
        dark = lambda c, f: tuple(int(v * f) for v in c)
        d.polygon(left, fill=dark(top, 0.70), outline=dark(top, 0.5))
        d.polygon(right, fill=dark(top, 0.85), outline=dark(top, 0.5))
        d.polygon(top_poly, fill=top, outline=dark(top, 0.6))

    def draw(self, env):
        img = Image.new("RGB", (self.w, self.h), (18, 24, 38))
        d = ImageDraw.Draw(img)
        g = env.grid_size
        cells = sorted(((y, x) for y in range(g) for x in range(g)), key=lambda c: c[0] + c[1])
        for y, x in cells:
            z = float(env.elevation[y, x]) * self.hs
            self._block(d, y, x, -8, z, _terrain_colour(env, y, x))
            b = env.buildings.get((y, x))
            if b:
                self._block(d, y, x, z, z + self.th * 1.1, _building_colour(b), inset=0.22)
                if b["status"] == "flagged":
                    cx, cy = self._iso(y + 0.5, x + 0.5, z + self.th * 1.1)
                    d.text((cx - 4, cy - 18), "!", fill=(255, 255, 255), font=self.font)

        # drone
        ay, ax = env.agent
        gz = float(env.elevation[ay, ax]) * self.hs
        cx, cy = self._iso(ay + 0.5, ax + 0.5, gz)
        hover = cy - self.th * 2.6
        d.ellipse([cx - 10, cy - 4, cx + 10, cy + 4], fill=(10, 10, 10))           # shadow
        if env.last_action == 4:                                                     # scan beam
            d.polygon([(cx, hover), (cx - 16, cy), (cx + 16, cy)], fill=(120, 230, 255))
        r = self.th * 0.55
        d.polygon([(cx, hover - r), (cx + r, hover), (cx, hover + r * 0.6), (cx - r, hover)],
                  fill=DRONE, outline=(255, 255, 255))
        for sx in (-1, 1):
            d.line([(cx + sx * r, hover), (cx + sx * (r + 8), hover - 6)], fill=(255, 255, 255), width=2)

        # HUD and legend
        d.rectangle([0, 0, self.w, 92], fill=(10, 14, 24))
        for i, line in enumerate(_hud_lines(env)):
            d.text((12, 8 + i * 20), line, fill=(230, 235, 245), font=self.font if i == 0 else self.small)
        legend = [("undecided", STATUS_COLOURS["unknown"]), ("inspected", STATUS_COLOURS["inspected"]),
                  ("flagged", STATUS_COLOURS["flagged"]), ("cleared", STATUS_COLOURS["cleared"]),
                  ("river", RIVER), ("drone", DRONE)]
        lx = 12
        for name, col in legend:
            d.rectangle([lx, self.h - 26, lx + 14, self.h - 12], fill=col)
            d.text((lx + 20, self.h - 27), name, fill=(220, 220, 220), font=self.small)
            lx += 110
        d.text((12, self.h - 50), "terrain: blue tint = flood plain, orange tint = steep (landslide) slope",
               fill=(180, 190, 205), font=self.small)
        return np.asarray(img)

    def close(self):
        pass


# ---------------------------------------------------------------------------
def _look_at(GL, eye, target, up):
    """Replacement for gluLookAt (GLU is missing on some systems)."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    GL.glMultMatrixd(m.T.flatten())
    GL.glTranslated(-eye[0], -eye[1], -eye[2])


class OpenGLRenderer:
    """Real-time 3D renderer using pygame + PyOpenGL (fixed-function pipeline)."""

    def __init__(self, env, width=1000, height=720, show_window=True):
        import os
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        from OpenGL import GL
        self.pygame, self.GL = pygame, GL
        self.w, self.h = width, height
        self.show = show_window
        pygame.init()
        pygame.display.set_caption("SafeBuild Inspector - 3D (OpenGL)")
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        if not show_window:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((width, height), flags)
        self.font = pygame.font.SysFont("dejavusans,arial", 16)
        self.clock = pygame.time.Clock()
        self.angle = 35.0
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.4, 1.0, 0.6, 0.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.35, 0.35, 0.4, 1.0))
        GL.glClearColor(0.07, 0.09, 0.15, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _box(self, x0, z0, x1, z1, y0, y1, colour, alpha=1.0):
        GL = self.GL
        c = [v / 255 for v in colour]
        GL.glColor4f(c[0], c[1], c[2], alpha)
        faces = [
            ((0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
            ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
            ((0, 0, -1), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
            ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
            ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
        ]
        GL.glBegin(GL.GL_QUADS)
        for n, verts in faces:
            GL.glNormal3f(*n)
            for v in verts:
                GL.glVertex3f(*v)
        GL.glEnd()

    def _hud(self, env):
        GL, pg = self.GL, self.pygame
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        for i, line in enumerate(_hud_lines(env)):
            surf = self.font.render(line, True, (235, 240, 250), (10, 14, 24))
            data = pg.image.tostring(surf, "RGBA", True)
            GL.glWindowPos2d(10, self.h - 24 - i * 22)
            GL.glDrawPixels(surf.get_width(), surf.get_height(), GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

    def draw(self, env):
        GL, pg = self.GL, self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit("Window closed by user")

        g = env.grid_size
        GL.glViewport(0, 0, self.w, self.h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        near, far = 0.1, 100.0
        top = near * math.tan(math.radians(45) / 2)
        GL.glFrustum(-top * self.w / self.h, top * self.w / self.h, -top, top, near, far)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        self.angle += 0.15  # slow orbit so the 3D structure is visible
        r = g * 1.25
        cx = cz = g / 2
        ex = cx + r * math.cos(math.radians(self.angle))
        ez = cz + r * math.sin(math.radians(self.angle))
        _look_at(GL, (ex, g * 0.95, ez), (cx, 0.0, cz), (0.0, 1.0, 0.0))

        H = 1.6  # elevation scale in world units
        for y in range(g):
            for x in range(g):
                top = float(env.elevation[y, x]) * H
                self._box(x, y, x + 0.98, y + 0.98, -0.2, top, _terrain_colour(env, y, x))
                b = env.buildings.get((y, x))
                if b:
                    self._box(x + 0.22, y + 0.22, x + 0.78, y + 0.78, top, top + 0.7, _building_colour(b))

        ay, ax = env.agent
        ground = float(env.elevation[ay, ax]) * H
        hover = ground + 2.0 + 0.1 * math.sin(self.angle)
        dx, dz = ax + 0.5, ay + 0.5
        c = [v / 255 for v in DRONE]
        GL.glColor3f(*c)
        GL.glBegin(GL.GL_TRIANGLES)  # drone as a 4-sided pyramid pointing down
        pts = [(dx - .35, hover, dz - .35), (dx + .35, hover, dz - .35),
               (dx + .35, hover, dz + .35), (dx - .35, hover, dz + .35)]
        tip = (dx, hover - .45, dz)
        for i in range(4):
            a, b2 = pts[i], pts[(i + 1) % 4]
            GL.glNormal3f(0, -1, 0)
            GL.glVertex3f(*a); GL.glVertex3f(*b2); GL.glVertex3f(*tip)
        GL.glEnd()
        if env.last_action == 4:  # translucent scan beam
            self._box(dx - .3, dz - .3, dx + .3, dz + .3, ground, hover - .45, (120, 230, 255), alpha=0.35)

        self._hud(env)
        frame = None
        if not self.show or env.render_mode == "rgb_array":
            buf = GL.glReadPixels(0, 0, self.w, self.h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            frame = np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)).copy()
        pg.display.flip()
        self.clock.tick(env.metadata["render_fps"])
        return frame

    def close(self):
        try:
            self.pygame.display.quit()
            self.pygame.quit()
        except Exception:
            pass


class PygameWindow:
    """Fallback 'human' mode: shows the software renderer in a plain pygame window."""

    def __init__(self, env):
        import os
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        self.pygame = pygame
        self.soft = SoftwareRenderer(env)
        pygame.init()
        self.screen = pygame.display.set_mode((self.soft.w, self.soft.h))
        pygame.display.set_caption("SafeBuild Inspector - 2.5D")
        self.clock = pygame.time.Clock()

    def draw(self, env):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit("Window closed by user")
        frame = self.soft.draw(env)
        surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        pg.display.flip()
        self.clock.tick(env.metadata["render_fps"])
        return None

    def close(self):
        try:
            self.pygame.display.quit()
        except Exception:
            pass


def make_renderer(env, render_mode, choice="auto"):
    """choice: 'auto' | 'opengl' | 'software'."""
    if render_mode == "rgb_array":
        if choice == "opengl":
            return OpenGLRenderer(env, show_window=False)
        return SoftwareRenderer(env)
    # human mode
    if choice in ("auto", "opengl"):
        try:
            return OpenGLRenderer(env, show_window=True)
        except Exception as exc:  # no OpenGL driver / no display
            print(f"[rendering] OpenGL unavailable ({exc}); using 2.5D pygame window instead.")
    return PygameWindow(env)
