from __future__ import annotations

from math import pi

try:
    import arcade
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "Arcade is required for graphical mode. Install with: pip install arcade==3.3.3"
    ) from exc

from .arena_api import ObjectKind
from .environment import TrainingEnvironment
from .geometry import heading_vector


SCREEN_W = 1240
SCREEN_H = 860
PANEL_W = 350
MARGIN = 22


class ArenaWindow(arcade.Window):
    """Arcade viewer for the training city.

    The renderer deliberately consumes debug data only. Hidden engine state is
    never fed back to an agent. Text uses cached ``arcade.Text`` instances so
    no ``draw_text`` performance warning is emitted.
    """

    def __init__(self, env: TrainingEnvironment, title: str = "RL Arena — Training City"):
        super().__init__(SCREEN_W, SCREEN_H, title, resizable=True, update_rate=1 / 60, vsync=True)
        self.env = env
        self.paused = False
        self.single_step = False
        self.speed_multiplier = 1.0
        self.accumulator = 0.0
        self.show_vision = True
        self.show_hearing = True
        self.show_touch = True
        self.teacher_debug = False
        self.selected_agent: str | None = next(iter(env.agent_ids), None)
        self.background_color = (17, 21, 27)

        self._texts: dict[str, arcade.Text] = {}
        self._agent_textures = self._load_agent_textures()
        self.crate_texture = arcade.load_texture(":resources:images/tiles/boxCrate_double.png")

    # ------------------------------------------------------------------
    # Cached text: replaces arcade.draw_text everywhere in this renderer.
    # ------------------------------------------------------------------
    def _text(
        self,
        key: str,
        value: object,
        x: float,
        y: float,
        color=(240, 240, 240),
        font_size: float = 10,
        *,
        bold: bool = False,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> arcade.Text:
        value = str(value)
        text = self._texts.get(key)
        if text is None:
            text = arcade.Text(
                value,
                x,
                y,
                color,
                font_size,
                bold=bold,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )
            self._texts[key] = text
        else:
            text.value = value
            text.x = x
            text.y = y
            text.color = color
        text.draw()
        return text

    @staticmethod
    def _load_agent_textures():
        candidates = [
            ":resources:images/animated_characters/female_adventurer/femaleAdventurer_idle.png",
            ":resources:images/animated_characters/male_adventurer/maleAdventurer_idle.png",
            ":resources:images/animated_characters/female_person/femalePerson_idle.png",
            ":resources:images/animated_characters/male_person/malePerson_idle.png",
            ":resources:images/animated_characters/robot/robot_idle.png",
            ":resources:images/animated_characters/zombie/zombie_idle.png",
        ]
        textures = []
        for path in candidates:
            try:
                textures.append(arcade.load_texture(path))
            except Exception:
                pass
        if not textures:
            textures.append(
                arcade.load_texture(
                    ":resources:images/animated_characters/female_adventurer/femaleAdventurer_idle.png"
                )
            )
        return textures

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------
    def on_update(self, delta_time: float):
        if self.paused and not self.single_step:
            return
        self.accumulator += delta_time * self.speed_multiplier
        real_seconds_per_tick = 0.5
        guard = 0
        while (self.accumulator >= real_seconds_per_tick or self.single_step) and guard < 20:
            guard += 1
            try:
                result = self.env.step({})
            except RuntimeError:
                self.paused = True
                break
            self.accumulator = max(0.0, self.accumulator - real_seconds_per_tick)
            self.single_step = False
            if result.truncated:
                self.paused = True
                break

    def on_draw(self):
        self.clear()
        debug = self.env.debug_state()
        if not debug:
            return
        left, bottom, world_px, scale = self._world_layout()
        self._draw_world(debug, left, bottom, scale)
        self._draw_panel(
            debug,
            left + world_px + 20,
            20,
            self.width - (left + world_px + 40),
            self.height - 40,
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _world_layout(self):
        available_w = max(320, self.width - PANEL_W - 3 * MARGIN)
        available_h = max(320, self.height - 2 * MARGIN)
        world_px = min(available_w, available_h)
        left = MARGIN
        bottom = (self.height - world_px) / 2
        scale = world_px / self.env.config.map_width
        return left, bottom, world_px, scale

    @staticmethod
    def _xy(x: float, y: float, left: float, bottom: float, scale: float):
        return left + x * scale, bottom + y * scale

    @staticmethod
    def _rect_xywh(rect, left: float, bottom: float, scale: float):
        return (
            left + rect.left * scale,
            bottom + rect.bottom * scale,
            rect.w * scale,
            rect.h * scale,
        )

    # ------------------------------------------------------------------
    # World rendering
    # ------------------------------------------------------------------
    def _draw_world(self, debug, left: float, bottom: float, scale: float):
        world = debug["world"]
        W = self.env.config.map_width * scale
        H = self.env.config.map_height * scale

        # Base: a tiled-looking ground instead of a flat color.
        arcade.draw_lbwh_rectangle_filled(left, bottom, W, H, (82, 112, 72))
        self._draw_ground_detail(left, bottom, scale)
        self._draw_districts(world, left, bottom, scale)
        self._draw_roads(left, bottom, scale)
        self._draw_buildings(world, left, bottom, scale)
        self._draw_areas(world, left, bottom, scale)
        self._draw_decorations(world, left, bottom, scale)
        self._draw_rect_obstacles(world, left, bottom, scale)
        self._draw_doors(world, left, bottom, scale)
        self._draw_world_objects(world, left, bottom, scale)
        self._draw_projectiles(debug, left, bottom, scale)
        self._draw_agents(debug, left, bottom, scale)
        self._draw_sensor_overlays(debug, left, bottom, scale)

        # Slight atmospheric veil: indicates degraded conditions without making
        # the map unreadable to the human observer.
        visibility = world.visibility_factor * world.low_light_factor
        if visibility < 0.92:
            alpha = int(min(38, (1.0 - visibility) * 95))
            arcade.draw_lbwh_rectangle_filled(left, bottom, W, H, (184, 194, 199, alpha))

        arcade.draw_lbwh_rectangle_outline(left, bottom, W, H, (224, 229, 232), 2)

    def _draw_ground_detail(self, left: float, bottom: float, scale: float):
        # Deterministic checker/noise pattern. Purely visual and cheap.
        step = 1.5
        W = self.env.config.map_width
        H = self.env.config.map_height
        iy = 0
        y = 0.0
        while y < H:
            ix = 0
            x = 0.0
            while x < W:
                if (ix * 3 + iy * 5) % 7 == 0:
                    sx, sy = self._xy(x + 0.35, y + 0.30, left, bottom, scale)
                    arcade.draw_circle_filled(sx, sy, max(1.2, 0.06 * scale), (101, 130, 82, 120))
                elif (ix + iy) % 5 == 0:
                    sx, sy = self._xy(x + 0.7, y + 0.8, left, bottom, scale)
                    arcade.draw_circle_filled(sx, sy, max(1.0, 0.045 * scale), (62, 97, 62, 100))
                ix += 1
                x += step
            iy += 1
            y += step

    def _draw_districts(self, world, left: float, bottom: float, scale: float):
        colors = {
            "shops": (196, 171, 129, 38),
            "park": (82, 139, 76, 46),
            "medical": (146, 173, 184, 35),
            "industrial": (116, 118, 125, 48),
        }
        for zone in world.zones:
            x, y, w, h = self._rect_xywh(zone.rect, left, bottom, scale)
            arcade.draw_lbwh_rectangle_filled(x, y, w, h, colors.get(zone.theme, (100, 100, 100, 30)))
            # District names are deliberately subtle; buildings are the main landmarks.
            self._text(
                f"zone:{zone.name}",
                zone.name.upper(),
                x + 8,
                y + h - 16,
                (245, 245, 242, 135),
                8,
                bold=True,
            )

    def _draw_roads(self, left: float, bottom: float, scale: float):
        # The main cross is the visual backbone of the town.
        road = (57, 61, 67)
        sidewalk = (178, 174, 165)
        curb = (213, 207, 194)
        main = 3.2
        walk = 4.15

        # Sidewalks under roads.
        x = left + (15.0 - walk / 2) * scale
        arcade.draw_lbwh_rectangle_filled(x, bottom, walk * scale, 30.0 * scale, sidewalk)
        y = bottom + (15.0 - walk / 2) * scale
        arcade.draw_lbwh_rectangle_filled(left, y, 30.0 * scale, walk * scale, sidewalk)

        # Curbs.
        for offset in (-walk / 2, walk / 2):
            sx = left + (15.0 + offset) * scale
            arcade.draw_line(sx, bottom, sx, bottom + 30.0 * scale, curb, max(1, int(scale * 0.07)))
            sy = bottom + (15.0 + offset) * scale
            arcade.draw_line(left, sy, left + 30.0 * scale, sy, curb, max(1, int(scale * 0.07)))

        # Asphalt.
        x = left + (15.0 - main / 2) * scale
        arcade.draw_lbwh_rectangle_filled(x, bottom, main * scale, 30.0 * scale, road)
        y = bottom + (15.0 - main / 2) * scale
        arcade.draw_lbwh_rectangle_filled(left, y, 30.0 * scale, main * scale, road)

        # Dashed lane markings.
        dash = 1.1
        gap = 0.8
        pos = 0.5
        while pos < 30:
            y0 = bottom + pos * scale
            arcade.draw_line(left + 15 * scale, y0, left + 15 * scale, y0 + dash * scale, (226, 203, 91), max(1, int(scale * 0.055)))
            x0 = left + pos * scale
            arcade.draw_line(x0, bottom + 15 * scale, x0 + dash * scale, bottom + 15 * scale, (226, 203, 91), max(1, int(scale * 0.055)))
            pos += dash + gap

        # Small road spurs make shop/service entrances visually believable.
        for rx, ry, rw, rh in [
            (6.8, 15.0, 1.9, 8.5),
            (9.5, 15.0, 5.4, 1.8),
            (23.0, 15.0, 2.0, 8.0),
            (15.0, 8.8, 14.0, 1.8),
        ]:
            arcade.draw_lbwh_rectangle_filled(
                left + (rx - rw / 2) * scale,
                bottom + (ry - rh / 2) * scale,
                rw * scale,
                rh * scale,
                (72, 75, 78),
            )

    def _draw_buildings(self, world, left: float, bottom: float, scale: float):
        theme = {
            "shop": ((205, 190, 158), (232, 220, 187), (177, 75, 65)),
            "medical": ((191, 215, 216), (225, 237, 235), (62, 137, 140)),
            "industrial": ((145, 148, 153), (178, 180, 181), (225, 174, 64)),
        }
        for idx, b in enumerate(world.buildings):
            x, y, w, h = self._rect_xywh(b.rect, left, bottom, scale)
            floor, tile, accent = theme.get(b.theme, theme["shop"])

            # Foundation shadow and interior floor.
            arcade.draw_lbwh_rectangle_filled(x + 3, y - 4, w, h, (20, 23, 27, 75))
            arcade.draw_lbwh_rectangle_filled(x, y, w, h, floor)

            # Floor grid gives readable scale and interiority.
            tile_step = max(12.0, 0.8 * scale)
            xx = x + tile_step
            while xx < x + w:
                arcade.draw_line(xx, y, xx, y + h, (*tile, 75), 1)
                xx += tile_step
            yy = y + tile_step
            while yy < y + h:
                arcade.draw_line(x, yy, x + w, yy, (*tile, 75), 1)
                yy += tile_step

            # Shop/clinic/warehouse visual identity.
            header_h = min(20, max(9, h * 0.16))
            arcade.draw_lbwh_rectangle_filled(x + 3, y + h - header_h - 3, max(5, w - 6), header_h, (*accent, 215))
            self._text(
                f"building:{idx}",
                b.label,
                x + w / 2,
                y + h - header_h / 2 - 3,
                (250, 249, 244),
                9 if scale < 22 else 10,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )

            # Windows along the lower facade make walls legible as architecture.
            pane_y = y + 5
            pane_w = max(7, min(15, w * 0.15))
            cursor = x + 8
            while cursor + pane_w < x + w - 8:
                arcade.draw_lbwh_rectangle_filled(cursor, pane_y, pane_w, max(4, 0.18 * scale), (103, 164, 183, 190))
                arcade.draw_lbwh_rectangle_outline(cursor, pane_y, pane_w, max(4, 0.18 * scale), (225, 235, 236), 1)
                cursor += pane_w + 8

            if b.theme == "medical":
                # Large medical cross painted on the floor.
                cx, cy = x + w * 0.5, y + h * 0.46
                s = max(5, 0.35 * scale)
                arcade.draw_lbwh_rectangle_filled(cx - s * 0.28, cy - s, s * 0.56, s * 2, (210, 73, 76, 170))
                arcade.draw_lbwh_rectangle_filled(cx - s, cy - s * 0.28, s * 2, s * 0.56, (210, 73, 76, 170))
            elif b.theme == "industrial":
                # Hazard stripe zone, decorative but clearly floor paint.
                base_y = y + min(h * 0.22, 1.2 * scale)
                for n in range(7):
                    x0 = x + 6 + n * max(8, w / 8)
                    arcade.draw_line(x0, base_y, x0 + max(7, 0.35 * scale), base_y + max(4, 0.2 * scale), (225, 174, 64, 170), 2)

    def _draw_areas(self, world, left: float, bottom: float, scale: float):
        for area in world.areas:
            x, y, w, h = self._rect_xywh(area.rect, left, bottom, scale)
            if area.kind is ObjectKind.WATER:
                # Shore + layered water creates much better depth than a flat patch.
                arcade.draw_lbwh_rectangle_filled(x - 3, y - 3, w + 6, h + 6, (185, 173, 122))
                arcade.draw_lbwh_rectangle_filled(x, y, w, h, (54, 142, 181))
                stripe = max(10, 0.55 * scale)
                yy = y + stripe * 0.55
                phase = 0
                while yy < y + h:
                    start = x + 5 + (phase % 2) * stripe * 0.35
                    while start < x + w - 6:
                        arcade.draw_line(start, yy, min(start + stripe * 0.65, x + w - 5), yy, (157, 218, 232, 175), 1.5)
                        start += stripe * 1.1
                    yy += stripe
                    phase += 1
            elif area.kind is ObjectKind.STEAM:
                arcade.draw_lbwh_rectangle_filled(x, y, w, h, (213, 221, 223, 38))
                for fx, fy, rr in ((0.22, 0.30, 0.22), (0.55, 0.62, 0.30), (0.78, 0.35, 0.18), (0.35, 0.78, 0.16)):
                    arcade.draw_circle_filled(x + w * fx, y + h * fy, max(5, min(w, h) * rr), (229, 234, 235, 68))

    def _draw_decorations(self, world, left: float, bottom: float, scale: float):
        for i, d in enumerate(world.decorations):
            sx, sy = self._xy(d.x, d.y, left, bottom, scale)
            if d.kind == "bush":
                r = max(5, d.w * scale * 0.36)
                arcade.draw_circle_filled(sx - r * 0.45, sy, r, (54, 108, 57))
                arcade.draw_circle_filled(sx + r * 0.45, sy + r * 0.08, r * 0.9, (65, 123, 62))
                arcade.draw_circle_filled(sx, sy + r * 0.2, r * 0.85, (73, 136, 67))
            elif d.kind == "lamp":
                arcade.draw_circle_filled(sx, sy, max(2, 0.07 * scale), (54, 55, 58))
                arcade.draw_circle_outline(sx, sy, max(4, 0.14 * scale), (242, 220, 130, 160), 1)
            elif d.kind == "parking":
                w, h = d.w * scale, d.h * scale
                arcade.draw_lbwh_rectangle_outline(sx - w / 2, sy - h / 2, w, h, (210, 213, 214, 120), 1)
                slots = max(2, int(d.w / 1.2))
                for n in range(1, slots):
                    xx = sx - w / 2 + w * n / slots
                    arcade.draw_line(xx, sy - h / 2, xx, sy + h / 2, (210, 213, 214, 90), 1)
            elif d.kind == "crosswalk":
                w, h = d.w * scale, d.h * scale
                vertical = h > w
                bands = 6
                for n in range(bands):
                    if vertical:
                        yy = sy - h / 2 + (n + 0.2) * h / bands
                        arcade.draw_lbwh_rectangle_filled(sx - w / 2, yy, w, h / bands * 0.45, (226, 226, 218, 190))
                    else:
                        xx = sx - w / 2 + (n + 0.2) * w / bands
                        arcade.draw_lbwh_rectangle_filled(xx, sy - h / 2, w / bands * 0.45, h, (226, 226, 218, 190))

    def _draw_rect_obstacles(self, world, left: float, bottom: float, scale: float):
        for e in world.rects:
            x, y, w, h = self._rect_xywh(e.rect, left, bottom, scale)
            if e.kind is ObjectKind.WALL:
                # Thick masonry wall with inner highlight.
                arcade.draw_lbwh_rectangle_filled(x, y, max(2, w), max(2, h), (57, 57, 61))
                inset = min(2.0, max(0.7, min(w, h) * 0.18))
                if w > h:
                    arcade.draw_line(x, y + h - inset, x + w, y + h - inset, (113, 109, 104), 1)
                else:
                    arcade.draw_line(x + w - inset, y, x + w - inset, y + h, (113, 109, 104), 1)
                continue

            label = str(e.properties.get("label", "obstacle"))
            if label == "bench":
                arcade.draw_lbwh_rectangle_filled(x, y, w, h, (100, 67, 42))
                for n in (0.2, 0.5, 0.8):
                    yy = y + h * n
                    arcade.draw_line(x + 2, yy, x + w - 2, yy, (171, 118, 65), 2)
            elif label == "sign":
                arcade.draw_lbwh_rectangle_filled(x + w * 0.42, y, w * 0.16, h * 0.55, (74, 76, 79))
                arcade.draw_lbwh_rectangle_filled(x, y + h * 0.45, w, h * 0.45, (54, 127, 159))
                arcade.draw_lbwh_rectangle_outline(x, y + h * 0.45, w, h * 0.45, (218, 232, 235), 1)
            elif label == "pallet":
                arcade.draw_lbwh_rectangle_filled(x, y, w, h, (117, 77, 45))
                for n in range(4):
                    xx = x + (n + 0.5) * w / 4
                    arcade.draw_line(xx, y + 1, xx, y + h - 1, (187, 133, 73), 2)
            elif label == "crate":
                arcade.draw_texture_rect(self.crate_texture, arcade.XYWH(x + w / 2, y + h / 2, max(12, w), max(12, h)))
            elif label == "pipe":
                arcade.draw_lbwh_rectangle_filled(x, y + h * 0.22, w, h * 0.56, (107, 116, 119))
                arcade.draw_circle_filled(x + 2, y + h / 2, max(3, h * 0.46), (76, 84, 88))
                arcade.draw_circle_filled(x + w - 2, y + h / 2, max(3, h * 0.46), (76, 84, 88))
            else:
                arcade.draw_lbwh_rectangle_filled(x, y, w, h, (96, 83, 73))

    def _draw_doors(self, world, left: float, bottom: float, scale: float):
        for d in world.doors:
            x, y, w, h = self._rect_xywh(d.rect, left, bottom, scale)
            if d.open:
                # Open doorway: dark threshold + leaf swung aside.
                arcade.draw_lbwh_rectangle_filled(x, y, max(3, w), max(3, h), (48, 49, 51))
                if w > h:
                    cx, cy = x + w / 2, y + h / 2
                    arcade.draw_line(cx, cy, cx + min(18, w * 0.45), cy + min(18, 0.75 * scale), (154, 105, 58), 3)
                else:
                    cx, cy = x + w / 2, y + h / 2
                    arcade.draw_line(cx, cy, cx + min(18, 0.75 * scale), cy + min(18, h * 0.45), (154, 105, 58), 3)
            else:
                arcade.draw_lbwh_rectangle_filled(x, y, max(3, w), max(3, h), (147, 92, 48))
                arcade.draw_lbwh_rectangle_outline(x, y, max(3, w), max(3, h), (217, 161, 90), 1)
                arcade.draw_circle_filled(x + w * 0.78, y + h * 0.55, max(1.5, 0.04 * scale), (235, 210, 122))

    def _draw_world_objects(self, world, left: float, bottom: float, scale: float):
        for obj in world.objects:
            sx, sy = self._xy(obj.x, obj.y, left, bottom, scale)
            if obj.kind is ObjectKind.MOVING_OBSTACLE:
                size = max(20, obj.radius * 2 * scale)
                arcade.draw_circle_filled(sx + 3, sy - 3, size * 0.54, (20, 20, 20, 70))
                arcade.draw_texture_rect(self.crate_texture, arcade.XYWH(sx, sy, size, size))
                continue
            if obj.kind is ObjectKind.OBSTACLE and obj.properties.get("label") == "tree":
                self._draw_tree(sx, sy, obj.radius * scale)
                continue
            self._draw_item_icon(obj, sx, sy, scale)

    @staticmethod
    def _draw_tree(sx: float, sy: float, radius_px: float):
        r = max(8, radius_px)
        arcade.draw_circle_filled(sx + 3, sy - 4, r * 0.92, (30, 49, 31, 100))
        arcade.draw_circle_filled(sx, sy, r * 0.34, (94, 62, 37))
        arcade.draw_circle_filled(sx - r * 0.35, sy + r * 0.12, r * 0.72, (47, 105, 53))
        arcade.draw_circle_filled(sx + r * 0.32, sy + r * 0.15, r * 0.68, (58, 126, 58))
        arcade.draw_circle_filled(sx, sy + r * 0.42, r * 0.74, (67, 137, 62))
        arcade.draw_circle_outline(sx, sy + r * 0.12, r * 0.92, (37, 83, 42), 1.5)

    def _draw_item_icon(self, obj, sx: float, sy: float, scale: float):
        label = str(obj.properties.get("label", ""))
        r = max(5.0, obj.radius * scale * 1.18)
        # Shared drop shadow and pickup halo.
        arcade.draw_circle_filled(sx + 2, sy - 2, r * 1.05, (15, 18, 20, 70))
        arcade.draw_circle_outline(sx, sy, r * 1.35, (240, 240, 230, 55), 1)

        if obj.kind is ObjectKind.FOOD:
            arcade.draw_circle_filled(sx, sy, r * 0.75, (218, 75, 63))
            arcade.draw_circle_filled(sx + r * 0.42, sy + r * 0.5, r * 0.24, (79, 139, 62))
            arcade.draw_line(sx, sy + r * 0.65, sx + r * 0.12, sy + r * 1.0, (82, 58, 38), 2)
        elif obj.kind is ObjectKind.DRINK:
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.48, sy - r * 0.72, r * 0.96, r * 1.36, (70, 170, 214))
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.29, sy + r * 0.58, r * 0.58, r * 0.30, (212, 230, 233))
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.35, sy - r * 0.16, r * 0.70, r * 0.22, (225, 245, 248, 150))
        elif obj.kind is ObjectKind.MEDICINE:
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.78, sy - r * 0.58, r * 1.56, r * 1.16, (235, 237, 233))
            arcade.draw_lbwh_rectangle_outline(sx - r * 0.78, sy - r * 0.58, r * 1.56, r * 1.16, (146, 153, 153), 1)
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.16, sy - r * 0.43, r * 0.32, r * 0.86, (207, 68, 71))
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.43, sy - r * 0.16, r * 0.86, r * 0.32, (207, 68, 71))
        elif obj.kind is ObjectKind.WEAPON and label == "rolling_pin":
            arcade.draw_line(sx - r, sy, sx + r, sy, (178, 117, 61), max(4, int(r * 0.55)))
            arcade.draw_line(sx - r * 1.35, sy, sx - r * 0.9, sy, (100, 63, 37), 3)
            arcade.draw_line(sx + r * 0.9, sy, sx + r * 1.35, sy, (100, 63, 37), 3)
        elif obj.kind is ObjectKind.WEAPON and label == "potato_launcher":
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.95, sy - r * 0.32, r * 1.65, r * 0.64, (78, 115, 83))
            arcade.draw_circle_outline(sx + r * 0.73, sy, r * 0.36, (188, 207, 187), 2)
            arcade.draw_line(sx - r * 0.35, sy - r * 0.3, sx - r * 0.72, sy - r * 0.95, (108, 75, 49), 4)
        elif obj.kind is ObjectKind.AMMUNITION:
            arcade.draw_circle_filled(sx, sy, r * 0.80, (169, 125, 72))
            for dx, dy in ((-0.3, 0.2), (0.25, -0.28), (0.25, 0.35)):
                arcade.draw_circle_filled(sx + r * dx, sy + r * dy, max(1, r * 0.09), (103, 74, 45))
        elif obj.kind is ObjectKind.WEAPON_MODIFIER:
            arcade.draw_circle_filled(sx, sy, r * 0.82, (153, 92, 188))
            arcade.draw_circle_outline(sx, sy, r * 0.45, (235, 219, 241), 2)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                arcade.draw_line(sx + dx * r * 0.45, sy + dy * r * 0.45, sx + dx * r * 0.9, sy + dy * r * 0.9, (235, 219, 241), 2)
        elif obj.kind is ObjectKind.RAW_MATERIAL:
            material = str(obj.properties.get("material", ""))
            color = {
                "wood": (157, 105, 58),
                "metal": (132, 143, 150),
                "cloth": (194, 112, 126),
                "plastic": (87, 155, 168),
            }.get(material, (148, 148, 148))
            arcade.draw_lbwh_rectangle_filled(sx - r * 0.75, sy - r * 0.65, r * 1.5, r * 1.3, color)
            arcade.draw_lbwh_rectangle_outline(sx - r * 0.75, sy - r * 0.65, r * 1.5, r * 1.3, (230, 231, 227), 1)
            arcade.draw_line(sx - r * 0.55, sy, sx + r * 0.55, sy, (255, 255, 255, 95), 1)
        else:
            arcade.draw_circle_filled(sx, sy, r * 0.72, (220, 220, 215))

    def _draw_projectiles(self, debug, left: float, bottom: float, scale: float):
        for p in debug["projectiles"]:
            sx, sy = self._xy(p.x, p.y, left, bottom, scale)
            r = max(4, p.radius * scale)
            arcade.draw_circle_filled(sx + 2, sy - 2, r, (22, 20, 17, 80))
            arcade.draw_circle_filled(sx, sy, r, (173, 128, 72))
            arcade.draw_circle_filled(sx - r * 0.25, sy + r * 0.20, max(1, r * 0.10), (103, 74, 45))

    def _draw_agents(self, debug, left: float, bottom: float, scale: float):
        ids = list(debug["agents"].keys())
        if self.selected_agent in ids:
            ids.remove(self.selected_agent)
            ids.append(self.selected_agent)

        for aid in ids:
            st = debug["agents"][aid]
            if st is None:
                continue
            sx, sy = self._xy(st.x, st.y, left, bottom, scale)
            size = max(24, st.radius * 2.3 * scale)
            if not st.alive:
                arcade.draw_circle_filled(sx, sy, size * 0.42, (65, 64, 62, 190))
                arcade.draw_line(sx - size * 0.2, sy - size * 0.2, sx + size * 0.2, sy + size * 0.2, (196, 191, 181), 2)
                arcade.draw_line(sx - size * 0.2, sy + size * 0.2, sx + size * 0.2, sy - size * 0.2, (196, 191, 181), 2)
                continue

            ring = self._agent_color(aid)
            if aid == self.selected_agent:
                arcade.draw_circle_outline(sx, sy, size * 0.72, (255, 233, 104), 3)
            arcade.draw_circle_filled(sx, sy, size * 0.59, (*ring, 220))
            texture = self._agent_textures[sum(ord(c) for c in aid) % len(self._agent_textures)]
            arcade.draw_texture_rect(texture, arcade.XYWH(sx, sy, size, size))

            # Direction arrow and compact health strip.
            dx, dy = heading_vector(st.orientation)
            arcade.draw_line(sx, sy, sx + dx * size * 0.82, sy + dy * size * 0.82, (250, 249, 240), 2)
            bw = size * 0.95
            by = sy + size * 0.66
            arcade.draw_lbwh_rectangle_filled(sx - bw / 2, by, bw, 4, (58, 58, 58, 190))
            arcade.draw_lbwh_rectangle_filled(sx - bw / 2, by, bw * max(0, min(1, st.health)), 4, (83, 190, 111, 230))
            self._text(
                f"agent:{aid}",
                aid,
                sx,
                by + 8,
                (249, 249, 246),
                8,
                bold=True,
                anchor_x="center",
            )

    # ------------------------------------------------------------------
    # Sensor overlays
    # ------------------------------------------------------------------
    def _draw_sensor_overlays(self, debug, left: float, bottom: float, scale: float):
        aid = self.selected_agent
        if aid is None or aid not in debug["agents"] or aid not in debug["observations"]:
            return
        st = debug["agents"][aid]
        obs = debug["observations"][aid]
        if st is None or not st.alive:
            return
        sx, sy = self._xy(st.x, st.y, left, bottom, scale)

        if self.show_vision:
            half = self.env.config.agent.vision_fov / 2
            rng_px = self.env.config.agent.vision_range * scale
            for angle in (st.orientation - half, st.orientation + half):
                dx, dy = heading_vector(angle)
                arcade.draw_line(sx, sy, sx + dx * rng_px, sy + dy * rng_px, (246, 224, 100, 125), 1)
            # A few range rings provide scale without flooding the view.
            for fraction in (0.33, 0.66, 1.0):
                arcade.draw_circle_outline(sx, sy, rng_px * fraction, (246, 224, 100, 28), 1)
            for det in obs.vision:
                angle = st.orientation + det.bearing
                dx, dy = heading_vector(angle)
                length = min(self.env.config.agent.vision_range, det.distance + 0.15) * scale
                arcade.draw_line(sx, sy, sx + dx * length, sy + dy * length, (255, 233, 91, 160), 1)
                arcade.draw_circle_filled(sx + dx * length, sy + dy * length, 2.5, (255, 233, 91, 210))

        if self.show_hearing:
            for sound in obs.hearing[:8]:
                angle = st.orientation + sound.bearing
                length = (1.2 + sound.intensity * 4.0) * scale
                dx, dy = heading_vector(angle)
                arcade.draw_line(sx, sy, sx + dx * length, sy + dy * length, (104, 198, 255, 205), 2)
                for sign in (-1, 1):
                    bx, by = heading_vector(angle + sign * sound.bearing_uncertainty)
                    arcade.draw_line(sx, sy, sx + bx * length * 0.75, sy + by * length * 0.75, (104, 198, 255, 90), 1)

        if self.show_touch:
            for contact in obs.touch:
                angle = st.orientation + contact.bearing
                dx, dy = heading_vector(angle)
                r = st.radius * scale * 1.15
                arcade.draw_circle_filled(sx + dx * r, sy + dy * r, 4, (255, 129, 194))

        if self.teacher_debug:
            for sound in debug["sounds"]:
                tx, ty = self._xy(sound.source_x, sound.source_y, left, bottom, scale)
                arcade.draw_line(tx - 5, ty - 5, tx + 5, ty + 5, (255, 80, 160), 2)
                arcade.draw_line(tx - 5, ty + 5, tx + 5, ty - 5, (255, 80, 160), 2)

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------
    def _draw_panel(self, debug, left: float, bottom: float, width: float, height: float):
        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, (29, 34, 43, 248))
        arcade.draw_lbwh_rectangle_outline(left, bottom, width, height, (59, 67, 78), 1)
        x = left + 18
        y = bottom + height - 32

        self._text("panel:title", "RL ARENA", x, y, (245, 245, 245), 18, bold=True)
        y -= 28
        self._text(
            "panel:tick",
            f"tick {debug['tick']}   •   affichage ×{self.speed_multiplier:g}",
            x,
            y,
            (180, 190, 204),
            11,
        )
        y -= 28

        aid = self.selected_agent
        if aid is None or aid not in debug["agents"]:
            self._text("panel:no-agent", "Aucun agent", x, y, (230, 230, 230), 12)
            return
        st = debug["agents"][aid]
        obs = debug["observations"].get(aid)
        self._text("panel:agent", f"Agent : {aid}", x, y, self._agent_color(aid), 15, bold=True)
        y -= 26
        if st is None:
            return

        for index, (label, value, bar_color) in enumerate(
            (
                ("Santé", st.health, (84, 190, 112)),
                ("Énergie", st.energy, (223, 183, 66)),
                ("Hydratation", st.hydration, (69, 169, 213)),
                ("Satiété", st.satiety, (216, 136, 74)),
            )
        ):
            self._draw_bar(f"bar:{index}", x, y, width - 36, label, value, bar_color)
            y -= 27

        self._text("panel:speed", f"Vitesse : {st.speed:.2f} m/s", x, y, (225, 225, 225), 11)
        y -= 18
        self._text(
            "panel:load",
            f"Charge : {st.carried_mass:.1f}/{self.env.config.agent.max_carry_mass:.0f} kg",
            x,
            y,
            (225, 225, 225),
            11,
        )
        y -= 18
        stats = debug.get("stats", {}).get(aid, {})
        self._text(
            "panel:reward",
            f"Reward cumulé : {stats.get('cumulative_reward', 0.0):.2f}",
            x,
            y,
            (225, 225, 225),
            11,
        )
        y -= 24

        self._text("panel:inventory-title", "Inventaire", x, y, (245, 245, 245), 12, bold=True)
        y -= 20
        if not st.inventory:
            self._text("panel:inv-empty", "— vide —", x + 8, y, (150, 160, 170), 10)
            y -= 18
        else:
            for line_index, (slot, item) in enumerate(sorted(st.inventory.items())):
                equipped = "  ◀ équipé" if slot == st.equipped_slot else ""
                self._text(
                    f"panel:inv:{line_index}",
                    f"[{slot}] {self._friendly_item_name(item)} ({item.mass:.1f} kg){equipped}",
                    x + 4,
                    y,
                    (205, 213, 222),
                    10,
                )
                y -= 17

        y -= 8
        if obs is not None:
            self._text(
                "panel:sensors",
                f"Capteurs : {len(obs.vision)} visuels • {len(obs.hearing)} sons • {len(obs.touch)} contacts",
                x,
                y,
                (190, 199, 211),
                10,
            )
            y -= 22

        command_y = max(bottom + 110, y)
        self._text("panel:commands", "Commandes", x, command_y, (245, 245, 245), 11, bold=True)
        y2 = command_y - 20
        lines = [
            "Espace pause   N pas-à-pas   Tab agent",
            "V vision   H audition   T toucher   D prof",
            "+ / - vitesse   clic : sélectionner",
        ]
        for i, line in enumerate(lines):
            self._text(f"panel:cmd:{i}", line, x, y2, (157, 168, 181), 9)
            y2 -= 16

    def _draw_bar(self, key: str, x: float, y: float, width: float, label: str, value: float, fill_color):
        self._text(f"{key}:label", label, x, y + 3, (218, 222, 227), 10)
        bx = x + 88
        bw = max(50, width - 90)
        arcade.draw_lbwh_rectangle_filled(bx, y, bw, 13, (60, 68, 78))
        arcade.draw_lbwh_rectangle_filled(bx, y, bw * max(0, min(1, value)), 13, fill_color)
        self._text(
            f"{key}:value",
            f"{value:0.2f}",
            bx + bw - 3,
            y + 1,
            (245, 245, 245),
            8,
            anchor_x="right",
        )

    @staticmethod
    def _friendly_item_name(item) -> str:
        label = str(item.properties.get("label", item.kind.name.lower()))
        return {
            "rolling_pin": "rouleau à pâtisserie",
            "potato_launcher": "lance-patate",
            "potato": "pomme de terre",
            "water_bottle": "bouteille d'eau",
            "bottle": "bouteille",
            "bandage": "bandage",
            "medicine": "médicament",
            "reinforced_spring": "ressort renforcé",
        }.get(label, label.replace("_", " "))

    @staticmethod
    def _agent_color(aid: str):
        palette = [
            (96, 185, 255),
            (255, 141, 106),
            (174, 223, 98),
            (221, 132, 255),
            (255, 213, 87),
            (105, 221, 190),
        ]
        return palette[sum(ord(c) for c in aid) % len(palette)]

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused
        elif symbol == arcade.key.N:
            self.single_step = True
            self.paused = True
        elif symbol == arcade.key.V:
            self.show_vision = not self.show_vision
        elif symbol == arcade.key.H:
            self.show_hearing = not self.show_hearing
        elif symbol == arcade.key.T:
            self.show_touch = not self.show_touch
        elif symbol == arcade.key.D:
            self.teacher_debug = not self.teacher_debug
        elif symbol == arcade.key.TAB:
            self._cycle_agent()
        elif symbol in (arcade.key.EQUAL, getattr(arcade.key, "NUM_ADD", -99999)):
            self.speed_multiplier = min(16.0, self.speed_multiplier * 2)
        elif symbol in (arcade.key.MINUS, getattr(arcade.key, "NUM_SUBTRACT", -99998)):
            self.speed_multiplier = max(0.25, self.speed_multiplier / 2)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        left, bottom, world_px, scale = self._world_layout()
        if not (left <= x <= left + world_px and bottom <= y <= bottom + world_px):
            return
        wx = (x - left) / scale
        wy = (y - bottom) / scale
        debug = self.env.debug_state()
        choices = []
        for aid, st in debug.get("agents", {}).items():
            if st is None or not st.alive:
                continue
            d2 = (st.x - wx) ** 2 + (st.y - wy) ** 2
            choices.append((d2, aid))
        if choices:
            d2, aid = min(choices)
            if d2 <= 1.5**2:
                self.selected_agent = aid

    def _cycle_agent(self):
        debug = self.env.debug_state()
        ids = [aid for aid, st in debug.get("agents", {}).items() if st is not None and st.alive]
        if not ids:
            return
        if self.selected_agent not in ids:
            self.selected_agent = ids[0]
            return
        idx = (ids.index(self.selected_agent) + 1) % len(ids)
        self.selected_agent = ids[idx]


def run_arcade(env: TrainingEnvironment) -> None:
    ArenaWindow(env)
    arcade.run()
