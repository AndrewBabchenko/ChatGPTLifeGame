"""
Life Game Demo - Pygame Visualization
Shows trained Actor-Critic agents in action with pheromones, energy, and age systems
"""

import os
import sys
import time
import random
import torch

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required for the demo. Install it with:\n"
        "  python -m pip install pygame"
    ) from exc

from src.config import SimulationConfig
from src.core.animal import Animal
from src.models.actor_critic_network import ActorCriticNetwork
from src.core.pheromone_system import PheromoneMap


# Windows 11 Fluent Design dark theme tokens.
THEME_TOKENS = {
    # Background layers - Mica-inspired gradient
    "bg": (11, 18, 32),              # Deep dark background (#0B1220)
    "mica_top": (15, 22, 35),        # Subtle gradient start
    "mica_bot": (8, 14, 26),         # Subtle gradient end
    
    # Card surfaces with Acrylic-like appearance
    "card": (17, 25, 40),            # Base card surface
    "card_elev": (20, 28, 44),       # Elevated card (hover state)
    "card_reflection": (35, 45, 65, 25),  # Subtle top reflection overlay
    
    # Borders and dividers
    "border": (42, 52, 74, 120),     # Border with translucency
    "divider": (32, 44, 64),         # Subtle divider lines
    
    # Typography hierarchy
    "text": (230, 236, 246),         # Primary text (#E6ECF6)
    "text_secondary": (170, 182, 202), # Secondary text
    "text_muted": (125, 135, 155),   # Muted/disabled text
    
    # Button states - Primary accent
    "primary": (60, 140, 230),       # Primary button base
    "primary_hover": (80, 156, 240), # Primary button hover
    "primary_pressed": (48, 124, 208), # Primary button pressed
    
    # Button states - Secondary neutral
    "secondary": (26, 36, 52),       # Secondary button base
    "secondary_hover": (34, 46, 64), # Secondary button hover
    "secondary_pressed": (22, 30, 44), # Secondary button pressed
    
    # Simulation visualization colors
    "prey": (90, 200, 250),          # Prey entities (light blue)
    "pred": (255, 100, 100),         # Predator entities (red)
    "warn": (255, 180, 60),          # Warning state (hungry predators)
    "grid": (30, 42, 62),            # Field grid lines
    
    # Shadow and depth
    "shadow": (0, 0, 0, 85),         # Drop shadow
    
    # Layout constants
    "pad": 16,                       # Global padding
    "gap": 16,                       # Element spacing
    "card_radius": 14,               # Card border radius
    "button_radius": 10,             # Button border radius
}



def _with_alpha(color, alpha):
    """Return RGBA tuple from RGB + alpha."""
    return (color[0], color[1], color[2], alpha)


def _to_screen(x, y, origin, scale, padding):
    """Convert simulation coords to screen coords with padding/clamp."""
    x = max(SimulationConfig.FIELD_MIN, min(SimulationConfig.FIELD_MAX, x))
    y = max(SimulationConfig.FIELD_MIN, min(SimulationConfig.FIELD_MAX, y))
    sx = origin[0] + padding + int((x - SimulationConfig.FIELD_MIN) * scale)
    sy = origin[1] + padding + int((y - SimulationConfig.FIELD_MIN) * scale)
    return sx, sy


def _draw_triangle(surface, center, size, color, outline):
    """Draw a small triangle for predators."""
    cx, cy = center
    points = [
        (cx, cy - size),
        (cx - size, cy + size),
        (cx + size, cy + size),
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, outline, points, width=1)


def _draw_shadow_rect(surface, rect, radius, shadow_color, offset=(0, 4), blur=10):
    """Soft drop shadow using a blurred rect buffer."""
    shadow = pygame.Surface((rect.width + blur * 2, rect.height + blur * 2), pygame.SRCALPHA)
    shadow_rect = pygame.Rect(blur, blur, rect.width, rect.height)
    pygame.draw.rect(shadow, shadow_color, shadow_rect, border_radius=radius)
    surface.blit(shadow, (rect.x + offset[0] - blur, rect.y + offset[1] - blur))


def _draw_card(surface, rect, theme, radius=None, elevated=False, shadow=True):
    """Windows 11 Fluent card with shadow, fill, border, and subtle top reflection."""
    if radius is None:
        radius = theme["card_radius"]
    
    # Draw drop shadow for depth
    if shadow:
        _draw_shadow_rect(surface, rect, radius, theme["shadow"], offset=(0, 3), blur=12)
    
    # Fill card with base or elevated color
    fill = theme["card_elev"] if elevated else theme["card"]
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    
    # Draw border
    pygame.draw.rect(surface, theme["border"], rect, width=1, border_radius=radius)
    
    # Subtle reflection overlay at top for Acrylic effect
    if "card_reflection" in theme:
        reflection = pygame.Surface((rect.width, max(1, int(rect.height * 0.15))), pygame.SRCALPHA)
        reflection.fill(theme["card_reflection"])
        surface.blit(reflection, (rect.x, rect.y + 1))



def _draw_text(surface, text, pos, font, color):
    """Text helper to keep rendering consistent."""
    surface.blit(font.render(text, True, color), pos)


def _draw_button(surface, rect, label, font, theme, style, hovered, pressed, icon=None):
    """Windows 11 Fluent button with proper hover/pressed states and optional icon."""
    if style == "primary":
        # Primary button uses accent colors
        fill = theme["primary_pressed"] if pressed else theme["primary_hover"] if hovered else theme["primary"]
        text_color = (255, 255, 255)
        border_color = theme["border"]
    else:
        # Secondary button uses neutral colors
        fill = theme["secondary_pressed"] if pressed else theme["secondary_hover"] if hovered else theme["secondary"]
        text_color = theme["text"]
        border_color = theme["border"]
    
    # Draw button background
    pygame.draw.rect(surface, fill, rect, border_radius=theme["button_radius"])
    pygame.draw.rect(surface, border_color, rect, width=1, border_radius=theme["button_radius"])
    
    # Draw icon if provided
    label_x = rect.x + 12
    if icon:
        icon(surface, (rect.x + 10, rect.y + rect.height // 2), text_color)
        label_x = rect.x + 28
    
    # Draw text label centered vertically
    text = font.render(label, True, text_color)
    surface.blit(text, (label_x, rect.y + (rect.height - text.get_height()) // 2))



def _draw_slider(surface, rect, value, theme, hovered):
    """Windows 11 Fluent slider with track and draggable knob."""
    # Draw track background
    pygame.draw.rect(surface, theme["secondary"], rect, border_radius=6)
    pygame.draw.rect(surface, theme["border"], rect, width=1, border_radius=6)
    
    # Draw filled portion up to knob position
    if value > 0:
        filled_rect = pygame.Rect(rect.x, rect.y, int(value * rect.width), rect.height)
        pygame.draw.rect(surface, theme["primary"], filled_rect, border_radius=6)
    
    # Draw draggable knob
    knob_x = rect.x + int(value * rect.width)
    knob_rect = pygame.Rect(knob_x - 6, rect.y - 4, 12, rect.height + 8)
    knob_color = theme["primary_hover"] if hovered else theme["primary"]
    pygame.draw.rect(surface, knob_color, knob_rect, border_radius=6)
    pygame.draw.rect(surface, theme["border"], knob_rect, width=1, border_radius=6)
    
    return knob_rect



def _icon_play(surface, center, color):
    """Toolbar play icon."""
    x, y = center
    points = [(x - 4, y - 6), (x - 4, y + 6), (x + 6, y)]
    pygame.draw.polygon(surface, color, points)


def _icon_pause(surface, center, color):
    """Toolbar pause icon."""
    x, y = center
    pygame.draw.rect(surface, color, (x - 6, y - 6, 4, 12))
    pygame.draw.rect(surface, color, (x + 2, y - 6, 4, 12))


def _icon_step(surface, center, color):
    """Toolbar step icon."""
    x, y = center
    points = [(x - 6, y - 6), (x - 6, y + 6), (x + 2, y)]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.rect(surface, color, (x + 4, y - 6, 3, 12))


def _icon_reset(surface, center, color):
    """Toolbar reset icon."""
    x, y = center
    pygame.draw.circle(surface, color, (x, y), 6, width=2)
    pygame.draw.line(surface, color, (x, y - 8), (x + 6, y - 4), 2)


def _icon_random(surface, center, color):
    """Toolbar randomize icon."""
    x, y = center
    pygame.draw.circle(surface, color, (x, y), 2)
    pygame.draw.circle(surface, color, (x - 6, y + 4), 2)
    pygame.draw.circle(surface, color, (x + 6, y - 4), 2)


def _build_background(size, theme):
    """Windows 11 Mica-inspired background gradient from darker bottom to slightly lighter top."""
    w, h = size
    bg = pygame.Surface((w, h))
    
    # Create vertical gradient from mica_bot to mica_top
    mica_top = theme.get("mica_top", theme["bg"])
    mica_bot = theme.get("mica_bot", theme["bg"])
    
    for y in range(h):
        t = y / max(1, h)  # 0 at top, 1 at bottom
        # Interpolate from top to bottom
        r = int(mica_top[0] * (1 - t) + mica_bot[0] * t)
        g = int(mica_top[1] * (1 - t) + mica_bot[1] * t)
        b = int(mica_top[2] * (1 - t) + mica_bot[2] * t)
        pygame.draw.line(bg, (r, g, b), (0, y), (w, y))
    
    return bg



def create_population(config: SimulationConfig) -> list:
    """Create initial population of animals."""
    animals = []
    for _ in range(config.INITIAL_PREY_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Animal(x, y, "A", "#00ff00", predator=False)
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)

    for _ in range(config.INITIAL_PREDATOR_COUNT):
        x = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        y = random.randint(config.FIELD_MIN, config.FIELD_MAX)
        animal = Animal(x, y, "B", "#ff0000", predator=True)
        animal.energy = config.INITIAL_ENERGY
        animals.append(animal)

    return animals


def run_demo():
    """Run visual demo of trained models."""
    print("\n" + "=" * 70)
    print("  LIFE GAME DEMO - Predator/Prey Simulation (Pygame)")
    print("=" * 70)

    # Simulation and model setup.
    config = SimulationConfig()
    device = torch.device("cpu")

    print("\nLoading models...")
    model_prey = ActorCriticNetwork(config).to(device)
    model_predator = ActorCriticNetwork(config).to(device)

    try:
        model_prey.load_state_dict(torch.load("outputs/checkpoints/model_A_ppo.pth", map_location=device))
        model_predator.load_state_dict(torch.load("outputs/checkpoints/model_B_ppo.pth", map_location=device))
        print("Loaded trained PPO models")
    except FileNotFoundError:
        print("No trained models found, using random initialization")

    model_prey.eval()
    model_predator.eval()

    # Initialize world state.
    animals = create_population(config)
    pheromone_map = PheromoneMap(
        config.GRID_SIZE,
        decay_rate=config.PHEROMONE_DECAY,
        diffusion_rate=config.PHEROMONE_DIFFUSION,
    )

    # Pygame init and window creation.
    pygame.init()
    screen_w, screen_h = 1280, 780
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
    pygame.display.set_caption("Life Game - Predator/Prey Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 16)
    font_bold = pygame.font.SysFont("Segoe UI Semibold", 18)
    font_small = pygame.font.SysFont("Segoe UI", 13)
    font_tab = pygame.font.SysFont("Consolas", 13)

    # Theme setup (dark Fluent-style by default).
    theme = dict(THEME_TOKENS)
    accent_override = os.environ.get("LIFE_GAME_ACCENT")
    if accent_override:
        try:
            parts = [int(p) for p in accent_override.split(",")]
            if len(parts) == 3:
                theme["accent"] = tuple(parts)
        except ValueError:
            pass

    # Cached layout and background surfaces to avoid per-frame recompute.
    layout_cache = {}
    bg_cache = None
    bg_size = None
    sidebar_scroll = 0
    speed_value = 0.6
    sim_accum = 0.0
    paused = False
    ended = False
    modal_open = False
    step_count = 0
    prey_counts = []
    predator_counts = []
    births = 0
    deaths = 0
    meals = 0
    render_positions = {}
    last_frame = time.time()

    def reset_simulation():
        """Reset simulation state while keeping UI intact."""
        nonlocal animals, pheromone_map, step_count, prey_counts, predator_counts, births, deaths, meals, ended, modal_open
        animals = create_population(config)
        pheromone_map.reset()
        step_count = 0
        prey_counts.clear()
        predator_counts.clear()
        births = 0
        deaths = 0
        meals = 0
        ended = False
        modal_open = False

    def simulate_one_step():
        """Advance the simulation by one tick (no rendering)."""
        nonlocal animals, births, deaths, meals
        animals_to_remove = []
        new_animals = []
        mated_animals = set()

        for animal in animals[:]:
            animal.update_age()
            if animal.is_old(config):
                animals_to_remove.append(animal)
                deaths += 1
                continue

            animal_input = animal.get_enhanced_input(animals, config, pheromone_map).to(device)
            visible_animals = animal.communicate(animals, config)
            visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32).unsqueeze(0).to(device)

            model = model_prey if not animal.predator else model_predator
            with torch.no_grad():
                action, _, _ = model.get_action(animal_input, visible_animals_input)
            action_idx = action.item()

            old_pos = (animal.x, animal.y)
            new_x, new_y = animal._apply_action(action_idx, config)
            if new_x < config.FIELD_MIN:
                new_x = config.FIELD_MAX
            elif new_x > config.FIELD_MAX:
                new_x = config.FIELD_MIN
            if new_y < config.FIELD_MIN:
                new_y = config.FIELD_MAX
            elif new_y > config.FIELD_MAX:
                new_y = config.FIELD_MIN
            animal.x, animal.y = new_x, new_y
            moved = (animal.x, animal.y) != old_pos

            animal.update_energy(config, moved)
            if animal.is_exhausted():
                animals_to_remove.append(animal)
                deaths += 1
                continue

            animal.deposit_pheromones(pheromone_map, config)

        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()

        for animal in animals[:]:
            if animal.predator:
                ate = animal.eat(animals)
                if ate:
                    meals += 1
                    animal.energy = min(animal.max_energy, animal.energy + config.EATING_ENERGY_GAIN)
                else:
                    animal.steps_since_last_meal += 1
                    if animal.steps_since_last_meal >= config.STARVATION_THRESHOLD:
                        animals_to_remove.append(animal)
                        deaths += 1

        for animal in animals_to_remove:
            if animal in animals:
                animals.remove(animal)
        animals_to_remove.clear()

        for i, animal1 in enumerate(animals):
            if animal1.id in mated_animals or not animal1.can_reproduce(config):
                continue
            for animal2 in animals[i + 1:]:
                if animal2.id in mated_animals or not animal2.can_reproduce(config):
                    continue
                if animal1.can_mate(animal2):
                    mating_prob = (config.MATING_PROBABILITY_PREY
                                   if animal1.name == "A"
                                   else config.MATING_PROBABILITY_PREDATOR)
                    if random.random() < mating_prob:
                        child_x = (animal1.x + animal2.x) // 2
                        child_y = (animal1.y + animal2.y) // 2
                        child = Animal(child_x, child_y, animal1.name, animal1.color,
                                       {animal1.id, animal2.id}, animal1.predator)
                        child.energy = config.INITIAL_ENERGY
                        new_animals.append(child)
                        births += 1

                        animal1.energy -= config.MATING_ENERGY_COST
                        animal2.energy -= config.MATING_ENERGY_COST
                        animal1.mating_cooldown = config.MATING_COOLDOWN
                        animal2.mating_cooldown = config.MATING_COOLDOWN
                        mated_animals.add(animal1.id)
                        mated_animals.add(animal2.id)
                        break

        max_animals_current = max(0, config.MAX_ANIMALS - config.OTHER_SPECIES_CAPACITY)
        if len(animals) + len(new_animals) <= max_animals_current:
            animals.extend(new_animals)
        else:
            available_slots = max_animals_current - len(animals)
            if available_slots > 0:
                animals.extend(new_animals[:available_slots])

        for animal in animals:
            if animal.mating_cooldown > 0:
                animal.mating_cooldown -= 1
            animal.survival_time += 1

        pheromone_map.update()

    def recompute_layout():
        """Compute layout based on current window size."""
        pad = theme["pad"]
        gap = theme["gap"]
        header_h = 56
        sidebar_w = min(420, max(360, int(screen_w * 0.32)))
        content_w = screen_w - pad * 2 - gap - sidebar_w
        content_h = screen_h - pad * 2 - header_h
        field_size = min(content_w, content_h)
        field_rect = pygame.Rect(pad, pad + header_h, field_size, field_size)
        field_pad = 16
        field_scale = (field_rect.width - field_pad * 2) / (SimulationConfig.FIELD_MAX - SimulationConfig.FIELD_MIN)
        sidebar_rect = pygame.Rect(
            field_rect.right + gap,
            pad + header_h,
            sidebar_w,
            content_h,
        )
        header_rect = pygame.Rect(pad, pad, screen_w - pad * 2, header_h)
        layout_cache.update({
            "header": header_rect,
            "field": field_rect,
            "sidebar": sidebar_rect,
            "field_padding": field_pad,
            "scale": field_scale,
        })

    recompute_layout()

    running = True
    while running:
        if bg_size != (screen_w, screen_h):
            bg_cache = _build_background((screen_w, screen_h), theme)
            bg_size = (screen_w, screen_h)

        # Input polling (mouse/keyboard/resize/scroll).
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = pygame.mouse.get_pressed()[0]

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_simulation()
                elif event.key == pygame.K_s and paused:
                    simulate_one_step()
                elif event.key == pygame.K_ESCAPE and modal_open:
                    modal_open = False
            elif event.type == pygame.VIDEORESIZE:
                screen_w, screen_h = max(900, event.w), max(600, event.h)
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
                recompute_layout()
            elif event.type == pygame.MOUSEWHEEL:
                if layout_cache["sidebar"].collidepoint(mouse_pos):
                    sidebar_scroll -= event.y * 20

        header_rect = layout_cache["header"]
        field_rect = layout_cache["field"]
        sidebar_rect = layout_cache["sidebar"]

        # Toolbar controls (layout + hit targets).
        btn_w, btn_h = 88, 32
        btn_y = header_rect.y + (header_rect.height - btn_h) // 2
        btn_play = pygame.Rect(header_rect.x + 12, btn_y, btn_w, btn_h)
        btn_reset = pygame.Rect(btn_play.right + 10, btn_y, btn_w, btn_h)
        btn_step = pygame.Rect(btn_reset.right + 10, btn_y, btn_w, btn_h)
        btn_rand = pygame.Rect(btn_step.right + 10, btn_y, btn_w + 8, btn_h)
        slider_rect = pygame.Rect(btn_rand.right + 20, btn_y + 10, 160, 10)

        def _clicked(rect):
            for e in events:
                if e.type == pygame.MOUSEBUTTONDOWN and rect.collidepoint(e.pos):
                    return True
            return False

        play_hover = btn_play.collidepoint(mouse_pos)
        reset_hover = btn_reset.collidepoint(mouse_pos)
        step_hover = btn_step.collidepoint(mouse_pos)
        rand_hover = btn_rand.collidepoint(mouse_pos)
        slider_hover = slider_rect.collidepoint(mouse_pos)

        if _clicked(btn_play):
            paused = not paused
        if _clicked(btn_reset):
            reset_simulation()
        if _clicked(btn_step) and paused:
            simulate_one_step()
        if _clicked(btn_rand):
            random.seed()
            reset_simulation()

        if mouse_down and slider_hover:
            speed_value = (mouse_pos[0] - slider_rect.x) / max(1, slider_rect.width)
            speed_value = max(0.0, min(1.0, speed_value))

        # Simulation updates (throttled by speed slider).
        if not paused and not ended and not modal_open:
            sim_accum += 0.2 + speed_value * 1.8
            while sim_accum >= 1.0:
                simulate_one_step()
                sim_accum -= 1.0

        # Stats collection (counts only; no reflow).
        prey_positions = [(a.x, a.y) for a in animals if not a.predator]
        predator_positions = [(a.x, a.y) for a in animals if a.predator]
        if not ended:
            prey_counts.append(len(prey_positions))
            predator_counts.append(len(predator_positions))
            step_count += 1
            if len(prey_counts) > 500:
                prey_counts.pop(0)
                predator_counts.pop(0)

        alive_ids = {a.id for a in animals}
        for aid in list(render_positions.keys()):
            if aid not in alive_ids:
                del render_positions[aid]

        for animal in animals:
            target = (float(animal.x), float(animal.y))
            if animal.id not in render_positions:
                render_positions[animal.id] = target
            else:
                cx, cy = render_positions[animal.id]
                nx = cx + (target[0] - cx) * 0.35
                ny = cy + (target[1] - cy) * 0.35
                render_positions[animal.id] = (nx, ny)

        # Draw UI surfaces.
        screen.blit(bg_cache, (0, 0))
        _draw_card(screen, header_rect, theme, radius=12, elevated=True, shadow=True)
        _draw_card(screen, field_rect, theme, radius=14, elevated=False, shadow=True)
        _draw_card(screen, sidebar_rect, theme, radius=14, elevated=True, shadow=True)

        _draw_text(screen, "Life Game", (header_rect.x + 14, header_rect.y + 14), font_bold, theme["text"])
        _draw_text(screen, "Predator/Prey Simulation", (header_rect.x + 112, header_rect.y + 16), font_small, theme["text_muted"])

        _draw_button(
            screen,
            btn_play,
            "Play" if paused else "Pause",
            font_small,
            theme,
            "secondary",
            play_hover,
            play_hover and mouse_down,
            icon=_icon_play if paused else _icon_pause,
        )
        _draw_button(screen, btn_reset, "Reset", font_small, theme, "secondary", reset_hover, reset_hover and mouse_down, icon=_icon_reset)
        _draw_button(screen, btn_step, "Step", font_small, theme, "secondary", step_hover, step_hover and mouse_down, icon=_icon_step)
        _draw_button(screen, btn_rand, "Random", font_small, theme, "secondary", rand_hover, rand_hover and mouse_down, icon=_icon_random)
        _draw_text(screen, "Speed", (slider_rect.x - 50, slider_rect.y - 4), font_small, theme["text_muted"])
        _draw_slider(screen, slider_rect, speed_value, theme, slider_hover)

        # Subtle grid in field.
        field_pad = layout_cache["field_padding"]
        scale = layout_cache["scale"]
        grid_spacing = int((field_rect.width - field_pad * 2) / 8)
        grid_color = theme["grid"]
        for i in range(1, 8):
            x = field_rect.x + field_pad + i * grid_spacing
            y = field_rect.y + field_pad + i * grid_spacing
            pygame.draw.line(screen, grid_color, (x, field_rect.y + field_pad), (x, field_rect.bottom - field_pad), 1)
            pygame.draw.line(screen, grid_color, (field_rect.x + field_pad, y), (field_rect.right - field_pad, y), 1)

        # Draw animals (smoothed render positions).
        for animal in animals:
            if animal.predator:
                continue
            x, y = render_positions[animal.id]
            sx, sy = _to_screen(x, y, (field_rect.x, field_rect.y), scale, field_pad)
            pygame.draw.circle(screen, theme["prey"], (sx, sy), 5)
            pygame.draw.circle(screen, theme["border"][:3], (sx, sy), 5, width=1)

        for animal in animals:
            if not animal.predator:
                continue
            x, y = render_positions[animal.id]
            sx, sy = _to_screen(x, y, (field_rect.x, field_rect.y), scale, field_pad)
            pred_color = theme["pred"] if animal.steps_since_last_meal < config.HUNGER_THRESHOLD else theme["warn"]
            _draw_triangle(screen, (sx, sy), 6, pred_color, theme["border"][:3])

        # Legend in field card.
        legend_x = field_rect.right - 150
        legend_y = field_rect.y + 14
        pygame.draw.circle(screen, theme["prey"], (legend_x + 8, legend_y + 6), 5)
        _draw_text(screen, "Prey", (legend_x + 18, legend_y - 2), font_small, theme["text_secondary"])
        _draw_triangle(screen, (legend_x + 78, legend_y + 6), 6, theme["pred"], theme["border"][:3])
        _draw_text(screen, "Predator", (legend_x + 90, legend_y - 2), font_small, theme["text_secondary"])

        # Sidebar scroll area (cards stack with scroll).
        scrollable_h = 0
        card_gap = 14
        stats_h = 180
        chart_h = 260
        params_h = 120
        scrollable_h = stats_h + chart_h + params_h + card_gap * 2
        max_scroll = max(0, scrollable_h - sidebar_rect.height + 32)
        sidebar_scroll = max(0, min(sidebar_scroll, max_scroll))

        scroll_y = sidebar_rect.y + 16 - sidebar_scroll

        # Stats card (labels left, values right).
        stats_card = pygame.Rect(sidebar_rect.x + 16, scroll_y, sidebar_rect.width - 32, stats_h)
        _draw_card(screen, stats_card, theme, radius=12, elevated=False, shadow=False)
        _draw_text(screen, "Simulation stats", (stats_card.x + 14, stats_card.y + 12), font_bold, theme["text"])

        stats_rows = [
            ("Step", str(step_count)),
            ("Prey", str(len(prey_positions))),
            ("Predators", str(len(predator_positions))),
            ("Births", str(births)),
            ("Deaths", str(deaths)),
            ("Eaten", str(meals)),
        ]
        row_y = stats_card.y + 48
        value_x = stats_card.right - 20
        for label, value in stats_rows:
            _draw_text(screen, label, (stats_card.x + 14, row_y), font_small, theme["text_secondary"])
            value_text = font_tab.render(value, True, theme["text"])
            screen.blit(value_text, (value_x - value_text.get_width(), row_y))
            row_y += 20

        # Chart card.
        chart_card = pygame.Rect(stats_card.x, stats_card.bottom + card_gap, stats_card.width, chart_h)
        _draw_card(screen, chart_card, theme, radius=12, elevated=False, shadow=False)
        _draw_text(screen, "Population over time", (chart_card.x + 14, chart_card.y + 12), font_bold, theme["text"])
        chart_inner = pygame.Rect(chart_card.x + 14, chart_card.y + 44, chart_card.width - 28, chart_card.height - 60)
        pygame.draw.rect(screen, theme["card_elev"], chart_inner, border_radius=10)

        for i in range(1, 5):
            y = chart_inner.y + int(i * chart_inner.height / 5)
            pygame.draw.line(screen, theme["divider"], (chart_inner.x, y), (chart_inner.right, y), 1)

        if prey_counts:
            max_count = max(max(prey_counts), max(predator_counts), 1)
            min_count = min(min(prey_counts), min(predator_counts), 0)
            count_range = max(1, max_count - min_count)

            def _chart_point(i, v):
                x = chart_inner.x + int(i * (chart_inner.width - 1) / max(1, len(prey_counts) - 1))
                y = chart_inner.y + chart_inner.height - int(
                    (v - min_count) * (chart_inner.height - 1) / count_range
                )
                return x, y

            prey_points = [_chart_point(i, v) for i, v in enumerate(prey_counts)]
            pred_points = [_chart_point(i, v) for i, v in enumerate(predator_counts)]

            if len(prey_points) >= 2:
                pygame.draw.aalines(screen, theme["prey"], False, prey_points, 1)
            if len(pred_points) >= 2:
                pygame.draw.aalines(screen, theme["pred"], False, pred_points, 1)

            _draw_text(screen, "Prey", (chart_inner.x, chart_inner.y - 20), font_small, theme["prey"])
            _draw_text(screen, "Predator", (chart_inner.x + 48, chart_inner.y - 20), font_small, theme["pred"])

        # Parameters card (read-only snapshot of config).
        params_card = pygame.Rect(chart_card.x, chart_card.bottom + card_gap, chart_card.width, params_h)
        _draw_card(screen, params_card, theme, radius=12, elevated=False, shadow=False)
        _draw_text(screen, "Parameters", (params_card.x + 14, params_card.y + 12), font_bold, theme["text"])
        _draw_text(screen, f"Vision: {config.VISION_RANGE}", (params_card.x + 14, params_card.y + 44), font_small, theme["text_secondary"])
        _draw_text(screen, f"Max visible: {config.MAX_VISIBLE_ANIMALS}", (params_card.x + 14, params_card.y + 64), font_small, theme["text_secondary"])
        _draw_text(screen, f"Max animals: {config.MAX_ANIMALS}", (params_card.x + 14, params_card.y + 84), font_small, theme["text_secondary"])

        # End modal (Fluent-style dialog).
        if not animals and not ended:
            ended = True
            modal_open = True

        if modal_open:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 115))
            screen.blit(overlay, (0, 0))

            modal_w, modal_h = 360, 200
            modal_rect = pygame.Rect(
                (screen_w - modal_w) // 2,
                (screen_h - modal_h) // 2,
                modal_w,
                modal_h,
            )
            _draw_card(screen, modal_rect, theme, radius=16, elevated=True, shadow=True)
            _draw_text(screen, "Simulation ended", (modal_rect.x + 24, modal_rect.y + 24), font_bold, theme["text"])
            _draw_text(screen, "All animals are dead.", (modal_rect.x + 24, modal_rect.y + 56), font_small, theme["text_secondary"])

            restart_rect = pygame.Rect(modal_rect.x + 24, modal_rect.bottom - 56, 120, 34)
            hovered = restart_rect.collidepoint(mouse_pos)
            _draw_button(screen, restart_rect, "Restart", font_small, theme, "primary", hovered, hovered and mouse_down)

            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN and restart_rect.collidepoint(event.pos):
                    reset_simulation()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run_demo()
