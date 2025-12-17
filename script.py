import pygame
import random
import math
import sys
import json
import os
import numpy as np

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

# Constants
WIDTH, HEIGHT = 400, 600
FPS = 60

# Data file path
DATA_FILE = "flappy_jellyfish_data.json"


# Simplified sound generation - only essential sounds
def generate_boss_music():
    """Generate epic boss music loop"""
    sample_rate = 22050
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)

    # Deep pulsing bass - more intense
    bass = np.sin(2 * np.pi * 55 * t) * 0.5
    bass += np.sin(2 * np.pi * 110 * t) * 0.3  # Octave up
    bass_pulse = (1 + np.sin(2 * np.pi * 4 * t)) / 2
    bass = bass * bass_pulse

    # Dramatic chord hits - louder and more frequent
    beat_pattern = np.zeros(n_samples)
    for i in range(8):
        beat_pos = int(i * n_samples / 8)
        beat_len = min(int(n_samples / 12), n_samples - beat_pos)
        if beat_len > 0:
            beat_t = np.linspace(0, beat_len / sample_rate, beat_len, False)
            # Power chord
            chord = (np.sin(2 * np.pi * 220 * beat_t) * 0.4 +
                     np.sin(2 * np.pi * 330 * beat_t) * 0.3 +
                     np.sin(2 * np.pi * 165 * beat_t) * 0.3)
            chord = chord * np.exp(-beat_t * 6)
            beat_pattern[beat_pos:beat_pos + beat_len] += chord

    # High tension synth
    tension = np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 3 * t) * 0.2

    # Driving rhythm
    kick_pattern = np.zeros(n_samples)
    for i in range(4):
        kick_pos = int(i * n_samples / 4)
        kick_len = min(int(sample_rate * 0.1), n_samples - kick_pos)
        if kick_len > 0:
            kick_t = np.linspace(0, kick_len / sample_rate, kick_len, False)
            kick = np.sin(2 * np.pi * (150 - 100 * kick_t / 0.1) * kick_t) * np.exp(-kick_t * 20) * 0.5
            kick_pattern[kick_pos:kick_pos + kick_len] += kick

    wave = (bass * 0.4 + beat_pattern * 0.5 + tension + kick_pattern) * 0.6 * 32767
    wave = np.clip(wave, -32767, 32767).astype(np.int16)
    # Convert to stereo (2D array)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


def generate_death_sound():
    """Simple death sound"""
    sample_rate = 22050
    duration = 0.5
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)

    freq = 300 - 200 * t / duration
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 4) * 0.5 * 32767
    wave = np.clip(wave, -32767, 32767).astype(np.int16)
    # Convert to stereo
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


def generate_boss_defeated_sound():
    """Epic victory fanfare"""
    sample_rate = 22050
    duration = 0.8
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)

    # Rising triumphant notes - C E G C
    wave = np.zeros(n_samples)
    notes = [523, 659, 784, 1047]
    for i, note in enumerate(notes):
        start = int(i * n_samples / 5)
        end = min(start + int(n_samples / 3), n_samples)
        if end > start:
            seg_t = np.linspace(0, (end - start) / sample_rate, end - start, False)
            # Main note + harmony
            note_wave = np.sin(2 * np.pi * note * seg_t) * 0.4
            note_wave += np.sin(2 * np.pi * note * 1.5 * seg_t) * 0.2  # Fifth
            note_wave = note_wave * np.exp(-seg_t * 3)
            wave[start:end] += note_wave

    # Add shimmer/sparkle effect
    shimmer = np.sin(2 * np.pi * 2000 * t) * np.sin(2 * np.pi * 10 * t) * 0.15 * np.exp(-t * 2)
    wave = (wave + shimmer) * 32767
    wave = np.clip(wave, -32767, 32767).astype(np.int16)
    # Convert to stereo
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


# Initialize sounds
try:
    SOUND_ENABLED = True
    sound_death = generate_death_sound()
    sound_boss_defeated = generate_boss_defeated_sound()
    music_boss = generate_boss_music()

    sound_death.set_volume(0.5)
    sound_boss_defeated.set_volume(0.7)
    music_boss.set_volume(0.6)
except Exception as e:
    print(f"Sound initialization failed: {e}")
    SOUND_ENABLED = False


def load_game_data():
    """Load leaderboard and stats from file"""
    default_data = {
        "leaderboard": [],
        "stats": {
            "games_played": 0,
            "total_score": 0,
            "bosses_defeated": 0,
            "highest_boss": 0,
            "best_score": 0,
            "total_time_played": 0
        },
        "player": {
            "username": "",
            "region": ""
        }
    }
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                # Ensure player key exists for older save files
                if "player" not in data:
                    data["player"] = default_data["player"]
                return data
    except:
        pass
    return default_data


def save_game_data(data):
    """Save leaderboard and stats to file"""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass


def add_to_leaderboard(data, score, bosses_defeated, username, region):
    """Add a new score to leaderboard"""
    entry = {
        "score": score,
        "bosses": bosses_defeated,
        "username": username,
        "region": region
    }
    data["leaderboard"].append(entry)
    # Sort by score descending and keep top 10
    data["leaderboard"] = sorted(data["leaderboard"], key=lambda x: x["score"], reverse=True)[:10]
    return data


def update_stats(data, score, bosses_defeated, time_played):
    """Update game statistics"""
    data["stats"]["games_played"] += 1
    data["stats"]["total_score"] += score
    data["stats"]["bosses_defeated"] += bosses_defeated
    data["stats"]["highest_boss"] = max(data["stats"]["highest_boss"], bosses_defeated)
    data["stats"]["best_score"] = max(data["stats"]["best_score"], score)
    data["stats"]["total_time_played"] += time_played
    return data


# Colors
OCEAN_DEEP = (10, 30, 80)
OCEAN_MID = (20, 60, 120)
OCEAN_LIGHT = (40, 100, 160)
WHITE = (255, 255, 255)
CORAL_PINK = (255, 127, 127)
CORAL_ORANGE = (255, 160, 100)
SEAWEED_GREEN = (50, 150, 100)
JELLYFISH_PURPLE = (200, 150, 255)
BUBBLE_BLUE = (150, 200, 255)
BUTTON_COLOR = (100, 200, 255)
BUTTON_HOVER = (150, 230, 255)
TURTLE_GREEN = (100, 180, 100)
BOSS_RED = (255, 80, 80)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Jellyfish 🪼")
clock = pygame.time.Clock()


class Particle:
    def __init__(self, x, y, color, size, speed_x, speed_y, lifetime):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.lifetime = lifetime
        self.age = 0
        self.alpha = 255

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.age += 1
        self.alpha = int(255 * (1 - self.age / self.lifetime))
        return self.age < self.lifetime

    def draw(self, surface):
        if self.alpha > 0:
            s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.alpha)
            pygame.draw.circle(s, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))


class Projectile:
    def __init__(self, x, y, target_y, speed=4, homing=1.0):
        self.x = x
        self.y = y
        self.target_y = target_y
        self.speed = speed
        self.radius = 8
        self.glow = 0
        self.homing = homing  # How strongly it homes in on target

    def update(self):
        self.x -= self.speed
        # Slight homing towards target
        diff = self.target_y - self.y
        if abs(diff) > 1:
            self.y += max(-self.homing, min(self.homing, diff * 0.05 * self.homing))
        self.glow += 0.2

    def draw(self, surface):
        # Outer glow
        glow_size = int(self.radius + abs(math.sin(self.glow)) * 5)
        s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 100, 100, 100), (glow_size, glow_size), glow_size)
        surface.blit(s, (int(self.x - glow_size), int(self.y - glow_size)))

        # Core projectile
        pygame.draw.circle(surface, (255, 200, 50), (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, (255, 100, 50), (int(self.x), int(self.y)), self.radius - 3)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)


class Boss:
    def __init__(self, difficulty=1):
        self.x = WIDTH + 100
        self.y = HEIGHT // 2
        self.size = 50 + difficulty * 5  # Bigger each round
        self.speed = 2 + difficulty * 0.5  # Faster movement
        self.entering = True
        self.shoot_timer = 0
        # Shoots much faster each round: 90 -> 75 -> 60 -> 50 -> 42 -> 36 -> 31 -> 27 -> 24 -> 21
        self.shoot_interval = max(20, int(90 / (1 + difficulty * 0.2)))
        self.health = 3 + difficulty  # More health each round
        self.max_health = self.health
        self.flash = 0
        self.move_timer = 0
        self.target_y = self.y
        self.tentacles_wave = 0
        self.difficulty = difficulty
        self.fight_timer = 0  # Timer for 15 second survival
        self.fight_duration = 15 * FPS  # 15 seconds to survive

        # Multi-shot: bosses shoot more projectiles at higher levels
        self.projectiles_per_shot = 1 + (difficulty // 3)  # 1, 1, 1, 2, 2, 2, 3, 3, 3, 4

        # Movement becomes more erratic at higher levels
        self.move_change_interval = max(40, 120 - difficulty * 10)

        # Projectile speed increases with difficulty
        self.projectile_speed = 4 + difficulty * 0.8

    def update(self, jellyfish_y):
        self.tentacles_wave += 0.15

        if self.entering:
            self.x -= self.speed
            if self.x <= WIDTH - 80:
                self.entering = False
        else:
            # Increment fight timer only when fully entered
            self.fight_timer += 1

            # Boss movement pattern - more erratic at higher difficulties
            self.move_timer += 1
            if self.move_timer > self.move_change_interval:
                self.target_y = random.randint(80, HEIGHT - 80)
                self.move_timer = 0

            # Smooth movement to target - faster at higher difficulties
            move_speed = 0.02 + self.difficulty * 0.01
            if abs(self.y - self.target_y) > 2:
                self.y += (self.target_y - self.y) * move_speed

        self.shoot_timer += 1
        self.flash = max(0, self.flash - 5)

    def is_defeated(self):
        return self.fight_timer >= self.fight_duration

    def shoot(self, jellyfish_y):
        if self.shoot_timer >= self.shoot_interval and not self.entering:
            self.shoot_timer = 0
            projectiles = []

            # Homing strength increases with difficulty
            homing = 1.0 + self.difficulty * 0.3

            # Calculate spread for multiple projectiles
            if self.projectiles_per_shot == 1:
                # Single shot aimed at jellyfish
                projectiles.append(Projectile(self.x - self.size, self.y, jellyfish_y, self.projectile_speed, homing))
            else:
                # Multiple shots with spread pattern
                spread = 80  # Total spread angle in pixels
                for i in range(self.projectiles_per_shot):
                    offset = (i - (self.projectiles_per_shot - 1) / 2) * (
                                spread / max(1, self.projectiles_per_shot - 1))
                    target_y = jellyfish_y + offset
                    projectiles.append(Projectile(self.x - self.size, self.y, target_y, self.projectile_speed, homing))

            return projectiles
        return None

    def draw(self, surface):
        # Boss body (evil octopus)
        flash_color = 255 if self.flash > 0 else 0
        body_color = (BOSS_RED[0], BOSS_RED[1] + flash_color, BOSS_RED[2] + flash_color)

        # Main body
        pygame.draw.circle(surface, body_color, (int(self.x), int(self.y)), self.size)

        # Darker shade for depth
        pygame.draw.circle(surface, (200, 50, 50), (int(self.x), int(self.y)), self.size - 5)

        # Tentacles
        num_tentacles = 8
        for i in range(num_tentacles):
            angle = (i / num_tentacles) * math.pi * 2
            wave = math.sin(self.tentacles_wave + i) * 10

            start_x = self.x + math.cos(angle) * self.size
            start_y = self.y + math.sin(angle) * self.size

            points = [(start_x, start_y)]
            for j in range(4):
                segment_x = start_x + math.cos(angle) * (j * 15 + wave)
                segment_y = start_y + math.sin(angle) * (j * 15) + math.sin(self.tentacles_wave + j) * 5
                points.append((segment_x, segment_y))

            pygame.draw.lines(surface, (180, 60, 60), False, points, 4)

        # Evil eyes
        eye_offset = 15
        # Left eye
        pygame.draw.circle(surface, (255, 255, 0), (int(self.x - eye_offset), int(self.y - 10)), 10)
        pygame.draw.circle(surface, (255, 0, 0), (int(self.x - eye_offset + 3), int(self.y - 10)), 6)
        # Right eye
        pygame.draw.circle(surface, (255, 255, 0), (int(self.x + eye_offset), int(self.y - 10)), 10)
        pygame.draw.circle(surface, (255, 0, 0), (int(self.x + eye_offset + 3), int(self.y - 10)), 6)

        # Warning indicator when about to shoot
        if self.shoot_timer > self.shoot_interval - 30 and not self.entering:
            warning_alpha = int(128 + 127 * math.sin(self.shoot_timer * 0.3))
            s = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 0, warning_alpha), (10, 10), 8)
            surface.blit(s, (int(self.x - self.size - 15), int(self.y - 10)))

        # Survival timer bar (instead of health)
        bar_width = 60
        bar_height = 8
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.size - 20

        # Background
        pygame.draw.rect(surface, (100, 100, 100),
                         (bar_x, bar_y, bar_width, bar_height))
        # Time remaining (green bar that depletes)
        time_remaining = max(0, 1 - self.fight_timer / self.fight_duration)
        time_width = time_remaining * bar_width
        pygame.draw.rect(surface, (50, 255, 50),
                         (bar_x, bar_y, time_width, bar_height))

    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size,
                           self.size * 2, self.size * 2)

    def hit(self):
        self.health -= 1
        self.flash = 255
        return self.health <= 0


class Turtle:
    def __init__(self):
        self.x = random.choice([WIDTH + 20, -20])
        self.y = random.randint(100, HEIGHT - 100)
        self.speed = random.uniform(0.5, 1.2)
        self.direction = -1 if self.x > WIDTH // 2 else 1
        self.size = 25
        self.swim_wave = random.uniform(0, math.pi * 2)
        self.swim_speed = random.uniform(0.05, 0.1)
        self.shell_rotation = 0

    def update(self):
        self.x += self.speed * self.direction
        self.swim_wave += self.swim_speed
        self.y += math.sin(self.swim_wave) * 0.3
        self.shell_rotation += 0.02

        # Reset when off screen
        if self.direction > 0 and self.x > WIDTH + 50:
            self.x = -20
            self.y = random.randint(100, HEIGHT - 100)
        elif self.direction < 0 and self.x < -50:
            self.x = WIDTH + 20
            self.y = random.randint(100, HEIGHT - 100)

    def draw(self, surface):
        # Shell (hexagonal pattern)
        shell_color = TURTLE_GREEN
        darker_green = (70, 130, 70)

        # Main shell
        pygame.draw.circle(surface, shell_color, (int(self.x), int(self.y)), self.size)

        # Shell pattern
        for i in range(6):
            angle = i * math.pi / 3 + self.shell_rotation
            pattern_x = self.x + math.cos(angle) * (self.size // 2)
            pattern_y = self.y + math.sin(angle) * (self.size // 2)
            pygame.draw.circle(surface, darker_green, (int(pattern_x), int(pattern_y)), 6)

        # Head
        head_x_offset = -self.size if self.direction < 0 else self.size
        head_x = self.x + head_x_offset
        pygame.draw.circle(surface, (120, 200, 120), (int(head_x), int(self.y)), 12)

        # Eye
        eye_x_offset = -5 if self.direction < 0 else 5
        pygame.draw.circle(surface, (0, 0, 0), (int(head_x + eye_x_offset), int(self.y - 3)), 3)

        # Flippers
        flipper_wave = math.sin(self.swim_wave * 2) * 8

        # Front flipper
        front_flipper_y = self.y + flipper_wave
        pygame.draw.ellipse(surface, (90, 160, 90),
                            (int(head_x - 10), int(front_flipper_y), 15, 8))

        # Back flipper
        back_flipper_x = self.x - head_x_offset * 0.5
        back_flipper_y = self.y - flipper_wave
        pygame.draw.ellipse(surface, (90, 160, 90),
                            (int(back_flipper_x - 10), int(back_flipper_y), 15, 8))


class Fish:
    def __init__(self):
        self.x = random.choice([WIDTH + 20, -20])
        self.y = random.randint(50, HEIGHT - 50)
        self.speed = random.uniform(1, 2.5)
        self.direction = -1 if self.x > WIDTH // 2 else 1
        self.size = random.randint(15, 30)
        self.swim_wave = random.uniform(0, math.pi * 2)
        self.swim_speed = random.uniform(0.1, 0.2)
        self.color = random.choice([
            (255, 180, 50),
            (100, 200, 255),
            (255, 100, 150),
            (150, 255, 150),
            (200, 150, 255),
        ])

    def update(self):
        self.x += self.speed * self.direction
        self.swim_wave += self.swim_speed
        self.y += math.sin(self.swim_wave) * 0.5

        if self.direction > 0 and self.x > WIDTH + 50:
            self.x = -20
            self.y = random.randint(50, HEIGHT - 50)
        elif self.direction < 0 and self.x < -50:
            self.x = WIDTH + 20
            self.y = random.randint(50, HEIGHT - 50)

    def draw(self, surface):
        body_points = []
        if self.direction > 0:
            body_points = [
                (self.x - self.size, self.y),
                (self.x, self.y - self.size // 2),
                (self.x + self.size, self.y),
                (self.x, self.y + self.size // 2)
            ]
            tail_wave = math.sin(self.swim_wave * 2) * 5
            tail_points = [
                (self.x - self.size, self.y),
                (self.x - self.size - 10, self.y + tail_wave - 8),
                (self.x - self.size - 10, self.y + tail_wave + 8)
            ]
        else:
            body_points = [
                (self.x + self.size, self.y),
                (self.x, self.y - self.size // 2),
                (self.x - self.size, self.y),
                (self.x, self.y + self.size // 2)
            ]
            tail_wave = math.sin(self.swim_wave * 2) * 5
            tail_points = [
                (self.x + self.size, self.y),
                (self.x + self.size + 10, self.y + tail_wave - 8),
                (self.x + self.size + 10, self.y + tail_wave + 8)
            ]

        pygame.draw.polygon(surface, self.color, tail_points)
        pygame.draw.polygon(surface, self.color, body_points)

        eye_x = self.x + (self.size // 3 if self.direction > 0 else -self.size // 3)
        pygame.draw.circle(surface, WHITE, (int(eye_x), int(self.y - 5)), 4)
        pygame.draw.circle(surface, (0, 0, 0), (int(eye_x), int(self.y - 5)), 2)


class Bubble:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = HEIGHT + random.randint(0, 100)
        self.radius = random.randint(3, 8)
        self.speed = random.uniform(0.5, 1.5)
        self.wobble = random.uniform(0, 2 * math.pi)
        self.wobble_speed = random.uniform(0.02, 0.05)

    def update(self):
        self.y -= self.speed
        self.wobble += self.wobble_speed
        self.x += math.sin(self.wobble) * 0.5

        if self.y < -20:
            self.y = HEIGHT + 20
            self.x = random.randint(0, WIDTH)

    def draw(self, surface):
        pygame.draw.circle(surface, BUBBLE_BLUE, (int(self.x), int(self.y)), self.radius, 1)
        highlight_x = int(self.x - self.radius // 3)
        highlight_y = int(self.y - self.radius // 3)
        pygame.draw.circle(surface, WHITE, (highlight_x, highlight_y), self.radius // 3)


class LightRay:
    def __init__(self):
        self.x = random.randint(-50, WIDTH)
        self.width = random.randint(30, 80)
        self.speed = random.uniform(0.2, 0.5)
        self.alpha = random.randint(20, 50)

    def update(self):
        self.x += self.speed
        if self.x > WIDTH + 50:
            self.x = -50

    def draw(self, surface):
        s = pygame.Surface((self.width, HEIGHT), pygame.SRCALPHA)
        for i in range(self.width):
            alpha = int(self.alpha * (1 - abs(i - self.width / 2) / (self.width / 2)))
            color = (255, 255, 200, alpha)
            pygame.draw.line(s, color, (i, 0), (i, HEIGHT))
        surface.blit(s, (int(self.x), 0))


class Jellyfish:
    def __init__(self):
        self.x = 100
        self.y = HEIGHT // 2
        self.velocity = 0
        self.radius = 20
        self.tentacle_wave = 0
        self.body_pulse = 0

    def flap(self):
        self.velocity = -7

    def update(self):
        self.velocity += 0.5
        self.y += self.velocity
        self.tentacle_wave += 0.2
        self.body_pulse += 0.15

    def draw(self, surface, particles):
        pulse = math.sin(self.body_pulse) * 2
        bell_radius = self.radius + pulse

        for r in range(int(bell_radius), 0, -2):
            alpha = int(180 * (r / bell_radius))
            color = (*JELLYFISH_PURPLE, alpha)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (r, r), r)
            surface.blit(s, (int(self.x - r), int(self.y - r)))

        num_tentacles = 6
        for i in range(num_tentacles):
            angle_offset = (i / num_tentacles) * math.pi * 2
            wave_offset = math.sin(self.tentacle_wave + angle_offset) * 5

            start_x = self.x + math.cos(angle_offset) * 10
            start_y = self.y + 5

            points = [(start_x, start_y)]
            for j in range(5):
                segment_x = start_x + wave_offset * j * 0.3
                segment_y = start_y + j * 8 + math.sin(self.tentacle_wave + j) * 3
                points.append((segment_x, segment_y))

            pygame.draw.lines(surface, (180, 120, 220), False, points, 2)

        if self.velocity < 0 and random.random() > 0.7:
            particles.append(Particle(
                self.x + random.randint(-10, 10),
                self.y + 20,
                (150, 180, 255),
                random.randint(2, 4),
                random.uniform(-0.5, 0.5),
                random.uniform(0.5, 1.5),
                30
            ))

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)


class Obstacle:
    def __init__(self, x, speed=3):
        self.x = x
        self.gap = random.randint(150, 200)
        self.top_height = random.randint(100, HEIGHT - self.gap - 100)
        self.width = 60
        self.speed = speed
        self.scored = False
        self.sway = 0

    def update(self):
        self.x -= self.speed
        self.sway += 0.05

    def draw(self, surface):
        self.draw_seaweed(surface, self.x, 0, self.top_height, True)
        bottom_start = self.top_height + self.gap
        self.draw_seaweed(surface, self.x, bottom_start, HEIGHT - bottom_start, False)

    def draw_seaweed(self, surface, x, start_y, height, from_top):
        sway_offset = math.sin(self.sway) * 5

        if from_top:
            for i in range(0, int(height), 10):
                width_var = 30 + math.sin(i * 0.1) * 10
                color = CORAL_PINK if i % 20 < 10 else CORAL_ORANGE
                rect = pygame.Rect(x + sway_offset + 15 - width_var / 2, start_y + i, width_var, 10)
                pygame.draw.rect(surface, color, rect, border_radius=5)
        else:
            for i in range(0, int(height), 10):
                width_var = 25 + math.sin(i * 0.15) * 8
                color = SEAWEED_GREEN
                sway_i = math.sin(self.sway + i * 0.1) * (i * 0.1)
                rect = pygame.Rect(x + sway_i + 15 - width_var / 2, start_y + i, width_var, 10)
                pygame.draw.ellipse(surface, color, rect)

    def collides_with(self, jellyfish):
        jelly_rect = jellyfish.get_rect()

        if jelly_rect.colliderect(pygame.Rect(self.x, 0, self.width, self.top_height)):
            return True

        bottom_y = self.top_height + self.gap
        if jelly_rect.colliderect(pygame.Rect(self.x, bottom_y, self.width, HEIGHT - bottom_y)):
            return True

        return False


class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
        self.text = text
        self.hovered = False

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface):
        color = BUTTON_HOVER if self.hovered else BUTTON_COLOR

        shadow_rect = self.rect.copy()
        shadow_rect.y += 4
        pygame.draw.rect(surface, (0, 50, 100), shadow_rect, border_radius=10)
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, 3, border_radius=10)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos, mouse_pressed):
        return self.rect.collidepoint(mouse_pos) and mouse_pressed


class InputBox:
    def __init__(self, x, y, width, height, label="", max_length=15):
        self.rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
        self.text = ""
        self.label = label
        self.active = False
        self.max_length = max_length
        self.cursor_visible = True
        self.cursor_timer = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            elif len(self.text) < self.max_length:
                # Only allow alphanumeric and some special chars
                if event.unicode.isalnum() or event.unicode in " _-":
                    self.text += event.unicode

    def update(self):
        self.cursor_timer += 1
        if self.cursor_timer > 30:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def draw(self, surface):
        # Label
        font = pygame.font.Font(None, 28)
        label_surface = font.render(self.label, True, (200, 220, 255))
        label_rect = label_surface.get_rect(midbottom=(self.rect.centerx, self.rect.top - 5))
        surface.blit(label_surface, label_rect)

        # Box
        color = (100, 200, 255) if self.active else (70, 130, 180)
        pygame.draw.rect(surface, (20, 40, 80), self.rect, border_radius=8)
        pygame.draw.rect(surface, color, self.rect, 3, border_radius=8)

        # Text
        font = pygame.font.Font(None, 32)
        display_text = self.text
        if self.active and self.cursor_visible:
            display_text += "|"
        text_surface = font.render(display_text, True, WHITE)
        text_rect = text_surface.get_rect(midleft=(self.rect.left + 10, self.rect.centery))
        surface.blit(text_surface, text_rect)


def draw_background(surface, bubbles, light_rays, fish_list, turtles):
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(OCEAN_DEEP[0] + (OCEAN_LIGHT[0] - OCEAN_DEEP[0]) * ratio)
        g = int(OCEAN_DEEP[1] + (OCEAN_LIGHT[1] - OCEAN_DEEP[1]) * ratio)
        b = int(OCEAN_DEEP[2] + (OCEAN_LIGHT[2] - OCEAN_DEEP[2]) * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))

    for ray in light_rays:
        ray.update()
        ray.draw(surface)

    for fish in fish_list:
        fish.update()
        fish.draw(surface)

    for turtle in turtles:
        turtle.update()
        turtle.draw(surface)

    for bubble in bubbles:
        bubble.update()
        bubble.draw(surface)


def draw_text(surface, text, size, x, y, color=WHITE):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    surface.blit(text_surface, text_rect)


def draw_intro(screen, frame, bubbles, light_rays, fish_list, turtles):
    """Draw animated intro sequence"""
    # Don't call draw_background here since it's already called in main loop

    # Overlay for darker effect
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 30, 150))
    screen.blit(overlay, (0, 0))

    # Calculate animation phases
    phase = frame // 120  # Change phase every 2 seconds
    local_frame = frame % 120
    fade_in = min(1, local_frame / 30)  # Fade in over 0.5 seconds

    if phase == 0:
        # Phase 1: Title appears
        title_y = 200 + math.sin(frame * 0.05) * 10

        # Shadow
        draw_text(screen, "Flappy Jellyfish", 52, WIDTH // 2 + 3, int(title_y) + 3, (50, 30, 80))
        # Main title
        draw_text(screen, "Flappy Jellyfish", 52, WIDTH // 2, int(title_y), (255, 220, 255))

        # Subtitle
        if local_frame > 40:
            draw_text(screen, "An Ocean Adventure", 28, WIDTH // 2, int(title_y) + 60, (150, 200, 255))

        # Animated jellyfish
        draw_mini_jellyfish(screen, WIDTH // 2, int(title_y) + 130, frame)

    elif phase == 1:
        # Phase 2: Story intro
        draw_text(screen, "Flappy Jellyfish", 42, WIDTH // 2, 80, (255, 220, 255))

        lines = [
            "Deep in the ocean...",
            "A brave little jellyfish",
            "begins an epic journey!"
        ]
        for i, line in enumerate(lines):
            if local_frame > i * 30:
                draw_text(screen, line, 26, WIDTH // 2, 180 + i * 45, (200, 230, 255))

        # Animated jellyfish swimming across
        jelly_x = -50 + (local_frame) * 4
        jelly_y = 380 + math.sin(frame * 0.1) * 20
        draw_mini_jellyfish(screen, jelly_x, jelly_y, frame)

    elif phase == 2:
        # Phase 3: Dangers
        draw_text(screen, "But beware!", 42, WIDTH // 2, 100, (255, 150, 150))

        dangers = [
            ("Coral Reefs", (255, 127, 127), 180),
            ("Seaweed", (50, 200, 100), 250),
            ("BOSS Octopus!", (255, 80, 80), 320),
        ]

        for i, (text, color, y) in enumerate(dangers):
            if local_frame > i * 30:
                shake = math.sin(frame * 0.3) * 3 if i == 2 else 0
                # Draw indicator circle
                pygame.draw.circle(screen, color, (WIDTH // 2 - 100, y), 12)
                draw_text(screen, text, 28, WIDTH // 2 + 20 + int(shake), y, color)

        # Warning for boss
        if local_frame > 90 and frame % 30 < 15:
            draw_text(screen, "WARNING!", 24, WIDTH // 2, 380, (255, 255, 0))

    elif phase == 3:
        # Phase 4: How to play
        draw_text(screen, "How to Play", 42, WIDTH // 2, 80, (100, 255, 200))

        instructions = [
            ("CLICK", "to swim up", (255, 255, 150)),
            ("DODGE", "obstacles", (255, 200, 150)),
            ("SURVIVE", "15s vs Boss", (150, 255, 150)),
            ("SCORE 30", "= Boss fight!", (255, 150, 150)),
        ]

        for i, (action, desc, color) in enumerate(instructions):
            if local_frame > i * 20:
                y = 160 + i * 60
                draw_text(screen, action, 30, WIDTH // 2 - 50, y, color)
                draw_text(screen, desc, 24, WIDTH // 2 + 70, y, (200, 200, 255))

        # Pulsing "Get Ready!"
        if local_frame > 80:
            pulse = 1 + math.sin(frame * 0.2) * 0.1
            size = int(38 * pulse)
            draw_text(screen, "Get Ready!", size, WIDTH // 2, 450, (255, 200, 100))

    elif phase >= 4:
        # Intro finished
        return True

    # Progress dots
    for i in range(4):
        color = (255, 255, 255) if i <= phase else (100, 100, 150)
        pygame.draw.circle(screen, color, (WIDTH // 2 - 30 + i * 20, HEIGHT - 40), 6 if i == phase else 4)

    # Skip hint
    draw_text(screen, "Click to skip", 20, WIDTH // 2, HEIGHT - 70, (150, 150, 200))

    return False


def draw_mini_jellyfish(surface, x, y, frame):
    """Draw a small animated jellyfish for intro"""
    pulse = math.sin(frame * 0.15) * 3
    radius = 18 + pulse

    # Bell (solid color, simpler)
    pygame.draw.circle(surface, (200, 150, 255), (int(x), int(y)), int(radius))
    pygame.draw.circle(surface, (180, 130, 235), (int(x), int(y)), int(radius - 4))

    # Highlight
    pygame.draw.circle(surface, (230, 200, 255), (int(x - 5), int(y - 5)), 5)

    # Tentacles
    for i in range(5):
        wave = math.sin(frame * 0.2 + i) * 6
        start_x = x + (i - 2) * 7
        points = [(start_x, y + radius - 5)]
        for j in range(5):
            px = start_x + wave * (j * 0.3)
            py = y + radius - 5 + j * 7
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(surface, (180, 120, 220), False, points, 2)


def main():
    jellyfish = Jellyfish()
    obstacles = []
    particles = []
    projectiles = []
    bubbles = [Bubble() for _ in range(20)]
    light_rays = [LightRay() for _ in range(3)]
    fish_list = [Fish() for _ in range(8)]
    turtles = [Turtle() for _ in range(4)]

    boss = None
    boss_defeated = False
    boss_round = 0
    max_boss_rounds = 10
    boss_spawn_score = 30  # Boss spawns every 30 score
    last_boss_spawn_score = 0  # Track when last boss spawned

    score = 0
    high_score = 0
    game_state = "intro"  # intro, menu, playing, gameover, leaderboard, stats, profile
    current_speed = 3  # Base obstacle speed
    intro_frame = 0  # Frame counter for intro animation

    # Music state
    music_channel = None
    boss_music_playing = False
    if SOUND_ENABLED:
        music_channel = pygame.mixer.Channel(0)

    def play_boss_music():
        nonlocal boss_music_playing
        if SOUND_ENABLED and music_channel:
            music_channel.play(music_boss, loops=-1)
            boss_music_playing = True

    def stop_music():
        nonlocal boss_music_playing
        if music_channel:
            music_channel.stop()
        boss_music_playing = False

    # Load saved data
    game_data = load_game_data()
    high_score = game_data["stats"]["best_score"]

    # Player info
    player_username = game_data["player"].get("username", "")
    player_region = game_data["player"].get("region", "")

    # Buttons
    start_button = Button(WIDTH // 2, 300, 200, 50, "START")
    leaderboard_button = Button(WIDTH // 2, 360, 200, 50, "LEADERBOARD")
    stats_button = Button(WIDTH // 2, 420, 200, 50, "STATS")
    profile_button = Button(WIDTH // 2, 480, 200, 50, "PROFILE")
    back_button = Button(WIDTH // 2, 520, 150, 45, "BACK")
    save_button = Button(WIDTH // 2, 420, 150, 50, "SAVE")

    # Input boxes for profile
    username_input = InputBox(WIDTH // 2, 220, 250, 45, "Username", 12)
    region_input = InputBox(WIDTH // 2, 320, 250, 45, "Region", 15)
    username_input.text = player_username
    region_input.text = player_region

    spawn_timer = 0
    spawn_interval = 90
    game_timer = 0
    session_start_time = 0
    stats_saved = False  # Flag to prevent saving stats multiple times

    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
                if game_state == "intro":
                    game_state = "menu"  # Skip intro on click
                elif game_state == "playing":
                    jellyfish.flap()
                elif game_state == "gameover":
                    game_state = "menu"
                    stop_music()
                elif game_state == "leaderboard" or game_state == "stats":
                    pass  # Handle with button clicks below

            # Handle input box events in profile screen
            if game_state == "profile":
                username_input.handle_event(event)
                region_input.handle_event(event)

        draw_background(screen, bubbles, light_rays, fish_list, turtles)

        if game_state == "intro":
            intro_frame += 1
            intro_finished = draw_intro(screen, intro_frame, bubbles, light_rays, fish_list, turtles)
            if intro_finished:
                game_state = "menu"

        elif game_state == "menu":
            draw_text(screen, "Flappy Jellyfish", 56, WIDTH // 2, 60, (255, 220, 255))
            draw_text(screen, "Swim through the coral reefs!", 22, WIDTH // 2, 102, (50, 100, 150))
            draw_text(screen, "Swim through the coral reefs!", 22, WIDTH // 2, 100, (200, 240, 255))

            demo_jelly = Jellyfish()
            demo_jelly.x = WIDTH // 2
            demo_jelly.y = 180
            demo_jelly.tentacle_wave += 0.2
            demo_jelly.body_pulse += 0.15
            demo_jelly.draw(screen, particles)

            particles = [p for p in particles if p.update()]
            for p in particles:
                p.draw(screen)

            # Show player info and best score
            if player_username:
                draw_text(screen, f"Player: {player_username}", 22, WIDTH // 2, 240, (150, 200, 255))
            if high_score > 0:
                draw_text(screen, f"Best: {high_score}", 26, WIDTH // 2, 265, (255, 215, 0))

            start_button.update(mouse_pos)
            start_button.draw(screen)
            leaderboard_button.update(mouse_pos)
            leaderboard_button.draw(screen)
            stats_button.update(mouse_pos)
            stats_button.draw(screen)
            profile_button.update(mouse_pos)
            profile_button.draw(screen)

            if start_button.is_clicked(mouse_pos, mouse_pressed):
                # Check if player has set username
                if not player_username:
                    game_state = "profile"
                else:
                    game_state = "playing"
                    obstacles = []
                    projectiles = []
                    boss = None
                    boss_defeated = False
                    boss_round = 0
                    last_boss_spawn_score = 0
                    score = 0
                    game_timer = 0
                    current_speed = 3  # Reset speed
                    session_start_time = pygame.time.get_ticks()
                    jellyfish = Jellyfish()
                    particles = []
                    stats_saved = False  # Reset stats saved flag
                    stop_music()  # Ensure music is stopped

            if leaderboard_button.is_clicked(mouse_pos, mouse_pressed):
                game_state = "leaderboard"

            if stats_button.is_clicked(mouse_pos, mouse_pressed):
                game_state = "stats"

            if profile_button.is_clicked(mouse_pos, mouse_pressed):
                game_state = "profile"

            draw_text(screen, "Click to swim upward", 18, WIDTH // 2, 530, (180, 220, 255))
            draw_text(screen, "Survive 15s vs BOSS every 30 pts!", 16, WIDTH // 2, 555, (255, 150, 150))

        elif game_state == "playing":
            jellyfish.update()
            game_timer += 1  # Increment game timer each frame

            # Boss spawn every 30 score, up to 10 bosses
            next_boss_score = (boss_round + 1) * boss_spawn_score
            if score >= next_boss_score and boss is None and boss_round < max_boss_rounds:
                boss_round += 1
                boss = Boss(boss_round)  # Pass difficulty level
                last_boss_spawn_score = score
                obstacles.clear()

                # Play epic boss music
                if SOUND_ENABLED:
                    play_boss_music()

                # Show boss intro with difficulty info
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                screen.blit(overlay, (0, 0))
                draw_text(screen, f"BOSS {boss_round}/{max_boss_rounds}!", 48, WIDTH // 2, HEIGHT // 2 - 40,
                          (255, 100, 100))

                # Difficulty description
                if boss_round <= 3:
                    diff_text = "Easy"
                    diff_color = (100, 255, 100)
                elif boss_round <= 6:
                    diff_text = "Medium"
                    diff_color = (255, 255, 100)
                elif boss_round <= 8:
                    diff_text = "Hard"
                    diff_color = (255, 150, 50)
                else:
                    diff_text = "EXTREME!"
                    diff_color = (255, 50, 50)

                draw_text(screen, f"Difficulty: {diff_text}", 28, WIDTH // 2, HEIGHT // 2 + 10, diff_color)
                draw_text(screen, f"Shots: {boss.projectiles_per_shot} | Speed: {boss.projectile_speed:.1f}", 20,
                          WIDTH // 2, HEIGHT // 2 + 45, (200, 200, 255))

                pygame.display.flip()
                pygame.time.wait(1500)

            # Boss logic
            if boss:
                boss.update(jellyfish.y)
                new_projectiles = boss.shoot(jellyfish.y)
                if new_projectiles:
                    projectiles.extend(new_projectiles)

                # Check collision with boss
                if boss.get_rect().colliderect(jellyfish.get_rect()):
                    game_state = "gameover"
                    high_score = max(score, high_score)
                    if SOUND_ENABLED:
                        sound_death.play()
                        stop_music()

            # Update projectiles
            for proj in projectiles[:]:
                proj.update()
                if proj.x < -20:
                    projectiles.remove(proj)
                elif proj.get_rect().colliderect(jellyfish.get_rect()):
                    game_state = "gameover"
                    high_score = max(score, high_score)
                    if SOUND_ENABLED:
                        sound_death.play()
                        stop_music()

            # Normal obstacles (only spawn if no boss)
            if not boss:
                spawn_timer += 1
                if spawn_timer >= spawn_interval:
                    # Calculate speed: increases by 0.3 every 10 score, max speed 6
                    current_speed = min(6, 3 + (score // 10) * 0.3)
                    obstacles.append(Obstacle(WIDTH, current_speed))
                    spawn_timer = 0

            for obs in obstacles[:]:
                obs.update()

                if obs.x < -obs.width:
                    obstacles.remove(obs)

                if obs.collides_with(jellyfish):
                    game_state = "gameover"
                    high_score = max(score, high_score)
                    if SOUND_ENABLED:
                        sound_death.play()
                        stop_music()

                if not obs.scored and obs.x + obs.width < jellyfish.x:
                    obs.scored = True
                    score += 1

            # Check if boss is defeated (survived 15 seconds)
            if boss and boss.is_defeated():
                boss = None
                projectiles.clear()  # Clear projectiles when boss defeated
                score += 10  # Bonus for defeating boss
                if SOUND_ENABLED:
                    sound_boss_defeated.play()
                    stop_music()  # Stop boss music

            if jellyfish.y < 0 or jellyfish.y > HEIGHT:
                game_state = "gameover"
                high_score = max(score, high_score)
                if SOUND_ENABLED:
                    sound_death.play()
                    stop_music()

            # Draw obstacles
            for obs in obstacles:
                obs.draw(screen)

            # Draw projectiles
            for proj in projectiles:
                proj.draw(screen)

            # Draw boss
            if boss:
                boss.draw(screen)

            particles = [p for p in particles if p.update()]
            for p in particles:
                p.draw(screen)

            jellyfish.draw(screen, particles)

            # Boss warning (show when close to next boss spawn score)
            next_boss_score = (boss_round + 1) * boss_spawn_score
            if score >= next_boss_score - 5 and score < next_boss_score and not boss and boss_round < max_boss_rounds:
                draw_text(screen, f"BOSS {boss_round + 1} APPROACHING!", 32, WIDTH // 2, HEIGHT - 50, (255, 200, 0))

            # Display boss round during boss fight with survival timer
            if boss:
                draw_text(screen, f"BOSS {boss_round}/{max_boss_rounds}", 24, WIDTH // 2, 70, (255, 100, 100))
                # Show seconds remaining
                seconds_left = max(0, (boss.fight_duration - boss.fight_timer) // FPS)
                draw_text(screen, f"Survive: {seconds_left}s", 28, WIDTH // 2, 100, (50, 255, 50))

            # Victory condition - all bosses defeated (optional - removed so game continues)
            # Game no longer ends after defeating all bosses

            draw_text(screen, f"Score: {score}", 38, WIDTH // 2 + 2, 42, (0, 30, 60))
            draw_text(screen, f"Score: {score}", 38, WIDTH // 2, 40, WHITE)

        elif game_state == "gameover":
            # Save stats only once when entering gameover
            if not stats_saved:
                time_played = (pygame.time.get_ticks() - session_start_time) // 1000
                game_data = update_stats(game_data, score, boss_round, time_played)
                game_data = add_to_leaderboard(game_data, score, boss_round, player_username, player_region)
                save_game_data(game_data)
                high_score = game_data["stats"]["best_score"]
                stats_saved = True

            for obs in obstacles:
                obs.draw(screen)

            for proj in projectiles:
                proj.draw(screen)

            if boss:
                boss.draw(screen)

            jellyfish.draw(screen, particles)

            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            draw_text(screen, "Game Over!", 52, WIDTH // 2, 180, (255, 120, 120))
            draw_text(screen, f"Score: {score}", 40, WIDTH // 2, 250, (255, 255, 150))
            draw_text(screen, f"Best: {high_score}", 36, WIDTH // 2, 300, (150, 255, 150))

            if boss_round > 0:
                draw_text(screen, f"Bosses Defeated: {boss_round}/{max_boss_rounds}", 26, WIDTH // 2, 350,
                          (255, 215, 0))

            if boss_round >= max_boss_rounds:
                draw_text(screen, "ALL BOSSES DEFEATED!", 28, WIDTH // 2, 390, (255, 215, 0))

            draw_text(screen, "Click to Continue", 26, WIDTH // 2, 450, (200, 255, 255))

        elif game_state == "leaderboard":
            # Draw leaderboard screen
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 30, 80, 230))
            screen.blit(overlay, (0, 0))

            draw_text(screen, "LEADERBOARD", 48, WIDTH // 2, 50, (255, 215, 0))
            draw_text(screen, "Top 10 Scores", 22, WIDTH // 2, 85, (200, 200, 255))

            leaderboard = game_data.get("leaderboard", [])
            if leaderboard:
                # Header
                draw_text(screen, "Rank", 18, 40, 115, (150, 150, 200))
                draw_text(screen, "Player", 18, 120, 115, (150, 150, 200))
                draw_text(screen, "Score", 18, 230, 115, (150, 150, 200))
                draw_text(screen, "Region", 18, 320, 115, (150, 150, 200))

                for i, entry in enumerate(leaderboard[:10]):
                    y_pos = 145 + i * 36
                    rank_color = (255, 215, 0) if i == 0 else (192, 192, 192) if i == 1 else (205, 127,
                                                                                              50) if i == 2 else WHITE

                    # Rank
                    draw_text(screen, f"#{i + 1}", 22, 40, y_pos, rank_color)

                    # Username (truncate if too long)
                    username = entry.get('username', 'Unknown')[:10]
                    draw_text(screen, username, 20, 120, y_pos, WHITE)

                    # Score
                    draw_text(screen, f"{entry['score']}", 24, 230, y_pos, (255, 255, 150))

                    # Region (truncate if too long)
                    region = entry.get('region', '')[:8]
                    draw_text(screen, region, 18, 320, y_pos, (150, 200, 255))
            else:
                draw_text(screen, "No scores yet!", 28, WIDTH // 2, 250, (150, 150, 200))
                draw_text(screen, "Play to set records!", 24, WIDTH // 2, 290, (150, 150, 200))

            back_button.update(mouse_pos)
            back_button.draw(screen)

            if back_button.is_clicked(mouse_pos, mouse_pressed):
                game_state = "menu"

        elif game_state == "stats":
            # Draw stats screen
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 30, 80, 230))
            screen.blit(overlay, (0, 0))

            draw_text(screen, "STATISTICS", 48, WIDTH // 2, 60, (100, 255, 200))

            stats = game_data.get("stats", {})

            # Stats display
            stat_items = [
                ("Games Played", stats.get("games_played", 0)),
                ("Best Score", stats.get("best_score", 0)),
                ("Total Score", stats.get("total_score", 0)),
                ("Bosses Defeated", stats.get("bosses_defeated", 0)),
                ("Highest Boss", f"{stats.get('highest_boss', 0)}/10"),
                ("Time Played",
                 f"{stats.get('total_time_played', 0) // 60}m {stats.get('total_time_played', 0) % 60}s"),
            ]

            for i, (label, value) in enumerate(stat_items):
                y_pos = 140 + i * 55
                draw_text(screen, label, 24, WIDTH // 2, y_pos, (180, 180, 220))
                draw_text(screen, str(value), 36, WIDTH // 2, y_pos + 28, (255, 255, 150))

            # Average score
            games = stats.get("games_played", 0)
            if games > 0:
                avg_score = stats.get("total_score", 0) // games
                draw_text(screen, f"Avg Score: {avg_score}", 22, WIDTH // 2, 480, (150, 200, 255))

            back_button.update(mouse_pos)
            back_button.draw(screen)

            if back_button.is_clicked(mouse_pos, mouse_pressed):
                game_state = "menu"

        elif game_state == "profile":
            # Draw profile screen
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 30, 80, 230))
            screen.blit(overlay, (0, 0))

            draw_text(screen, "PLAYER PROFILE", 44, WIDTH // 2, 60, (255, 180, 100))
            draw_text(screen, "Enter your info for leaderboard", 20, WIDTH // 2, 100, (200, 200, 255))

            # Update and draw input boxes
            username_input.update()
            region_input.update()
            username_input.draw(screen)
            region_input.draw(screen)

            # Instructions
            draw_text(screen, "Click box to type, Enter to confirm", 16, WIDTH // 2, 370, (150, 150, 200))

            save_button.update(mouse_pos)
            save_button.draw(screen)

            back_button.update(mouse_pos)
            back_button.draw(screen)

            if save_button.is_clicked(mouse_pos, mouse_pressed):
                if username_input.text.strip():
                    player_username = username_input.text.strip()
                    player_region = region_input.text.strip()
                    game_data["player"]["username"] = player_username
                    game_data["player"]["region"] = player_region
                    save_game_data(game_data)
                    game_state = "menu"

            if back_button.is_clicked(mouse_pos, mouse_pressed):
                # Restore original values if cancelled
                username_input.text = player_username
                region_input.text = player_region
                game_state = "menu"

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()