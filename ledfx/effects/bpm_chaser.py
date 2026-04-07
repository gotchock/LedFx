import time

import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


def _smoothstep(t):
    """Hermite curve: 0→1 with zero slope at both ends (eases in and out)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class BPMChasers(AudioReactiveEffect, GradientEffect):
    """Luminosity pulsing synced to BPM with smooth attack/decay and break detection."""

    NAME = "BPM Chasers"
    CATEGORY = "BPM"
    HIDDEN_KEYS = ["gradient_roll"]

    BEAT_DIVISIONS = {
        "Every beat": 1,
        "Every 2 beats": 2,
        "Every 4 beats": 4,
    }

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "beat_cycle",
                description="How many beats per full bright-dim cycle",
                default="Every beat",
            ): vol.In(list(BEAT_DIVISIONS.keys())),
            vol.Optional(
                "attack",
                description="Attack: 0=instant jump to peak on beat, 1=progressive smooth rise over the full attack half",
                default=0.3,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "decay",
                description="Decay: 0=instant drop to minimum after peak, 1=progressive smooth fall over the full decay half",
                default=0.7,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "min_brightness",
                description="Minimum brightness at the dimmest point",
                default=0.05,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.5)),
        }
    )

    def on_activate(self, pixel_count):
        self.color = self.get_gradient_color(0)
        self.output_brightness = 0.0
        now = time.time()
        # Beat period estimation (default 120 BPM)
        self._beat_period = 0.5
        self._prev_beat_time = None
        # Pulsation cycle tracking via wall clock (avoids oscillator-reset jitter)
        self._cycle_start = now
        self._beat_seq = 0
        # Break detection
        self._last_beat_time = now
        self._in_break = False
        self._break_start = now

    def config_updated(self, config):
        self.beat_cycle = self.BEAT_DIVISIONS[self._config["beat_cycle"]]
        self.attack = self._config["attack"]
        self.decay = self._config["decay"]
        self.min_brightness = self._config["min_brightness"]

    def audio_data_updated(self, data):
        now = time.time()

        # Colour follows bar oscillator; cosmetic micro-glitches there are acceptable
        bar = data.bar_oscillator()
        self.color = self.get_gradient_color((bar % 4) / 4)

        # --- Beat period estimation and cycle management ---
        if data.bpm_beat_now():
            if self._prev_beat_time is not None:
                measured = now - self._prev_beat_time
                # Accept only plausible inter-beat intervals (30–300 BPM)
                if 0.2 <= measured <= 2.0:
                    # Exponential moving average towards measured period
                    self._beat_period += 0.25 * (measured - self._beat_period)
            self._prev_beat_time = now
            self._last_beat_time = now

            # Advance beat sequence; reset pulsation cycle every beat_cycle beats
            self._beat_seq = (self._beat_seq + 1) % self.beat_cycle
            if self._beat_seq == 0:
                self._cycle_start = now

            # First beat after a break: restart the cycle immediately
            if self._in_break:
                self._in_break = False
                self._cycle_start = now
                self._beat_seq = 0

        # --- Break detection: 1 bar (4 beats) with no detected beat ---
        bar_period = self._beat_period * 4
        if not self._in_break and (now - self._last_beat_time) > bar_period:
            self._in_break = True
            self._break_start = now

        # --- Brightness computation ---
        if self._in_break:
            # No pulsation during break; ramp luminosity 0 → 1 over 4 bars
            ramp = min((now - self._break_start) / (bar_period * 4), 1.0)
            shaped = ramp
        else:
            cycle_period = self._beat_period * self.beat_cycle
            phase = ((now - self._cycle_start) / cycle_period) % 1.0

            if phase < 0.5:
                # Attack half: rise from 0 → 1
                # attack=0 → instant jump; attack=1 → full smoothstep ramp across the half
                t = phase * 2  # 0 → 1 across first half
                shaped = (
                    1.0
                    if self.attack < 1e-3
                    else _smoothstep(min(t / self.attack, 1.0))
                )
            else:
                # Decay half: fall from 1 → 0
                # decay=0 → instant drop; decay=1 → full smoothstep fall across the half
                t = (phase - 0.5) * 2  # 0 → 1 across second half
                shaped = (
                    0.0
                    if self.decay < 1e-3
                    else 1.0 - _smoothstep(min(t / self.decay, 1.0))
                )

        self.output_brightness = self.min_brightness + shaped * (
            1.0 - self.min_brightness
        )

    def render(self):
        self.pixels[:] = self.color * self.output_brightness
