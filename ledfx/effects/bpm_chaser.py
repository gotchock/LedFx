import time

import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


def _smoothstep(t):
    """Hermite curve: 0→1 with zero slope at both ends (eases in and out)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease(t, aggressiveness):
    """Blend linear (aggressiveness=0) towards smoothstep (aggressiveness=1)."""
    t = max(0.0, min(1.0, t))
    return t + (_smoothstep(t) - t) * aggressiveness


def _resync(phase, target, correction, gain, max_correction):
    """Nudge a phase's playback-rate correction towards closing the gap
    between where it is and where a beat says it should be, without ever
    moving the phase itself. Rate stays positive and only changes by a
    small bounded amount, so the phase keeps moving in the same direction
    and just gently speeds up or slows down until it resyncs.
    """
    err = phase - target
    err -= round(err)  # shortest signed distance around the 0..1 circle
    correction -= gain * err
    return max(1.0 - max_correction, min(1.0 + max_correction, correction))


class BPMChasers(AudioReactiveEffect, GradientEffect):
    """Luminosity pulsing synced to BPM with smooth attack/decay and break detection."""

    NAME = "BPM Chasers"
    CATEGORY = "BPM"
    HIDDEN_KEYS = ["gradient_roll"]

    # Below this, filtered low-frequency (kick+bass) power is treated as
    # "nothing happening" for break detection. The bpm tracker keeps
    # extrapolating a beat grid through quiet passages by design, so it
    # can't be used on its own to notice a break.
    BREAK_ENERGY_THRESHOLD = 0.1

    # Resync tuning: how hard each beat's phase error pulls the playback
    # rate back into alignment, and how far that rate is allowed to stray
    # from 1x. Kept small so desyncs are corrected by quietly speeding up
    # or slowing down rather than ever snapping the phase itself.
    RESYNC_GAIN = 0.5
    MAX_RATE_CORRECTION = 0.12

    BEAT_DIVISIONS = {
        "Every beat": 1,
        "Every 2 beats": 2,
        "Every 4 beats": 4,
        "Every 8 beats": 8,
    }

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "beat_cycle",
                description="How many beats per full bright-dim cycle",
                default="Every 2 beats",
            ): vol.In(list(BEAT_DIVISIONS.keys())),
            vol.Optional(
                "gradient_speed",
                description="How many beats to fully traverse the gradient",
                default="Every 4 beats",
            ): vol.In(list(BEAT_DIVISIONS.keys())),
            vol.Optional(
                "attack",
                description="Attack duration, as a fraction of the full beat cycle, starting at phase 0",
                default=0.15,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "hold",
                description="Hold at full brightness, as a fraction of the full beat cycle, right after attack",
                default=0.1,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "decay",
                description="Decay duration, as a fraction of the full beat cycle, right after hold",
                default=0.25,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "ease",
                description="Ease in/out aggressiveness shared by attack and decay: 0=linear ramp, 1=full S-curve ease",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "min_brightness",
                description="Minimum brightness at the dimmest point",
                default=0.05,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1)),
            vol.Optional(
                "max_slew",
                description="Minimum time (seconds) for brightness to sweep fully from 0 to 1. Hard caps any abrupt jump (beat/break transitions, tempo mismatch) to this rate; 0 disables the limiter",
                default=0.05,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=5)),
        }
    )

    def on_activate(self, pixel_count):
        self.color = self.get_gradient_color(0)
        self.output_brightness = 0.0
        now = time.time()
        self._last_render_time = now
        # Beat period estimation (default 120 BPM)
        self._beat_period = 0.5
        self._prev_beat_time = None
        # Pulsation cycle: a phase that only ever moves forward (0 -> 1,
        # wrapping back to 0) at a rate close to 1 cycle per beat_cycle
        # beats. Beats never snap the phase itself; they only nudge
        # _phase_rate_corr up or down a little, so a desync is corrected
        # by quietly speeding up/slowing down instead of jumping/reversing.
        self._phase = 0.0
        self._phase_rate_corr = 1.0
        self._beat_seq = 0
        # Gradient traversal cycle, same phase-locked approach
        self._gradient_phase = 0.0
        self._gradient_rate_corr = 1.0
        self._gradient_beat_seq = 0
        # Break detection, driven by actual low-frequency energy rather
        # than the bpm tracker (which keeps ticking through breaks)
        self._last_energetic_time = now
        self._in_break = False
        self._break_start = now

    def config_updated(self, config):
        self.beat_cycle = self.BEAT_DIVISIONS[self._config["beat_cycle"]]
        self.gradient_speed = self.BEAT_DIVISIONS[
            self._config["gradient_speed"]
        ]
        self.attack_pct = self._config["attack"]
        self.hold_pct = self._config["hold"]
        self.decay_pct = self._config["decay"]
        self.ease = self._config["ease"]
        self.min_brightness = self._config["min_brightness"]
        self.max_slew = self._config["max_slew"]

    def audio_data_updated(self, data):
        now = time.time()
        dt = max(0.0, now - self._last_render_time)

        # --- Advance both phases forward at their current (gently
        # corrected) rate. Rate is always positive, so phase only ever
        # increases and wraps 1.0 -> 0.0 in the same direction: it never
        # reverses, no matter how far off tempo the beat tracker is. ---
        brightness_rate = self._phase_rate_corr / (
            self._beat_period * self.beat_cycle
        )
        self._phase = (self._phase + dt * brightness_rate) % 1.0

        gradient_rate = self._gradient_rate_corr / (
            self._beat_period * self.gradient_speed
        )
        self._gradient_phase = (
            self._gradient_phase + dt * gradient_rate
        ) % 1.0

        # --- Beat period estimation and gentle resync ---
        if data.bpm_beat_now():
            if self._prev_beat_time is not None:
                measured = now - self._prev_beat_time
                # Accept only plausible inter-beat intervals (30–300 BPM)
                if 0.2 <= measured <= 2.0:
                    # Exponential moving average towards measured period
                    self._beat_period += 0.25 * (measured - self._beat_period)
            self._prev_beat_time = now

            # This beat should land at beat_seq/beat_cycle through the
            # pulsation cycle; compare to where the phase actually is and
            # nudge its rate correction accordingly, rather than snapping
            # the phase itself to that target.
            self._beat_seq = (self._beat_seq + 1) % self.beat_cycle
            target = self._beat_seq / self.beat_cycle
            self._phase_rate_corr = _resync(
                self._phase,
                target,
                self._phase_rate_corr,
                self.RESYNC_GAIN,
                self.MAX_RATE_CORRECTION,
            )

            # Same idea for the gradient cycle, at its own division
            self._gradient_beat_seq = (
                self._gradient_beat_seq + 1
            ) % self.gradient_speed
            gradient_target = self._gradient_beat_seq / self.gradient_speed
            self._gradient_rate_corr = _resync(
                self._gradient_phase,
                gradient_target,
                self._gradient_rate_corr,
                self.RESYNC_GAIN,
                self.MAX_RATE_CORRECTION,
            )

        # Colour cycles through the gradient every gradient_speed beats
        self.color = self.get_gradient_color(self._gradient_phase)

        # --- Break detection: driven by actual kick/bass energy, since
        # bpm_beat_now() keeps extrapolating a beat grid through breaks
        # and would never go quiet on its own ---
        bar_period = self._beat_period * 4
        if data.lows_power(filtered=True) > self.BREAK_ENERGY_THRESHOLD:
            self._last_energetic_time = now
            if self._in_break:
                # Real energy is back: exit the break and restart the
                # cycles right on this hit. This is the one deliberate
                # hard resync point -- landing a fresh attack exactly on
                # the hit is the desired effect, not a bug.
                self._in_break = False
                self._phase = 0.0
                self._phase_rate_corr = 1.0
                self._beat_seq = 0
                self._gradient_phase = 0.0
                self._gradient_rate_corr = 1.0
                self._gradient_beat_seq = 0
        elif (
            not self._in_break
            and (now - self._last_energetic_time) > bar_period
        ):
            self._in_break = True
            self._break_start = now

        # --- Brightness computation ---
        if self._in_break:
            # No pulsation during break; ramp luminosity 0 → 1 over 4 bars
            ramp = min((now - self._break_start) / (bar_period * 4), 1.0)
            shaped = ramp
        else:
            phase = self._phase

            # Segment boundaries: attack, then hold, then decay, then low hold
            attack_end = self.attack_pct
            hold_end = attack_end + self.hold_pct
            decay_end = hold_end + self.decay_pct
            if decay_end > 1.0:
                # Segments overrun the cycle: scale them down to fit
                scale = 1.0 / decay_end
                attack_end *= scale
                hold_end *= scale
                decay_end *= scale

            if phase < attack_end:
                # Attack: rise from 0 → 1
                t = phase / attack_end if attack_end > 1e-6 else 1.0
                shaped = _ease(t, self.ease)
            elif phase < hold_end:
                # Hold at full brightness
                shaped = 1.0
            elif phase < decay_end:
                # Decay: fall from 1 → 0
                span = decay_end - hold_end
                t = (phase - hold_end) / span if span > 1e-6 else 1.0
                shaped = 1.0 - _ease(t, self.ease)
            else:
                # Low hold until the cycle wraps back to the next attack
                shaped = 0.0

        target_brightness = self.min_brightness + shaped * (
            1.0 - self.min_brightness
        )

        # Slew-limit the output: whatever caused the target to move (break
        # entry/exit, tempo re-estimation), the rendered brightness can
        # only change at a bounded rate, so it can never pop in a single
        # frame.
        self._last_render_time = now
        if self.max_slew > 1e-6:
            max_delta = dt / self.max_slew
            delta = max(
                -max_delta,
                min(max_delta, target_brightness - self.output_brightness),
            )
            self.output_brightness += delta
        else:
            self.output_brightness = target_brightness

    def render(self):
        self.pixels[:] = self.color * self.output_brightness
