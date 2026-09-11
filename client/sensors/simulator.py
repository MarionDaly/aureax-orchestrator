"""Simulated IoT sensor data generator."""

from __future__ import annotations

import hashlib
import math
import random
import time
from datetime import UTC, datetime


class SensorSimulator:
    """Generates realistic IoT sensor data with a stable per-node personality.

    The per-node seed fixes the baselines, the drain rate and the noise
    sequence, so each node reads differently from its neighbours and identically
    across restarts. The emitted VALUES are not reproducible: they move with
    wall-clock time (daily cycle, drift, battery drain) and with how many reads
    preceded them, since every read draws the next number from the noise stream.
    """

    def __init__(self, node_id: str) -> None:
        """Initialize with a deterministic seed derived from node_id."""
        self.node_id = node_id
        # Deterministic seed from node_id
        seed = int(hashlib.sha256(node_id.encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed)
        self._start_time = time.time()
        # Battery starts full, slowly drains
        self._battery = 100.0
        # Fixed per node, not per read — see read_battery.
        self._battery_drain_rate = 0.5 + self._rng.uniform(0, 0.3)  # % per hour
        # Drawn ONCE per instance, like the drain rate above: the baselines are
        # this node's personality, so a read only adds the daily cycle and
        # noise on top. Drawing them per read would make consecutive samples
        # jump by whole degrees with no daily-cycle explanation.
        self._temp_base = 20.0 + self._rng.uniform(-2, 2)
        self._humidity_base = 55.0 + self._rng.uniform(-10, 10)
        self._pressure_base = 1013.25 + self._rng.uniform(-5, 5)

    def _time_factor(self) -> float:
        """0-1 factor over a synthetic 600s "day" cycle.

        Phased from simulator construction, NOT wall-clock time of day: two
        nodes started a minute apart report opposite points in the cycle, and a
        pod restart resets the phase.
        """
        elapsed = time.time() - self._start_time
        # Complete a "day" cycle every 600 seconds for demo purposes
        return (math.sin(elapsed * 2 * math.pi / 600) + 1) / 2

    def read_temperature(self) -> dict:
        """Simulated temperature sensor (Celsius)."""
        daily = self._time_factor() * 15  # Daily variation 0-15C
        noise = self._rng.gauss(0, 0.5)
        value = round(self._temp_base + daily + noise, 2)
        return self._reading("temperature", value, "celsius")

    def read_humidity(self) -> dict:
        """Simulated humidity sensor (percentage)."""
        daily = (1 - self._time_factor()) * 20  # Inverse of temperature
        noise = self._rng.gauss(0, 2)
        value = round(max(0, min(100, self._humidity_base + daily + noise)), 2)
        return self._reading("humidity", value, "percent")

    def read_pressure(self) -> dict:
        """Simulated barometric pressure sensor (hPa)."""
        base = self._pressure_base
        # +/-3 hPa over a ~6.3h period (2*pi*3600s), keyed to ABSOLUTE time, so
        # the drift is fleet-wide in phase and survives restarts — unlike the
        # per-instance _time_factor cycle above.
        drift = math.sin(time.time() / 3600) * 3
        noise = self._rng.gauss(0, 0.3)
        value = round(base + drift + noise, 2)
        return self._reading("pressure", value, "hPa")

    def read_light(self) -> dict:
        """Simulated ambient light sensor (lux)."""
        daily = self._time_factor()
        # 0-80000 lux range, peaks at the cycle maximum (see _time_factor —
        # this is the synthetic 600s cycle, not any real time of day)
        value = round(daily * 80000 * self._rng.uniform(0.8, 1.0), 1)
        return self._reading("light", value, "lux")

    def read_battery(self) -> dict:
        """Simulated battery level (slowly decreasing).

        The drain rate is fixed per simulator instance, because each call
        recomputes `100 - elapsed * rate` from it: a rate re-rolled per read
        would make the level jump up and down between samples, the one thing a
        battery reading must never do. Monotonicity is also enforced directly in
        :meth:`_battery_level`, so a clock adjustment cannot make it climb.
        """
        return self._reading("battery", round(self._battery_level(), 1), "percent")

    def _battery_level(self) -> float:
        """Advance and return the monotonic battery level.

        The single source for both MCP tools, so read_sensor("battery") and
        get_sensor_status cannot disagree: the cached ``self._battery`` only
        advances here, so a caller that read it directly would report 100.0 for
        the life of a pod whose battery sensor was never read.
        """
        elapsed_hours = (time.time() - self._start_time) / 3600
        level = max(0.0, 100 - elapsed_hours * self._battery_drain_rate)
        self._battery = min(self._battery, level)
        return self._battery

    def read_all(self) -> list[dict]:
        """Read all sensors at once."""
        return [
            self.read_temperature(),
            self.read_humidity(),
            self.read_pressure(),
            self.read_light(),
            self.read_battery(),
        ]

    def read(self, sensor_type: str) -> dict | list[dict]:
        """Read a specific sensor or all sensors."""
        readers = {
            "temperature": self.read_temperature,
            "humidity": self.read_humidity,
            "pressure": self.read_pressure,
            "light": self.read_light,
            "battery": self.read_battery,
            "all": self.read_all,
        }
        reader = readers.get(sensor_type)
        if not reader:
            # Raise, don't return {"error": ...}: the MCP server json-dumps a
            # returned dict verbatim, so the CallToolResult came back with
            # isError=False and MCPPool parsed it into the same shape it uses
            # for "No session for node" and transport failures — the agent loop
            # fed a bad argument back to the model as a normal tool result.
            raise ValueError(
                f"Unknown sensor type: {sensor_type!r}; expected one of {sorted(readers)}"
            )
        return reader()

    def get_status(self) -> dict:
        """Get operational status of all simulated sensors."""
        return {
            "node_id": self.node_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "sensors": {
                "temperature": {"status": "online", "unit": "celsius"},
                "humidity": {"status": "online", "unit": "percent"},
                "pressure": {"status": "online", "unit": "hPa"},
                "light": {"status": "online", "unit": "lux"},
                "battery": {
                    "status": "online",
                    "unit": "percent",
                    "level": round(self._battery_level(), 1),
                },
            },
        }

    def _reading(self, sensor_type: str, value: float, unit: str) -> dict:
        """Build a timestamped sensor reading dict."""
        return {
            "node_id": self.node_id,
            "sensor_type": sensor_type,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(UTC).isoformat(),
        }
