import argparse
import json
import math
import os
import queue
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation


G = 9.8
BODY_AXIS_COLORS = {
    "IMU X": "tab:red",
    "IMU Y": "tab:green",
    "IMU Z": "tab:blue",
}
WORLD_AXIS_COLORS = {
    "Earth X": "#666666",
    "Earth Y": "#888888",
    "Earth Up": "#111111",
}
EARTH_AXIS_MIN_VISIBLE = 0.08
LINEAR_ACC_VISUAL_SCALE = 0.22
LINEAR_ACC_VISUAL_MAX = 1.2
LINEAR_ACC_DEADBAND = 0.25
EARTH_AXIS_DECAY_PER_FRAME = 0.08
EARTH_AXIS_HYSTERESIS = 0.18
VELOCITY_DEADBAND = 0.05
VELOCITY_LEAK = 0.85


def rotation_matrix_from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    """
    Body frame -> world frame.
    roll around X, pitch around Y, yaw around Z.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return rz @ ry @ rx


def gravity_body_from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    rmat = rotation_matrix_from_euler_deg(roll_deg, pitch_deg, yaw_deg)
    gravity_world = np.array([0.0, 0.0, G], dtype=float)
    return rmat.T @ gravity_world


def clamp_vector_magnitude(vec, visual_scale, max_length):
    scaled_vec = np.array(vec, dtype=float) * visual_scale
    scaled_norm = np.linalg.norm(scaled_vec)
    if scaled_norm <= max_length or scaled_norm < 1e-9:
        return scaled_vec
    return scaled_vec / scaled_norm * max_length


def scale_axis_by_component(axis_vec, component_value, visual_scale, min_visible, max_length):
    axis_vec = np.array(axis_vec, dtype=float)
    axis_unit = axis_vec / np.linalg.norm(axis_vec)
    signed_length = component_value * visual_scale
    if 0.0 < abs(signed_length) < min_visible:
        signed_length = math.copysign(min_visible, signed_length)
    signed_length = max(-max_length, min(max_length, signed_length))
    return axis_unit * signed_length


def wrap_angle_deg(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def load_calibration(calibration_path):
    if not calibration_path:
        return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    with open(calibration_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "roll": float(data.get("roll_offset_deg", 0.0)),
        "pitch": float(data.get("pitch_offset_deg", 0.0)),
        "yaw": float(data.get("yaw_offset_deg", 0.0)),
    }


def serial_reader(port, baud, out_queue, stop_event):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=1)
    except serial.SerialException as exc:
        print(f"[SERIAL ERROR] Cannot open {port} at {baud}: {exc}")
        return

    print(f"[SERIAL] Opened {port} at {baud}")
    last_status_log = 0.0

    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line or not line.startswith("{"):
                continue

            data = json.loads(line)

            if "ax" not in data:
                now = time.time()
                if now - last_status_log >= 1.0:
                    print("[INFO]", data)
                    last_status_log = now
                continue

            out_queue.put(data)

        except json.JSONDecodeError:
            continue
        except serial.SerialException as exc:
            print("[SERIAL ERROR]", exc)
            break

    try:
        ser.close()
    except Exception:
        pass


class ImuVisualizer:
    def __init__(self, port, baud, calibration_path=None):
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.calibration_path = calibration_path
        self.angle_offsets = load_calibration(calibration_path)

        self.reader_thread = threading.Thread(
            target=serial_reader,
            args=(port, baud, self.data_queue, self.stop_event),
            daemon=True,
        )
        self.reader_thread.start()

        self.latest = None
        self.last_t = None
        self.velocity_world = np.zeros(3)
        self.displayed_acc_world = np.zeros(3)

        self.fig = plt.figure(figsize=(13, 7))
        self.ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_text = self.fig.add_subplot(1, 2, 2)
        self.ax_text.axis("off")

        self.text_handle = self.ax_text.text(
            0.02,
            0.98,
            "Waiting for IMU data...",
            va="top",
            family="monospace",
            fontsize=9,
        )

        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_close(self, event):
        self.stop_event.set()

    def update_latest_data(self):
        updated = False

        while not self.data_queue.empty():
            self.latest = self.data_queue.get()
            updated = True

        return updated

    def corrected_angles(self, data):
        roll = wrap_angle_deg(float(data.get("roll", 0.0)) - self.angle_offsets["roll"])
        pitch = wrap_angle_deg(float(data.get("pitch", 0.0)) - self.angle_offsets["pitch"])
        yaw = wrap_angle_deg(float(data.get("yaw", 0.0)) - self.angle_offsets["yaw"])
        return roll, pitch, yaw

    def update_velocity_estimate(self, data):
        now = time.time()

        if self.last_t is None:
            self.last_t = now
            return np.zeros(3), np.zeros(3), np.zeros(3)

        dt = now - self.last_t
        self.last_t = now

        if dt <= 0 or dt > 0.5:
            return np.zeros(3), np.zeros(3), self.velocity_world

        roll, pitch, yaw = self.corrected_angles(data)

        ax = float(data.get("ax", 0.0))
        ay = float(data.get("ay", 0.0))
        az = float(data.get("az", 0.0))

        rmat = rotation_matrix_from_euler_deg(roll, pitch, yaw)
        gravity_body = gravity_body_from_euler_deg(roll, pitch, yaw)

        acc_body = np.array([ax, ay, az], dtype=float)
        linear_acc_body = acc_body - gravity_body
        linear_acc_world = rmat @ linear_acc_body

        if np.linalg.norm(linear_acc_world) < LINEAR_ACC_DEADBAND:
            linear_acc_body = np.zeros(3)
            linear_acc_world = np.zeros(3)

        self.velocity_world = VELOCITY_LEAK * self.velocity_world + linear_acc_world * dt

        if np.linalg.norm(self.velocity_world) < VELOCITY_DEADBAND:
            self.velocity_world = np.zeros(3)

        return linear_acc_body, linear_acc_world, self.velocity_world

    def update_display_acc_world(self, linear_acc_world):
        for index, value in enumerate(linear_acc_world):
            shown = self.displayed_acc_world[index]

            if abs(value) >= abs(shown) + EARTH_AXIS_HYSTERESIS:
                self.displayed_acc_world[index] = value
                continue

            delta = value - shown
            if abs(delta) <= EARTH_AXIS_DECAY_PER_FRAME:
                self.displayed_acc_world[index] = value
            else:
                self.displayed_acc_world[index] = shown + math.copysign(
                    EARTH_AXIS_DECAY_PER_FRAME,
                    delta,
                )

        return self.displayed_acc_world.copy()

    def setup_3d_axes(self):
        self.ax3d.clear()
        self.ax3d.set_title("IMU Orientation + Motion Vector")
        self.ax3d.set_xlim([-1.5, 1.5])
        self.ax3d.set_ylim([-1.5, 1.5])
        self.ax3d.set_zlim([-1.5, 1.5])
        self.ax3d.set_xlabel("World X")
        self.ax3d.set_ylabel("World Y")
        self.ax3d.set_zlabel("World Z")
        self.ax3d.view_init(elev=25, azim=35)

    def draw_vector(self, origin, vec, label, color="black"):
        vec = np.array(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return

        self.ax3d.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            arrow_length_ratio=0.15,
            linewidth=2.0,
        )

        end = origin + vec
        self.ax3d.text(end[0], end[1], end[2], label, color=color)

    def draw_orientation(self, data, display_acc_world):
        roll, pitch, yaw = self.corrected_angles(data)

        rmat = rotation_matrix_from_euler_deg(roll, pitch, yaw)
        body_x = rmat @ np.array([1.0, 0.0, 0.0])
        body_y = rmat @ np.array([0.0, 1.0, 0.0])
        body_z = rmat @ np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, 0.0])
        earth_x = scale_axis_by_component(
            np.array([1.0, 0.0, 0.0]),
            display_acc_world[0],
            LINEAR_ACC_VISUAL_SCALE,
            EARTH_AXIS_MIN_VISIBLE,
            LINEAR_ACC_VISUAL_MAX,
        )
        earth_y = scale_axis_by_component(
            np.array([0.0, 1.0, 0.0]),
            display_acc_world[1],
            LINEAR_ACC_VISUAL_SCALE,
            EARTH_AXIS_MIN_VISIBLE,
            LINEAR_ACC_VISUAL_MAX,
        )
        earth_up = scale_axis_by_component(
            np.array([0.0, 0.0, 1.0]),
            display_acc_world[2],
            LINEAR_ACC_VISUAL_SCALE,
            EARTH_AXIS_MIN_VISIBLE,
            LINEAR_ACC_VISUAL_MAX,
        )

        self.draw_vector(
            origin, earth_x, "Earth X", color=WORLD_AXIS_COLORS["Earth X"]
        )
        self.draw_vector(
            origin, earth_y, "Earth Y", color=WORLD_AXIS_COLORS["Earth Y"]
        )
        self.draw_vector(
            origin, earth_up, "Earth Up", color=WORLD_AXIS_COLORS["Earth Up"]
        )

        self.draw_vector(
            origin, body_x, "IMU X", color=BODY_AXIS_COLORS["IMU X"]
        )
        self.draw_vector(
            origin, body_y, "IMU Y", color=BODY_AXIS_COLORS["IMU Y"]
        )
        self.draw_vector(
            origin, body_z, "IMU Z", color=BODY_AXIS_COLORS["IMU Z"]
        )

    def format_text(self, data, linear_acc_body, linear_acc_world, display_acc_world, velocity_world):
        ax = float(data.get("ax", 0.0))
        ay = float(data.get("ay", 0.0))
        az = float(data.get("az", 0.0))

        gx = float(data.get("gx", 0.0))
        gy = float(data.get("gy", 0.0))
        gz = float(data.get("gz", 0.0))

        roll_raw = float(data.get("roll", 0.0))
        pitch_raw = float(data.get("pitch", 0.0))
        yaw_raw = float(data.get("yaw", 0.0))
        roll, pitch, yaw = self.corrected_angles(data)

        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        linear_acc_body_mag = float(np.linalg.norm(linear_acc_body))
        linear_acc_world_mag = float(np.linalg.norm(linear_acc_world))
        display_acc_world_mag = float(np.linalg.norm(display_acc_world))
        vel_mag = float(np.linalg.norm(velocity_world))

        lines = [
            "========== IMU LIVE DATA ==========",
            f"Host time     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Serial        : {data.get('imu_baud', 'n/a')} baud, {data.get('imu_rx_bytes', 'n/a')} bytes",
            f"ESP ms        : {data.get('esp_ms', 0)}",
            f"IMU time      : {data.get('time', '')}",
            f"frames        : {data.get('frame_total', 0)}",
            f"bad checksum  : {data.get('bad_checksum', 0)}",
            "",
            "--- Raw acceleration, m/s^2 ---",
            f"ax            : {ax: .5f}",
            f"ay            : {ay: .5f}",
            f"az            : {az: .5f}",
            f"|a|           : {acc_mag: .5f}",
            "",
            "--- Linear acceleration, body ---",
            f"lax           : {linear_acc_body[0]: .5f}",
            f"lay           : {linear_acc_body[1]: .5f}",
            f"laz           : {linear_acc_body[2]: .5f}",
            f"|linear a|    : {linear_acc_body_mag: .5f}",
            "",
            "--- Linear acceleration, earth/world ---",
            f"lwax          : {linear_acc_world[0]: .5f}",
            f"lway          : {linear_acc_world[1]: .5f}",
            f"lwaz          : {linear_acc_world[2]: .5f}",
            f"|linear aw|   : {linear_acc_world_mag: .5f}",
            "",
            "--- Display acceleration, earth/world ---",
            f"dwax          : {display_acc_world[0]: .5f}",
            f"dway          : {display_acc_world[1]: .5f}",
            f"dwaz          : {display_acc_world[2]: .5f}",
            f"|display aw|  : {display_acc_world_mag: .5f}",
            "",
            "--- Gyroscope, deg/s ---",
            f"gx            : {gx: .5f}",
            f"gy            : {gy: .5f}",
            f"gz            : {gz: .5f}",
            f"|gyro|        : {gyro_mag: .5f}",
            "",
            "--- Angle, deg ---",
            f"roll raw      : {roll_raw: .5f}",
            f"pitch raw     : {pitch_raw: .5f}",
            f"yaw raw       : {yaw_raw: .5f}",
            f"roll          : {roll: .5f}",
            f"pitch         : {pitch: .5f}",
            f"yaw           : {yaw: .5f}",
            f"offsets       : {self.angle_offsets['roll']: .3f}, {self.angle_offsets['pitch']: .3f}, {self.angle_offsets['yaw']: .3f}",
            "",
            "--- Velocity estimate, world ---",
            f"vx            : {velocity_world[0]: .5f}",
            f"vy            : {velocity_world[1]: .5f}",
            f"vz            : {velocity_world[2]: .5f}",
            f"|v|           : {vel_mag: .5f}",
            "",
            "--- Earth axis scaling ---",
            f"min visible    : {EARTH_AXIS_MIN_VISIBLE: .2f}",
            f"a scale        : x{LINEAR_ACC_VISUAL_SCALE:.2f} (max {LINEAR_ACC_VISUAL_MAX:.1f})",
            f"decay/frame    : {EARTH_AXIS_DECAY_PER_FRAME: .2f}",
            f"hysteresis     : {EARTH_AXIS_HYSTERESIS: .2f}",
            "",
            "--- Quaternion ---",
            f"q0            : {float(data.get('q0', 0.0)): .6f}",
            f"q1            : {float(data.get('q1', 0.0)): .6f}",
            f"q2            : {float(data.get('q2', 0.0)): .6f}",
            f"q3            : {float(data.get('q3', 0.0)): .6f}",
        ]

        return "\n".join(lines)

    def format_waiting_text(self):
        port_name = "unknown"
        if len(os.sys.argv) > 1:
            port_name = " ".join(os.sys.argv[1:])

        lines = [
            "Waiting for IMU data...",
            "",
            "If the COM port is open but nothing appears:",
            "1. Check WT61 baud. The firmware now tries 9600 and 115200 automatically.",
            "2. Check UART wiring: IMU TX -> ESP32 RX, IMU RX -> ESP32 TX, common GND.",
            "3. Open a serial monitor and look for JSON status packets from the ESP32.",
            "4. If IMU is tilted while on a flat surface, run imu_calibrate.py first.",
            "",
            f"Command args  : {port_name}",
        ]
        return "\n".join(lines)

    def animate(self, _):
        self.update_latest_data()

        if self.latest is None:
            self.ax3d.clear()
            self.ax3d.set_title("IMU Orientation + Motion Vector")
            self.ax3d.set_xlim([-1.5, 1.5])
            self.ax3d.set_ylim([-1.5, 1.5])
            self.ax3d.set_zlim([-1.5, 1.5])
            self.ax3d.set_xlabel("World X")
            self.ax3d.set_ylabel("World Y")
            self.ax3d.set_zlabel("World Z")
            self.text_handle.set_text(self.format_waiting_text())
            return

        linear_acc_body, linear_acc_world, velocity_world = self.update_velocity_estimate(
            self.latest
        )
        display_acc_world = self.update_display_acc_world(linear_acc_world)

        self.setup_3d_axes()
        self.draw_orientation(self.latest, display_acc_world)
        self.text_handle.set_text(
            self.format_text(
                self.latest,
                linear_acc_body,
                linear_acc_world,
                display_acc_world,
                velocity_world,
            )
        )

    def run(self):
        _ani = FuncAnimation(
            self.fig,
            self.animate,
            interval=50,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()
        self.stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="ESP32 USB serial baud")
    parser.add_argument("--calibration", help="Path to calibration JSON produced by imu_calibrate.py")
    args = parser.parse_args()

    app = ImuVisualizer(args.port, args.baud, calibration_path=args.calibration)
    app.run()


if __name__ == "__main__":
    main()
