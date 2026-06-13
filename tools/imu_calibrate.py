import argparse
import json
import math
import statistics
import time

import serial


def wrap_angle_deg(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def mean_angle_deg(samples):
    radians_sum_sin = 0.0
    radians_sum_cos = 0.0

    for angle in samples:
        radians = math.radians(angle)
        radians_sum_sin += math.sin(radians)
        radians_sum_cos += math.cos(radians)

    return math.degrees(math.atan2(radians_sum_sin, radians_sum_cos))


def read_json_line(ser):
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("{"):
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "roll" in data and "pitch" in data and "yaw" in data:
            return data


def main():
    parser = argparse.ArgumentParser(
        description="Collect stationary IMU samples and save angle offsets."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="ESP32 USB serial baud")
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="How long to average while IMU stays still on a flat surface",
    )
    parser.add_argument(
        "--output",
        default="tools/imu_calibration.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    print("Place the IMU still on a flat surface before calibration starts.")
    print(f"Opening {args.port} at {args.baud} baud...")

    ser = serial.Serial(args.port, baudrate=args.baud, timeout=1)
    roll_samples = []
    pitch_samples = []
    yaw_samples = []
    ax_samples = []
    ay_samples = []
    az_samples = []

    try:
        warmup_deadline = time.time() + 2.0
        while time.time() < warmup_deadline:
            read_json_line(ser)

        print(f"Collecting samples for {args.seconds:.1f} seconds...")
        deadline = time.time() + args.seconds

        while time.time() < deadline:
            data = read_json_line(ser)
            roll_samples.append(float(data["roll"]))
            pitch_samples.append(float(data["pitch"]))
            yaw_samples.append(float(data["yaw"]))
            ax_samples.append(float(data.get("ax", 0.0)))
            ay_samples.append(float(data.get("ay", 0.0)))
            az_samples.append(float(data.get("az", 0.0)))

        calibration = {
            "created_at_epoch": time.time(),
            "sample_count": len(roll_samples),
            "roll_offset_deg": wrap_angle_deg(mean_angle_deg(roll_samples)),
            "pitch_offset_deg": wrap_angle_deg(mean_angle_deg(pitch_samples)),
            "yaw_offset_deg": wrap_angle_deg(mean_angle_deg(yaw_samples)),
            "roll_stddev_deg": statistics.pstdev(roll_samples) if len(roll_samples) > 1 else 0.0,
            "pitch_stddev_deg": statistics.pstdev(pitch_samples) if len(pitch_samples) > 1 else 0.0,
            "yaw_stddev_deg": statistics.pstdev(yaw_samples) if len(yaw_samples) > 1 else 0.0,
            "avg_acc_m_s2": {
                "ax": statistics.fmean(ax_samples) if ax_samples else 0.0,
                "ay": statistics.fmean(ay_samples) if ay_samples else 0.0,
                "az": statistics.fmean(az_samples) if az_samples else 0.0,
            },
            "notes": "Offsets assume the IMU was motionless on the desired zero-angle surface.",
        }

        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=2)

        print("Calibration saved.")
        print(json.dumps(calibration, indent=2))
        print("")
        print("Use it with:")
        print(
            f"python tools/imu_visualizer.py --port {args.port} --baud {args.baud} --calibration {args.output}"
        )
    finally:
        ser.close()


if __name__ == "__main__":
    main()
