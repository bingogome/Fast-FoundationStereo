#!/usr/bin/env python3
"""Capture one frame from RealSense IR stereo (and optional RGB) and save it."""

import argparse
import os
import time
from contextlib import ExitStack
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_frames", type=int, default=15)
    parser.add_argument("--use_color", type=int, default=1)
    parser.add_argument("--emitter", type=int, default=1, help="IR emitter: 0 off, 1 on, -1 keep current.")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense serial number.")
    parser.add_argument(
        "--hardware_reset_on_start",
        type=int,
        default=1,
        help="Reset RealSense device before start (software unplug/replug).",
    )
    parser.add_argument(
        "--reset_wait_sec",
        type=float,
        default=3.0,
        help="Wait time after hardware reset for USB reconnect.",
    )
    parser.add_argument("--out_dir", type=str, default="output/realsense_test")
    parser.add_argument("--save_intrinsic_file", type=str, default=None)
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Show OpenCV preview windows (0 for headless/SSH, 1 for local GUI).",
    )
    return parser.parse_args()


def intrinsics_to_k(intr):
    return np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def save_runtime_k(path, K, baseline):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{x:.8f}" for x in K.reshape(-1)) + "\n")
        f.write(f"{baseline:.8f}\n")


def frame_to_ndarray(frame):
    """Convert a pyrealsense2 video frame into a concrete uint8 numpy array."""
    if frame is None:
        return None

    # First try pyrealsense2's array interface.
    arr = np.asarray(frame.get_data())
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.size > 0:
        return np.ascontiguousarray(arr)

    # Fallback for platforms where array conversion returns an object-like view.
    video = frame.as_video_frame()
    h, w = video.get_height(), video.get_width()
    bpp = video.get_bytes_per_pixel()
    channels = max(1, int(bpp))

    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    expected = h * w * channels
    if raw.size < expected:
        raise RuntimeError(
            f"Frame buffer too small: got {raw.size} bytes, expected at least {expected}"
        )
    raw = raw[:expected]
    if channels == 1:
        return raw.reshape(h, w)
    return raw.reshape(h, w, channels)


def save_image(path, img, *, bgr=False):
    """Save an image using Pillow to avoid OpenCV imwrite ABI/runtime issues."""
    arr = np.ascontiguousarray(np.asarray(img))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
        return

    if arr.ndim == 3 and arr.shape[2] == 3:
        if bgr:
            arr = arr[:, :, ::-1]  # BGR -> RGB
        Image.fromarray(arr, mode="RGB").save(path)
        return

    raise ValueError(f"Unsupported image shape for save: {arr.shape}")


def stop_pipeline_safely(pipeline):
    if pipeline is None:
        return
    pipeline.stop()


def pick_device_or_raise(rs, args):
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    if args.serial:
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == args.serial:
                return dev
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        raise RuntimeError(
            f"Requested serial {args.serial} not found. Connected serials: {serials}"
        )

    if len(devices) > 1:
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        raise RuntimeError(
            f"Multiple RealSense devices found {serials}. Please pass --serial."
        )
    return devices[0]


def reset_device_if_requested(rs, args):
    if not args.hardware_reset_on_start:
        return
    device = pick_device_or_raise(rs, args)
    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    print(f"Hardware reset RealSense {name} (serial={serial})")
    device.hardware_reset()
    time.sleep(max(0.0, args.reset_wait_sec))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    reset_device_if_requested(rs, args)

    if args.show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display detected; forcing headless mode (--show 0).")
        args.show = 0

    pipeline = None
    try:
        with ExitStack() as cleanup:
            if args.show:
                cleanup.callback(cv2.destroyAllWindows)

            pipeline = rs.pipeline()
            config = rs.config()
            if args.serial:
                config.enable_device(args.serial)
            config.enable_stream(
                rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps
            )
            config.enable_stream(
                rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps
            )
            if args.use_color:
                config.enable_stream(
                    rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
                )

            profile = pipeline.start(config)
            cleanup.callback(stop_pipeline_safely, pipeline)

            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if args.emitter in (0, 1) and depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))

            for _ in range(max(0, args.warmup_frames)):
                pipeline.wait_for_frames()

            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            color_frame = frames.get_color_frame() if args.use_color else None

            if not left_frame or not right_frame:
                raise RuntimeError("Failed to get IR stereo frames from RealSense.")
            if args.use_color and not color_frame:
                raise RuntimeError("Failed to get color frame while --use_color=1.")

            left = frame_to_ndarray(left_frame)
            right = frame_to_ndarray(right_frame)
            color = frame_to_ndarray(color_frame) if color_frame else None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            left_path = os.path.join(args.out_dir, f"left_{ts}.png")
            right_path = os.path.join(args.out_dir, f"right_{ts}.png")
            pair_path = os.path.join(args.out_dir, f"pair_{ts}.png")
            pair = np.concatenate([left, right], axis=1)
            save_image(left_path, left)
            save_image(right_path, right)
            save_image(pair_path, pair)

            color_path = None
            if color is not None:
                color_path = os.path.join(args.out_dir, f"color_{ts}.png")
                save_image(color_path, color, bgr=True)

            left_profile = left_frame.get_profile().as_video_stream_profile()
            right_profile = right_frame.get_profile().as_video_stream_profile()
            K = intrinsics_to_k(left_profile.get_intrinsics())
            baseline = abs(left_profile.get_extrinsics_to(right_profile).translation[0])

            if args.save_intrinsic_file:
                save_runtime_k(args.save_intrinsic_file, K, baseline)

            print("Captured one RealSense frame set.")
            print(f"Left IR:  {left_path}")
            print(f"Right IR: {right_path}")
            print(f"Pair:     {pair_path}")
            if color_path:
                print(f"Color:    {color_path}")
            if args.save_intrinsic_file:
                print(f"K+baseline file: {args.save_intrinsic_file}")
            print("Left IR intrinsic K:")
            print(K)
            print(f"Stereo baseline (m): {baseline:.6f}")

            if args.show:
                cv2.imshow("IR Left | IR Right", pair)
                if color is not None:
                    cv2.imshow("Color", color)
                print("Press 'q' or ESC to close preview.")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("q"), 27):
                        break

    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
