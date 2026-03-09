#!/usr/bin/env python3
"""Auto-save keyframe point clouds from RealSense + Fast-FoundationStereo."""

import argparse
import csv
import logging
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import yaml

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CODE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, o3d, set_logging_format, vis_disparity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=f"{REPO_ROOT}/weights/23-36-37/model_best_bp2_serialize.pth",
        type=str,
    )
    parser.add_argument("--out_dir", type=str, default=f"{REPO_ROOT}/output/realsense_keyframes")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rectify", type=int, default=1)
    parser.add_argument("--equalize_hist", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--znear", type=float, default=0.1)
    parser.add_argument("--zfar", type=float, default=10.0)
    parser.add_argument("--pc_stride", type=int, default=2)
    parser.add_argument("--remove_invisible", type=int, default=1)
    parser.add_argument("--use_color", type=int, default=1)
    parser.add_argument("--emitter", type=int, default=1)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--hardware_reset_on_start", type=int, default=1)
    parser.add_argument("--reset_wait_sec", type=float, default=3.0)
    parser.add_argument("--save_intrinsic_file", type=str, default=None)
    parser.add_argument("--frame_timeout_ms", type=int, default=5000)
    parser.add_argument("--startup_retries", type=int, default=1)
    parser.add_argument("--restart_backoff_sec", type=float, default=0.8)
    parser.add_argument("--show_disp", type=int, default=1)
    parser.add_argument("--show_pc", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=30)

    parser.add_argument(
        "--diff_reference",
        type=str,
        default="last_saved",
        choices=["last_saved", "previous"],
        help="Frame used as reference for difference scoring.",
    )
    parser.add_argument(
        "--diff_mean_thresh",
        type=float,
        default=0.06,
        help="Save if mean absolute pixel diff / 255 exceeds this threshold.",
    )
    parser.add_argument(
        "--diff_ratio_thresh",
        type=float,
        default=0.20,
        help="Save if changed-pixel ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--pixel_diff_thresh",
        type=int,
        default=20,
        help="Pixel delta threshold used for changed-pixel ratio.",
    )
    parser.add_argument("--min_frames_between_saves", type=int, default=8)
    parser.add_argument("--min_seconds_between_saves", type=float, default=0.3)
    parser.add_argument(
        "--max_keyframes",
        type=int,
        default=0,
        help="Stop after N saved keyframes. 0 means unlimited.",
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


def distortion_to_array(intr):
    coeffs = np.array(intr.coeffs, dtype=np.float32)
    if coeffs.size < 5:
        out = np.zeros(5, dtype=np.float32)
        out[: coeffs.size] = coeffs
        return out
    return coeffs[:5]


def build_rectify_maps(left_profile, right_profile, width, height):
    intr_l = left_profile.get_intrinsics()
    intr_r = right_profile.get_intrinsics()
    K_l = intrinsics_to_k(intr_l)
    K_r = intrinsics_to_k(intr_r)
    D_l = distortion_to_array(intr_l)
    D_r = distortion_to_array(intr_r)

    extr = left_profile.get_extrinsics_to(right_profile)
    R = np.array(extr.rotation, dtype=np.float32).reshape(3, 3)
    T = np.array(extr.translation, dtype=np.float32)

    image_size = (width, height)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1[:, :3], image_size, cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2[:, :3], image_size, cv2.CV_32FC1
    )
    K_rect = P1[:, :3].astype(np.float32)
    baseline = float(abs(P2[0, 3] / P2[0, 0]))
    return (map_l1, map_l2, map_r1, map_r2), K_rect, baseline


def scale_image_and_k(img, K, scale):
    if abs(scale - 1.0) < 1e-6:
        return img, K.copy()
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    k_scaled = K.copy()
    sx = new_w / float(w)
    sy = new_h / float(h)
    k_scaled[0, 0] *= sx
    k_scaled[0, 2] *= sx
    k_scaled[1, 1] *= sy
    k_scaled[1, 2] *= sy
    return resized, k_scaled


def save_runtime_k(path, K, baseline):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{x:.8f}" for x in K.reshape(-1)) + "\n")
        f.write(f"{baseline:.8f}\n")


def stop_pipeline_safely(pipeline):
    if pipeline is None:
        return
    pipeline.stop()


def destroy_o3d_window_safely(vis):
    if vis is None:
        return
    vis.destroy_window()


def cleanup_runtime(state):
    stop_pipeline_safely(state.get("pipeline"))
    destroy_o3d_window_safely(state.get("vis"))
    cv2.destroyAllWindows()
    logging.info("Shutdown complete.")


def create_realsense_config(args):
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
    if args.use_color:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    return config


def pick_device_or_raise(args):
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
        raise RuntimeError(f"Multiple RealSense devices found {serials}. Please pass --serial.")
    return devices[0]


def reset_device_if_requested(args):
    if not args.hardware_reset_on_start:
        return
    dev = pick_device_or_raise(args)
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    logging.info("Hardware reset RealSense %s (serial=%s)", name, serial)
    dev.hardware_reset()
    time.sleep(max(0.0, args.reset_wait_sec))


def start_pipeline_once(args):
    with ExitStack() as cleanup:
        pipeline = rs.pipeline()
        profile = pipeline.start(create_realsense_config(args))
        cleanup.callback(stop_pipeline_safely, pipeline)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if args.emitter in (0, 1) and depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))

        frames = None
        for _ in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=args.frame_timeout_ms)
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        if not left_frame or not right_frame:
            raise RuntimeError("Failed to read IR frames during startup.")

        left_profile = left_frame.get_profile().as_video_stream_profile()
        right_profile = right_frame.get_profile().as_video_stream_profile()

        if args.rectify:
            rectify_maps, K_ir, baseline = build_rectify_maps(
                left_profile, right_profile, args.width, args.height
            )
        else:
            rectify_maps = None
            K_ir = intrinsics_to_k(left_profile.get_intrinsics())
            baseline = float(abs(left_profile.get_extrinsics_to(right_profile).translation[0]))

        K_color = None
        R_ir_to_color = None
        T_ir_to_color = None
        if args.use_color:
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            K_color = intrinsics_to_k(color_profile.get_intrinsics())
            extr = left_profile.get_extrinsics_to(color_profile)
            R_ir_to_color = np.array(extr.rotation, dtype=np.float32).reshape(3, 3)
            T_ir_to_color = np.array(extr.translation, dtype=np.float32)

        state = {
            "rectify_maps": rectify_maps,
            "K_ir": K_ir,
            "baseline": baseline,
            "K_color": K_color,
            "R_ir_to_color": R_ir_to_color,
            "T_ir_to_color": T_ir_to_color,
        }
        cleanup.pop_all()
        return pipeline, state


def start_pipeline_with_retries(args, reason):
    last_exc = None
    for attempt in range(1, args.startup_retries + 1):
        try:
            pipeline, state = start_pipeline_once(args)
            if attempt > 1:
                logging.info("RealSense recovered after %d attempts (%s).", attempt, reason)
            return pipeline, state
        except RuntimeError as exc:
            last_exc = exc
            logging.warning(
                "RealSense start attempt %d/%d failed (%s): %s",
                attempt,
                args.startup_retries,
                reason,
                exc,
            )
            time.sleep(max(0.0, args.restart_backoff_sec))
    raise RuntimeError(
        f"Failed to start RealSense after {args.startup_retries} attempts ({reason})."
    ) from last_exc


def load_model(args):
    if not os.path.isfile(args.model_dir):
        raise FileNotFoundError(f"Model not found: {args.model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime FFS inference.")
    with open(os.path.join(os.path.dirname(args.model_dir), "cfg.yaml"), "r", encoding="utf-8") as f:
        _ = yaml.safe_load(f)

    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()
    return model


def build_point_cloud(depth, K, stride):
    h, w = depth.shape
    valid = depth > 0
    if stride > 1:
        keep = np.zeros((h, w), dtype=bool)
        keep[::stride, ::stride] = True
        valid &= keep
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), valid
    v, u = np.where(valid)
    z = depth[v, u]
    x = (u.astype(np.float32) - K[0, 2]) * z / K[0, 0]
    y = (v.astype(np.float32) - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1), valid


def colorize_points_from_rgb(points_ir, color_bgr, K_color, R_ir_to_color, T_ir_to_color):
    if len(points_ir) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts_color = (R_ir_to_color @ points_ir.T).T + T_ir_to_color[None, :]
    z = pts_color[:, 2]
    valid_z = z > 1e-6
    if not np.any(valid_z):
        return np.zeros((len(points_ir), 3), dtype=np.float32)

    uu = np.zeros(len(points_ir), dtype=np.int32)
    vv = np.zeros(len(points_ir), dtype=np.int32)
    uu[valid_z] = (K_color[0, 0] * pts_color[valid_z, 0] / z[valid_z] + K_color[0, 2]).astype(np.int32)
    vv[valid_z] = (K_color[1, 1] * pts_color[valid_z, 1] / z[valid_z] + K_color[1, 2]).astype(np.int32)

    h, w = color_bgr.shape[:2]
    in_bounds = valid_z & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    colors = np.zeros((len(points_ir), 3), dtype=np.float32)
    if np.any(in_bounds):
        colors[in_bounds] = color_bgr[vv[in_bounds], uu[in_bounds], ::-1].astype(np.float32) / 255.0
    return colors


def compute_diff_metrics(cur_gray, ref_gray, pixel_diff_thresh):
    diff = cv2.absdiff(cur_gray, ref_gray)
    mean_norm = float(diff.mean() / 255.0)
    changed_ratio = float((diff >= pixel_diff_thresh).mean())
    return mean_norm, changed_ratio


def save_image_or_raise(path, image):
    if not cv2.imwrite(path, image):
        raise IOError(f"Failed to save image: {path}")


def save_point_cloud_or_raise(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if not o3d.io.write_point_cloud(path, cloud):
        raise IOError(f"Failed to save point cloud: {path}")


def save_keyframe(args, key_idx, left_proc, right_proc, color_bgr, depth, points, colors):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{key_idx:06d}_{ts}"
    left_path = os.path.join(args.out_dir, f"left_{base}.png")
    right_path = os.path.join(args.out_dir, f"right_{base}.png")
    depth_path = os.path.join(args.out_dir, f"depth_{base}.npy")
    cloud_path = os.path.join(args.out_dir, f"cloud_{base}.ply")
    save_image_or_raise(left_path, left_proc)
    save_image_or_raise(right_path, right_proc)
    if args.use_color and color_bgr is not None:
        color_path = os.path.join(args.out_dir, f"color_{base}.png")
        save_image_or_raise(color_path, color_bgr)
    else:
        color_path = None
    np.save(depth_path, depth)
    if len(points) == 0:
        raise RuntimeError("Keyframe triggered but point cloud is empty.")
    save_point_cloud_or_raise(cloud_path, points, colors)
    return {
        "left_path": left_path,
        "right_path": right_path,
        "color_path": color_path,
        "depth_path": depth_path,
        "cloud_path": cloud_path,
    }


def main():
    args = parse_args()
    set_logging_format()
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    if o3d is None:
        raise RuntimeError("open3d is required for point cloud saving.")
    if args.startup_retries < 1:
        raise ValueError("--startup_retries must be >= 1")

    runtime_state = {"pipeline": None, "vis": None}
    pcd = None
    try:
        reset_device_if_requested(args)
        model = load_model(args)
        logging.info("Model loaded: %s", args.model_dir)

        with ExitStack() as cleanup:
            cleanup.callback(cleanup_runtime, runtime_state)

            pipeline, calib = start_pipeline_with_retries(args, reason="initial startup")
            runtime_state["pipeline"] = pipeline
            rectify_maps = calib["rectify_maps"]
            K_ir = calib["K_ir"]
            baseline = calib["baseline"]
            K_color = calib["K_color"]
            R_ir_to_color = calib["R_ir_to_color"]
            T_ir_to_color = calib["T_ir_to_color"]
            logging.info("Rectification %s", "enabled" if args.rectify else "disabled")

            if args.save_intrinsic_file:
                save_runtime_k(args.save_intrinsic_file, K_ir, baseline)
                logging.info("Saved runtime intrinsic file: %s", args.save_intrinsic_file)

            csv_path = os.path.join(args.out_dir, "captures.csv")
            csv_exists = os.path.isfile(csv_path)
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            cleanup.callback(csv_file.close)
            writer = csv.writer(csv_file)
            if not csv_exists:
                writer.writerow(
                    [
                        "key_idx",
                        "timestamp",
                        "unix_time",
                        "frame_id",
                        "diff_mean",
                        "diff_ratio",
                        "num_points",
                        "left_path",
                        "right_path",
                        "color_path",
                        "depth_path",
                        "cloud_path",
                    ]
                )
                csv_file.flush()

            dummy_h = max(1, int(round(args.height * args.scale)))
            dummy_w = max(1, int(round(args.width * args.scale)))
            dummy_l = torch.randn(1, 3, dummy_h, dummy_w, device="cuda")
            dummy_r = torch.randn(1, 3, dummy_h, dummy_w, device="cuda")
            padder = InputPadder(dummy_l.shape, divis_by=32, force_square=False)
            dummy_l, dummy_r = padder.pad(dummy_l, dummy_r)
            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                _ = model.forward(
                    dummy_l,
                    dummy_r,
                    iters=args.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )
            logging.info("Model warmup done")

            if args.show_pc:
                vis = o3d.visualization.Visualizer()
                vis.create_window("FFS Keyframe Capture Point Cloud", width=1280, height=720)
                vis.get_render_option().point_size = 2.0
                vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
                pcd = o3d.geometry.PointCloud()
                vis.add_geometry(pcd)
                runtime_state["vis"] = vis

            frame_id = 0
            saved_count = 0
            last_saved_frame = None
            prev_frame = None
            last_saved_frame_id = -10**9
            last_saved_time = -10**9
            t_last_log = time.time()

            while True:
                t0 = time.time()
                frames = runtime_state["pipeline"].wait_for_frames(timeout_ms=args.frame_timeout_ms)
                left_frame = frames.get_infrared_frame(1)
                right_frame = frames.get_infrared_frame(2)
                color_frame = frames.get_color_frame() if args.use_color else None
                if not left_frame or not right_frame:
                    raise RuntimeError("Missing IR frame during capture.")
                if args.use_color and not color_frame:
                    raise RuntimeError("Missing color frame while --use_color=1.")

                left = np.asanyarray(left_frame.get_data())
                right = np.asanyarray(right_frame.get_data())
                color_bgr = (
                    np.asanyarray(color_frame.get_data())
                    if args.use_color and color_frame
                    else None
                )

                if args.rectify and rectify_maps is not None:
                    map_l1, map_l2, map_r1, map_r2 = rectify_maps
                    left = cv2.remap(left, map_l1, map_l2, interpolation=cv2.INTER_LINEAR)
                    right = cv2.remap(right, map_r1, map_r2, interpolation=cv2.INTER_LINEAR)

                if args.equalize_hist:
                    left = cv2.equalizeHist(left)
                    right = cv2.equalizeHist(right)

                left_proc, K_proc = scale_image_and_k(left, K_ir, args.scale)
                right_proc, _ = scale_image_and_k(right, K_ir, args.scale)
                h, w = left_proc.shape[:2]
                left_rgb = np.repeat(left_proc[:, :, None], 3, axis=-1)
                right_rgb = np.repeat(right_proc[:, :, None], 3, axis=-1)

                img0 = torch.as_tensor(left_rgb, device="cuda").float()[None].permute(0, 3, 1, 2)
                img1 = torch.as_tensor(right_rgb, device="cuda").float()[None].permute(0, 3, 1, 2)
                padder = InputPadder(img0.shape, divis_by=32, force_square=False)
                img0, img1 = padder.pad(img0, img1)

                with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                    disp = model.forward(
                        img0,
                        img1,
                        iters=args.valid_iters,
                        test_mode=True,
                        optimize_build_volume="pytorch1",
                    )
                disp = padder.unpad(disp.float())[0, 0].detach().cpu().numpy()
                disp = np.clip(disp, 0.0, None)

                if args.remove_invisible:
                    xx = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
                    disp[(xx - disp) < 0] = np.inf

                depth = K_proc[0, 0] * baseline / disp
                depth[(depth < args.znear) | (depth > args.zfar) | ~np.isfinite(depth)] = 0

                points_ir, valid_mask = build_point_cloud(depth, K_proc, args.pc_stride)
                if args.use_color and color_bgr is not None:
                    colors = colorize_points_from_rgb(
                        points_ir, color_bgr, K_color, R_ir_to_color, T_ir_to_color
                    )
                    if len(colors) > 0:
                        gray = left_proc[valid_mask]
                        gray_rgb = np.repeat(gray[:, None], 3, axis=1).astype(np.float32) / 255.0
                        missing = np.isclose(colors.sum(axis=1), 0.0)
                        if np.any(missing):
                            colors[missing] = gray_rgb[missing]
                else:
                    gray = left_proc[valid_mask]
                    colors = np.repeat(gray[:, None], 3, axis=1).astype(np.float32) / 255.0

                if args.show_pc:
                    pcd.points = o3d.utility.Vector3dVector(points_ir.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    runtime_state["vis"].update_geometry(pcd)
                    if not runtime_state["vis"].poll_events():
                        break
                    runtime_state["vis"].update_renderer()

                now = time.time()
                if last_saved_frame is None:
                    diff_mean = 1.0
                    diff_ratio = 1.0
                    should_save = True
                    reason = "first_frame"
                else:
                    if args.diff_reference == "previous":
                        if prev_frame is None:
                            ref_frame = last_saved_frame
                        else:
                            ref_frame = prev_frame
                    else:
                        ref_frame = last_saved_frame
                    diff_mean, diff_ratio = compute_diff_metrics(
                        left_proc, ref_frame, args.pixel_diff_thresh
                    )
                    threshold_hit = (
                        diff_mean >= args.diff_mean_thresh or diff_ratio >= args.diff_ratio_thresh
                    )
                    frame_gap_ok = (frame_id - last_saved_frame_id) >= args.min_frames_between_saves
                    time_gap_ok = (now - last_saved_time) >= args.min_seconds_between_saves
                    should_save = threshold_hit and frame_gap_ok and time_gap_ok
                    reason = "threshold" if should_save else "none"

                if should_save:
                    save_info = save_keyframe(
                        args=args,
                        key_idx=saved_count,
                        left_proc=left_proc,
                        right_proc=right_proc,
                        color_bgr=color_bgr,
                        depth=depth,
                        points=points_ir,
                        colors=colors,
                    )
                    ts_iso = datetime.now().isoformat(timespec="milliseconds")
                    writer.writerow(
                        [
                            saved_count,
                            ts_iso,
                            f"{now:.6f}",
                            frame_id,
                            f"{diff_mean:.6f}",
                            f"{diff_ratio:.6f}",
                            len(points_ir),
                            save_info["left_path"],
                            save_info["right_path"],
                            save_info["color_path"] if save_info["color_path"] else "",
                            save_info["depth_path"],
                            save_info["cloud_path"],
                        ]
                    )
                    csv_file.flush()
                    logging.info(
                        "Saved keyframe %06d (reason=%s, diff_mean=%.4f, diff_ratio=%.4f, points=%d)",
                        saved_count,
                        reason,
                        diff_mean,
                        diff_ratio,
                        len(points_ir),
                    )
                    saved_count += 1
                    last_saved_frame = left_proc.copy()
                    last_saved_time = now
                    last_saved_frame_id = frame_id
                    if args.max_keyframes > 0 and saved_count >= args.max_keyframes:
                        logging.info("Reached max_keyframes=%d. Exiting.", args.max_keyframes)
                        break

                prev_frame = left_proc.copy()

                if args.show_disp:
                    disp_vis = vis_disparity(disp, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO)
                    panel = np.concatenate([left_rgb, right_rgb, disp_vis], axis=1)
                    fps_now = 1.0 / max(time.time() - t0, 1e-6)
                    cv2.putText(
                        panel,
                        f"FPS:{fps_now:.1f} Keys:{saved_count} dM:{diff_mean:.3f} dR:{diff_ratio:.3f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("FFS Keyframe Capture (left | right | disp)", panel[:, :, ::-1])
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        save_info = save_keyframe(
                            args=args,
                            key_idx=saved_count,
                            left_proc=left_proc,
                            right_proc=right_proc,
                            color_bgr=color_bgr,
                            depth=depth,
                            points=points_ir,
                            colors=colors,
                        )
                        ts_iso = datetime.now().isoformat(timespec="milliseconds")
                        writer.writerow(
                            [
                                saved_count,
                                ts_iso,
                                f"{now:.6f}",
                                frame_id,
                                f"{diff_mean:.6f}",
                                f"{diff_ratio:.6f}",
                                len(points_ir),
                                save_info["left_path"],
                                save_info["right_path"],
                                save_info["color_path"] if save_info["color_path"] else "",
                                save_info["depth_path"],
                                save_info["cloud_path"],
                            ]
                        )
                        csv_file.flush()
                        logging.info("Saved keyframe %06d (manual)", saved_count)
                        saved_count += 1
                        last_saved_frame = left_proc.copy()
                        last_saved_time = now
                        last_saved_frame_id = frame_id

                frame_id += 1
                if frame_id % args.log_every == 0:
                    dt = time.time() - t_last_log
                    logging.info(
                        "frame=%d, avg_fps=%.2f, keyframes=%d",
                        frame_id,
                        args.log_every / max(dt, 1e-6),
                        saved_count,
                    )
                    t_last_log = time.time()

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")


if __name__ == "__main__":
    main()
