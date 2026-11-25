"""Run B1 on a dashcam video and annotate detections against a fixed ego-lane ROI."""

import argparse
from collections import Counter
from ipaddress import IPv4Address
from math import isfinite
import os
from pathlib import Path
import shutil
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter


DEFAULT_LANE = (0.45, 0.55, 0.45, 0.17, 0.83, 0.97)
STATES = ("WARNING: IN LANE", "DEBRIS OUTSIDE LANE", "NO DEBRIS DETECTED")
COLORS = ((55, 72, 255), (84, 220, 123), (210, 210, 210))


def in_ego_lane(x, y, lane):
    top_left, top_right, top_y, bottom_left, bottom_right, bottom_y = lane
    if not top_y <= y <= bottom_y:
        return False
    fraction = (y - top_y) / (bottom_y - top_y)
    left = top_left + (bottom_left - top_left) * fraction
    right = top_right + (bottom_right - top_right) * fraction
    return left <= x <= right


def draw_frame(cv2, np, frame, boxes, lane):
    height, width = frame.shape[:2]
    top_left, top_right, top_y, bottom_left, bottom_right, bottom_y = lane
    polygon = np.array([
        (round(top_left * width), round(top_y * height)),
        (round(top_right * width), round(top_y * height)),
        (round(bottom_right * width), round(bottom_y * height)),
        (round(bottom_left * width), round(bottom_y * height)),
    ], dtype=np.int32)
    lane_color = (240, 217, 68)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], lane_color)
    cv2.addWeighted(overlay, 0.17, frame, 0.83, 0, frame)
    cv2.polylines(frame, [polygon], True, lane_color, 2, cv2.LINE_AA)

    state = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(width / 550, 0.9)
    for (left, top, right, bottom), confidence in zip(
        boxes.xyxyn.cpu().tolist(), boxes.conf.cpu().tolist()
    ):
        center_x = (left + right) / 2
        inside = in_ego_lane(center_x, bottom, lane)
        state = min(state, 0 if inside else 1)
        color = COLORS[0] if inside else COLORS[1]
        cv2.rectangle(frame, (round(left * width), round(top * height)),
                      (round(right * width), round(bottom * height)), color, 2)
        cv2.circle(frame, (round(center_x * width), round(bottom * height)),
                   4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"debris {confidence:.2f}",
                    (round(left * width), max(15, round(top * height) - 5)),
                    font, scale * 0.65, color, 2, cv2.LINE_AA)

    label = STATES[state]
    (label_width, label_height), _ = cv2.getTextSize(label, font, scale, 2)
    cv2.rectangle(frame, (8, 8), (label_width + 28, label_height + 31), (18, 23, 31), -1)
    cv2.putText(frame, label, (18, label_height + 19), font, scale,
                COLORS[state], 2, cv2.LINE_AA)
    return state


def open_video(cv2, video, stage):
    capture = cv2.VideoCapture(str(video))
    if capture.isOpened():
        return capture
    capture.release()

    # Some Windows OpenCV builds cannot open paths containing Hangul.
    if not str(video).isascii():
        local_video = stage / f"input{video.suffix}"
        shutil.copyfile(video, local_video)
        capture = cv2.VideoCapture(str(local_video))
        if capture.isOpened():
            return capture
        capture.release()
    raise ValueError(f"원본 영상을 열지 못했어요. {video}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Original dashcam video")
    parser.add_argument("weights", type=Path, help="B1 YOLO .pt weights")
    parser.add_argument("--output", type=Path, default=Path("results/lane_demo_video.mp4"))
    parser.add_argument("--conf", type=float, default=0.21, help="YOLO confidence threshold")
    parser.add_argument("--class-id", type=int, default=0, help="Debris class ID in this model")
    parser.add_argument("--lane", nargs=6, type=float, default=DEFAULT_LANE,
                        metavar=("TOP_L", "TOP_R", "TOP_Y", "BOTTOM_L", "BOTTOM_R", "BOTTOM_Y"),
                        help="Fixed lane trapezoid coordinates normalized to 0–1")
    parser.add_argument("--udp-host", type=IPv4Address, help="Send warnings to this receiver IPv4 address")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP receiver port (default: 5005)")
    args = parser.parse_args()

    video, weights, output = (path.resolve() for path in (args.video, args.weights, args.output))
    if not video.is_file():
        parser.error(f"영상 파일을 찾을 수 없어요. {video}")
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        parser.error(f"B1 가중치 .pt 파일을 확인해 주세요. {weights}")
    if output.suffix.lower() != ".mp4" or output == video:
        parser.error("출력은 원본과 다른 .mp4 경로로 지정해 주세요.")
    top_left, top_right, top_y, bottom_left, bottom_right, bottom_y = args.lane
    if not (0 <= top_left < top_right <= 1 and 0 <= bottom_left < bottom_right <= 1
            and 0 <= top_y < bottom_y <= 1):
        parser.error("차로 좌표는 0~1 사이이며 왼쪽 < 오른쪽, 상단 < 하단이어야 해요.")
    if not 0 < args.conf <= 1:
        parser.error("--conf는 0보다 크고 1 이하이어야 해요.")
    if not 1 <= args.udp_port <= 65535:
        parser.error("--udp-port는 1~65535 범위에서 지정해 주세요.")

    udp_address = None
    if args.udp_host is not None:
        try:
            from udp_warning_test import send_warning
        except ModuleNotFoundError:
            parser.error("UDP 시험에 쓰던 udp_warning_test.py를 lane_demo.py와 같은 폴더에 두세요.")
        udp_address = (str(args.udp_host), args.udp_port)

    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "detect" or args.class_id not in model.names:
        parser.error("낙하물 클래스가 있는 YOLO 검출 가중치와 --class-id를 확인해 주세요.")

    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(output.parent)
    counts = Counter()
    warning_delays = []
    with TemporaryDirectory(prefix="lane_video_", dir=".") as directory:
        stage = Path(os.path.relpath(directory))
        staged_output = stage / "lane_demo_video.mp4"
        capture = open_video(cv2, video, stage)
        writer = cv2.VideoWriter()
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not isfinite(fps) or fps <= 0:
                raise ValueError("원본 영상의 FPS를 읽을 수 없어요.")
            ok, frame = capture.read()
            if not ok:
                raise ValueError("원본 영상의 첫 프레임을 읽을 수 없어요.")
            height, width = frame.shape[:2]
            if not writer.open(str(staged_output), cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (width, height)):
                raise ValueError("MP4 출력 파일을 열지 못했어요.")

            print(f"영상 판정 시작 ({width}×{height}, 원본 영상 {fps:g} FPS)", flush=True)
            previous_state = 2
            frame_number = 0
            processing_started = frame_started = perf_counter()
            while ok:
                frame_number += 1
                result = model.predict(
                    frame, classes=[args.class_id], imgsz=1280, conf=args.conf,
                    iou=0.7, max_det=300, verbose=False,
                )[0]
                state = draw_frame(cv2, np, frame, result.boxes, args.lane)
                counts[state] += 1
                if udp_address is not None and state == 0 and previous_state != 0:
                    send_warning(udp_address)
                    delay_ms = (perf_counter() - frame_started) * 1000
                    warning_delays.append(delay_ms)
                    video_seconds = (frame_number - 1) / fps
                    print(f"[차로 경고] 영상 {video_seconds:.2f}초, 프레임 {frame_number}, 송신까지 {delay_ms:.1f} ms",
                          flush=True)
                previous_state = state
                writer.write(frame)
                ok, frame = capture.read()
                frame_started = perf_counter()
            processing_seconds = perf_counter() - processing_started
        finally:
            writer.release()
            capture.release()

        if sum(counts.values()) < 2 or not staged_output.is_file() or not staged_output.stat().st_size:
            raise ValueError("영상 출력에는 읽을 수 있는 프레임이 두 개 이상 필요해요.")
        staged_output.replace(output)

    print(f"차로 판단 영상 저장 완료 → {output}")
    for state, label in enumerate(STATES):
        print(f"{label}: {counts[state]} frames")
    print(f"처리 속도: {frame_number / processing_seconds:.2f} FPS (검출·차로 표시·영상 저장 포함)")
    if udp_address is not None:
        print(f"UDP 경고 송신: {len(warning_delays)}건")
        if warning_delays:
            print(f"프레임 처리 시작부터 UDP 송신까지 중앙값: {median(warning_delays):.1f} ms")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError:
        raise SystemExit("YOLO 평가에 쓰던 .venv의 Python으로 실행해 주세요.") from None
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"차로 판단 영상 생성 중단: {error}") from None
