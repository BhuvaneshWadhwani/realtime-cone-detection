"""
RT-DETRv4-S TensorRT showcase — runs inference and writes annotated video.
Uses the FP16 TensorRT engine for maximum performance.

Usage:
    python3 predict_video_trt_showcase.py
"""
from pathlib import Path
import time
import csv
import threading
import queue

import cv2
import numpy as np
import tensorrt as trt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RTDETR_ROOT  = PROJECT_ROOT / "RT-DETRv4"

RUN_NAME       = "rtdetrv4_s_fsoco_trt"
ENGINE_PATH    = RTDETR_ROOT / "outputs" / "rtv4_hgnetv2_s_fsoco" / "best_stg2_fp16.trt"
VIDEO_PATH     = PROJECT_ROOT / "videos" / "Formula Student Germany Driverless Skidpad.mp4"
OUTPUT_DIR     = RTDETR_ROOT / "outputs" / "rtv4_hgnetv2_s_fsoco" / "video_predictions"
OUTPUT_VIDEO   = OUTPUT_DIR / f"{RUN_NAME}_skidpad_showcase.mp4"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{RUN_NAME}_showcase_summary.csv"

CONF          = 0.40
MAX_FRAMES    = 0    # 0 = full video
READER_QUEUE  = 16
WRITER_QUEUE  = 16

CLASS_NAMES = {
    0: "yellow_cone",
    1: "blue_cone",
    2: "orange_cone",
    3: "large_orange_cone",
    4: "unknown_cone",
}


# ---------------------------------------------------------------------------
# TensorRT engine
# ---------------------------------------------------------------------------

def load_engine(engine_path: Path):
    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
    print(f"Engine loaded: {engine_path.name}")
    return engine, runtime


class TRTInferencer:
    def __init__(self, engine, device: str = "cuda"):
        self.engine  = engine
        self.context = engine.create_execution_context()
        self.device  = device
        self.tensors = {}
        for i in range(engine.num_io_tensors):
            name  = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = engine.get_tensor_shape(name)
            shape = tuple(1 if s == -1 else s for s in shape)
            tensor = torch.zeros(shape, dtype=torch.from_numpy(
                np.empty(0, dtype=dtype)).dtype, device=device)
            self.tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def infer(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        self.tensors["images"].copy_(images)
        self.tensors["orig_target_sizes"].copy_(orig_target_sizes)
        self.context.set_input_shape("images", tuple(images.shape))
        self.context.set_input_shape("orig_target_sizes", tuple(orig_target_sizes.shape))
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        return self.tensors["labels"], self.tensors["boxes"], self.tensors["scores"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(frame_rgb).to(device, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(640, 640), mode="nearest")
    t = t / 255.0
    return t.half()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_predictions(frame_bgr, labels, boxes, scores, conf):
    for label, box, score in zip(labels, boxes, scores):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in box]
        name    = CLASS_NAMES.get(int(label), str(int(label)))
        caption = f"{name} {score:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame_bgr, caption, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
        )
    return frame_bgr


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def reader_thread_fn(cap, q: queue.Queue, max_frames: int):
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and idx >= max_frames):
            q.put(None)
            break
        q.put(frame)
        idx += 1


def writer_thread_fn(writer, q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        writer.write(item)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RT-DETRv4-S  |  TensorRT FP16 Showcase (video output)")
    print("=" * 60)

    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"TRT engine not found: {ENGINE_PATH}\n"
                                "Run build_trt_engine.py first.")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    device = "cuda"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine, runtime = load_engine(ENGINE_PATH)
    inferencer = TRTInferencer(engine, device)

    print("Warming up (5 iters)…", end=" ", flush=True)
    dummy       = torch.zeros(1, 3, 640, 640, dtype=torch.float16, device=device)
    dummy_sizes = torch.tensor([[640, 640]], dtype=torch.int64, device=device)
    for _ in range(5):
        inferencer.infer(dummy, dummy_sizes)
    torch.cuda.synchronize()
    print("done")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = frame_count / video_fps if video_fps > 0 else 0

    print(f"Video     : {width}x{height}  {video_fps:.2f} fps  {frame_count} frames  {duration:.1f}s")
    print(f"Output    : {OUTPUT_VIDEO}")

    writer_cv = cv2.VideoWriter(
        str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
    )

    orig_size = torch.tensor([[width, height]], dtype=torch.int64, device=device)

    read_q  = queue.Queue(maxsize=READER_QUEUE)
    write_q = queue.Queue(maxsize=WRITER_QUEUE)

    t_reader = threading.Thread(target=reader_thread_fn, args=(cap, read_q, MAX_FRAMES), daemon=True)
    t_writer = threading.Thread(target=writer_thread_fn, args=(writer_cv, write_q), daemon=True)
    t_reader.start()
    t_writer.start()

    processed_frames = 0
    total_infer_time = 0.0
    t0 = time.time()

    print("\nRunning inference…")
    print("-" * 60)

    with torch.no_grad():
        while True:
            frame_bgr = read_q.get()
            if frame_bgr is None:
                break

            image_tensor = preprocess(frame_bgr, device)

            t_inf = time.perf_counter()
            labels, boxes, scores = inferencer.infer(image_tensor, orig_size)
            total_infer_time += time.perf_counter() - t_inf

            labels = labels[0].cpu().numpy()
            boxes  = boxes[0].cpu().numpy()
            scores = scores[0].cpu().numpy()

            frame_bgr = draw_predictions(frame_bgr, labels, boxes, scores, CONF)
            write_q.put(frame_bgr)

            processed_frames += 1
            if processed_frames % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {processed_frames:5d} / {frame_count}  |  "
                      f"eff {processed_frames/elapsed:.1f} fps")

    write_q.put(None)
    t_writer.join()
    cap.release()
    writer_cv.release()

    total_elapsed = time.time() - t0
    spf  = total_infer_time / processed_frames if processed_frames else 0
    mfps = 1.0 / spf if spf > 0 else 0
    efps = processed_frames / total_elapsed if total_elapsed > 0 else 0

    print("\nDone")
    print("=" * 60)
    print(f"Frames processed   : {processed_frames}")
    print(f"Total time         : {total_elapsed:.2f}s")
    print(f"Infer ms / frame   : {spf*1000:.2f}")
    print(f"Infer FPS          : {mfps:.1f}")
    print(f"Effective FPS      : {efps:.1f}   ← report this one")
    print(f"Output video       : {OUTPUT_VIDEO}")
    print(f"Summary CSV        : {OUTPUT_SUMMARY}")

    with open(OUTPUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "run_name", "engine", "video", "output_video", "frames_processed",
            "video_fps", "video_duration_seconds", "total_elapsed_seconds",
            "infer_seconds_per_frame", "infer_fps", "effective_fps",
            "conf", "device",
        ])
        w.writeheader()
        w.writerow({
            "run_name": RUN_NAME, "engine": str(ENGINE_PATH), "video": str(VIDEO_PATH),
            "output_video": str(OUTPUT_VIDEO), "frames_processed": processed_frames,
            "video_fps": video_fps, "video_duration_seconds": duration,
            "total_elapsed_seconds": total_elapsed,
            "infer_seconds_per_frame": spf, "infer_fps": mfps, "effective_fps": efps,
            "conf": CONF, "device": device,
        })


if __name__ == "__main__":
    main()
