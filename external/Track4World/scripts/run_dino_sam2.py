import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from PIL import Image
from supervision.draw.color import ColorPalette
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Add the project submodule root to Python path.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "submodules")
sys.path.append(os.path.abspath(PROJECT_ROOT))
sys.path.insert(0, "submodules")

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.supervision_utils import CUSTOM_COLOR_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a video or frame directory with Grounded SAM 2")
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the input video file (mp4, avi) or directory of frames",
    )
    parser.add_argument(
        "--text-prompt",
        required=True,
        help="Text prompts for detection, e.g., 'car. person. dog.'",
    )
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="output_single_video", help="Directory to save results")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process")
    parser.add_argument(
        "--tracking-step",
        type=int,
        default=0,
        help="If > 0 and --video-path is a frame directory, seed SAM2 on every Nth frame and propagate masks",
    )
    parser.add_argument(
        "--reverse-tracking",
        action="store_true",
        help="When using --tracking-step, also propagate backward from later keyframes",
    )
    parser.add_argument(
        "--per-frame-refresh-labels",
        default="",
        help="Comma-separated labels to refresh with per-frame detections after temporal propagation, e.g. 'cup,saucer'",
    )
    parser.add_argument(
        "--keep-labels",
        default="",
        help="Comma-separated labels to keep, others will be ignored, e.g. 'saucer'",
    )
    return parser.parse_args()


ARGS = parse_args()
GROUNDING_MODEL = ARGS.grounding_model
VIDEO_PATH = ARGS.video_path
TEXT_PROMPT = ARGS.text_prompt
SAM2_CHECKPOINT = ARGS.sam2_checkpoint
SAM2_MODEL_CONFIG = ARGS.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not ARGS.force_cpu else "cpu"
OUTPUT_DIR = Path(ARGS.output_dir)

OUTPUT_PATH_VIS = OUTPUT_DIR / "vis"
OUTPUT_PATH_MASK = OUTPUT_DIR / "mask"
OUTPUT_PATH_VIS.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH_MASK.mkdir(parents=True, exist_ok=True)

if DEVICE == "cuda":
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

print(f"Loading models on {DEVICE}...")
SAM2_MODEL = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
SAM2_PREDICTOR = SAM2ImagePredictor(SAM2_MODEL)
VIDEO_PREDICTOR = None
if ARGS.tracking_step > 0:
    if not os.path.isdir(VIDEO_PATH):
        raise ValueError("--tracking-step requires --video-path to be a directory of frames")
    if DEVICE != "cuda":
        raise ValueError("--tracking-step currently requires CUDA")
    VIDEO_PREDICTOR = build_sam2_video_predictor(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)

PROCESSOR = AutoProcessor.from_pretrained(GROUNDING_MODEL)
GROUNDING_MODEL_OBJ = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)


def id_to_colors(idx: int) -> np.ndarray:
    rgb = np.zeros((3,), dtype=np.uint8)
    for i in range(3):
        rgb[i] = idx % 256
        idx //= 256
    return rgb


IDX_TO_ID = [i for i in range(256 * 256 * 256)]
np.random.shuffle(IDX_TO_ID)


def canonicalize_label(label: str) -> str:
    label = label.strip().lower()
    if "hand" in label:
        return "hand"
    if "forearm" in label or "arm" in label or "wrist" in label:
        return "arm"
    if "person" in label or "human" in label:
        return "person"
    if "saucer" in label:
        return "saucer"
    if "cup" in label:
        return "cup"
    return label


KEEP_LABELS = {
    canonicalize_label(label)
    for label in ARGS.keep_labels.split(",")
    if label.strip()
}
REFRESH_LABELS = {
    canonicalize_label(label)
    for label in ARGS.per_frame_refresh_labels.split(",")
    if label.strip()
}


def compute_bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def get_frame_records(source_path: str) -> list[tuple[str, np.ndarray, Image.Image]]:
    records = []
    if os.path.isdir(source_path):
        print(f"Processing directory: {source_path}")
        files = sorted(
            f for f in os.listdir(source_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        for filename in files:
            full_path = os.path.join(source_path, filename)
            image_pil = Image.open(full_path).convert("RGB")
            records.append((filename, np.array(image_pil), image_pil))
    else:
        print(f"Processing video file: {source_path}")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Loading Frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                records.append((f"{frame_idx:05d}.jpg", frame_rgb, image_pil))
                frame_idx += 1
                pbar.update(1)
        cap.release()
    if ARGS.max_frames != -1:
        records = records[: ARGS.max_frames]
    return records


def detect_and_segment(image: np.ndarray, image_pil: Image.Image) -> list[dict]:
    SAM2_PREDICTOR.set_image(image)
    inputs = PROCESSOR(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = GROUNDING_MODEL_OBJ(**inputs)

    results = PROCESSOR.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=ARGS.box_threshold,
        text_threshold=ARGS.text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    if input_boxes.shape[0] == 0:
        return []

    masks, _, _ = SAM2_PREDICTOR.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = results[0]["scores"].cpu().numpy()
    class_names = [canonicalize_label(label) for label in results[0]["labels"]]

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(input_boxes).float(),
        torch.from_numpy(confidences).float(),
        ARGS.nms_threshold,
    ).cpu().numpy().tolist()

    detections = []
    for idx in nms_idx:
        label = class_names[idx]
        if KEEP_LABELS and label not in KEEP_LABELS:
            continue

        detections.append(
            {
                "label": label,
                "score": float(confidences[idx]),
                "box": input_boxes[idx].astype(np.float32),
                "mask": masks[idx].astype(bool),
            }
        )
    return detections


def merge_detection(frame_detections: dict[int, dict], candidate: dict, iou_threshold: float = 0.6) -> None:
    for existing in frame_detections.values():
        if existing["label"] != candidate["label"]:
            continue
        if mask_iou(existing["mask"], candidate["mask"]) >= iou_threshold:
            if candidate["score"] >= existing["score"]:
                existing.update(candidate)
            return
    frame_detections[candidate["obj_id"]] = candidate


def save_frame_outputs(image_file: str, image: np.ndarray, detections: list[dict]) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not detections:
        cv2.imwrite(str(OUTPUT_PATH_VIS / image_file), image_bgr)
        png_name = os.path.splitext(image_file)[0] + ".png"
        json_name = os.path.splitext(image_file)[0] + ".json"
        color_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_PATH_MASK / png_name), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        with open(OUTPUT_PATH_MASK / json_name, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    xyxy = np.stack([det["box"] for det in detections], axis=0)
    masks = np.stack([det["mask"] for det in detections], axis=0)
    confidences = np.array([det["score"] for det in detections], dtype=np.float32)
    class_ids = np.arange(len(detections))
    labels = [f'{det["label"]} {det["score"]:.2f}' for det in detections]

    supervision_detections = sv.Detections(
        xyxy=xyxy,
        mask=masks,
        class_id=class_ids,
        confidence=confidences,
    )
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=image_bgr.copy(), detections=supervision_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=supervision_detections,
        labels=labels,
    )
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=supervision_detections)

    color_mask = np.zeros(image.shape, dtype=np.uint8)
    obj_info_json = []
    mask_sizes = [int(det["mask"].sum()) for det in detections]
    sorted_mask_idx = np.argsort(mask_sizes)[::-1]
    for rank in sorted_mask_idx:
        det = detections[rank]
        color_mask[det["mask"]] = id_to_colors(IDX_TO_ID[rank])
        obj_info_json.append(
            {
                "id": IDX_TO_ID[rank],
                "label": det["label"],
                "score": float(det["score"]),
            }
        )

    png_name = os.path.splitext(image_file)[0] + ".png"
    json_name = os.path.splitext(image_file)[0] + ".json"
    cv2.imwrite(str(OUTPUT_PATH_VIS / image_file), annotated_frame)
    cv2.imwrite(str(OUTPUT_PATH_MASK / png_name), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    with open(OUTPUT_PATH_MASK / json_name, "w", encoding="utf-8") as f:
        json.dump(obj_info_json, f)


def run_per_frame(records: list[tuple[str, np.ndarray, Image.Image]]) -> None:
    iterator = tqdm(records, desc="Processing Frames") if os.path.isdir(VIDEO_PATH) else records
    for image_file, image, image_pil in iterator:
        detections = detect_and_segment(image, image_pil)
        save_frame_outputs(image_file, image, detections)


def run_temporal(records: list[tuple[str, np.ndarray, Image.Image]]) -> None:
    if VIDEO_PREDICTOR is None:
        raise RuntimeError("Temporal mode requires the video predictor")

    inference_state = VIDEO_PREDICTOR.init_state(video_path=VIDEO_PATH)
    frame_detections: dict[int, dict[int, dict]] = {idx: {} for idx in range(len(records))}
    keyframe_ids: list[int] = []
    keyframe_objects: dict[int, list[dict]] = {}
    next_obj_id = 1

    step = max(1, ARGS.tracking_step)
    for start_idx in tqdm(range(0, len(records), step), desc="Tracking Keyframes"):
        image_file, image, image_pil = records[start_idx]
        detections = detect_and_segment(image, image_pil)
        if not detections:
            continue
        keyframe_ids.append(start_idx)
        VIDEO_PREDICTOR.reset_state(inference_state)
        seeded = []
        for det in detections:
            det = det.copy()
            det["obj_id"] = next_obj_id
            next_obj_id += 1
            seeded.append(det)
            VIDEO_PREDICTOR.add_new_mask(inference_state, start_idx, det["obj_id"], det["mask"])
        keyframe_objects[start_idx] = seeded

        for out_frame_idx, out_obj_ids, out_mask_logits in VIDEO_PREDICTOR.propagate_in_video(
            inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_idx,
        ):
            if out_frame_idx >= len(records):
                continue
            meta_by_id = {det["obj_id"]: det for det in seeded}
            for mask_idx, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[mask_idx] > 0.0).cpu().numpy()[0]
                if mask.sum() == 0:
                    continue
                meta = meta_by_id[int(obj_id)]
                candidate = {
                    "obj_id": int(obj_id),
                    "label": meta["label"],
                    "score": meta["score"],
                    "mask": mask.astype(bool),
                    "box": compute_bbox_from_mask(mask),
                }
                merge_detection(frame_detections[out_frame_idx], candidate)

    if ARGS.reverse_tracking:
        for start_idx in tqdm(keyframe_ids[1:], desc="Reverse Tracking"):
            seeded = keyframe_objects[start_idx]
            VIDEO_PREDICTOR.reset_state(inference_state)
            for det in seeded:
                VIDEO_PREDICTOR.add_new_mask(inference_state, start_idx, det["obj_id"], det["mask"])
            for out_frame_idx, out_obj_ids, out_mask_logits in VIDEO_PREDICTOR.propagate_in_video(
                inference_state,
                max_frame_num_to_track=step,
                start_frame_idx=start_idx,
                reverse=True,
            ):
                if out_frame_idx < 0 or out_frame_idx >= len(records):
                    continue
                meta_by_id = {det["obj_id"]: det for det in seeded}
                for mask_idx, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[mask_idx] > 0.0).cpu().numpy()[0]
                    if mask.sum() == 0:
                        continue
                    meta = meta_by_id[int(obj_id)]
                    candidate = {
                        "obj_id": int(obj_id),
                        "label": meta["label"],
                        "score": meta["score"],
                        "mask": mask.astype(bool),
                        "box": compute_bbox_from_mask(mask),
                    }
                    merge_detection(frame_detections[out_frame_idx], candidate)

    if REFRESH_LABELS:
        for idx, (_, image, image_pil) in enumerate(tqdm(records, desc="Refreshing Labels")):
            refresh_detections = detect_and_segment(image, image_pil)
            for det in refresh_detections:
                if det["label"] not in REFRESH_LABELS:
                    continue
                det = det.copy()
                det["obj_id"] = next_obj_id
                next_obj_id += 1
                merge_detection(frame_detections[idx], det)

    for idx, (image_file, image, _) in enumerate(tqdm(records, desc="Saving Frames")):
        save_frame_outputs(image_file, image, list(frame_detections[idx].values()))


def main() -> None:
    print(f"Starting inference with prompt: '{TEXT_PROMPT}'")
    records = get_frame_records(VIDEO_PATH)
    if ARGS.tracking_step > 0:
        run_temporal(records)
    else:
        run_per_frame(records)


if __name__ == "__main__":
    main()
