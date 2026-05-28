#!/usr/bin/env python3
"""Simple LiteRT inference example for ImageNet classification.

Usage:
    python litert_inference.py --image 000000039769.jpg
    python litert_inference.py --model mobilenet_v2.tflite --image path/to/image.jpg
"""

import argparse
import os
import sys

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from PIL import Image


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LITERT_TYPE_TO_NP = {
    1: np.float32,   # kLiteRtElementTypeFloat32
    9: np.int8,      # kLiteRtElementTypeInt8
    3: np.uint8,     # kLiteRtElementTypeUInt8
    2: np.int32,     # kLiteRtElementTypeInt32
}


def _infer_input_layout(model, sig: int) -> tuple[int, int, bool]:
    """Return (height, width, channels_first) from model input metadata."""
    try:
        req = model.get_input_buffer_requirements(0, sig)
    except Exception:
        return 224, 224, True
    dims = req.get("dimensions") or req.get("shape") or req.get("dims")
    if not dims:
        return 224, 224, True
    try:
        dims = [int(d) for d in dims]
    except Exception:
        return 224, 224, True
    if len(dims) == 4:
        if dims[1] == 3:
            return dims[2], dims[3], True   # NCHW
        if dims[-1] == 3:
            return dims[1], dims[2], False  # NHWC
    if len(dims) == 3:
        if dims[0] == 3:
            return dims[1], dims[2], True   # CHW
        if dims[-1] == 3:
            return dims[0], dims[1], False  # HWC
    return 224, 224, True


def preprocess_image(
    image_path: str,
    crop_height: int,
    crop_width: int,
    channels_first: bool,
) -> np.ndarray:
    """Load and preprocess an image using torchvision-style resize/crop/normalize."""
    resize_size = int(round(max(crop_height, crop_width) / 0.875))

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w < h:
        new_w, new_h = resize_size, int(round(h * resize_size / w))
    else:
        new_h, new_w = resize_size, int(round(w * resize_size / h))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = int(round((new_w - crop_width) / 2.0))
    top = int(round((new_h - crop_height) / 2.0))
    image = image.crop((left, top, left + crop_width, top + crop_height))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # HWC
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    return arr


def _pick_output_dtype(requirements: dict) -> np.dtype:
    supported = requirements.get("supported_types", [])
    for type_id in (1, 9, 3, 2):
        if type_id in supported:
            return _LITERT_TYPE_TO_NP[type_id]
    if supported:
        return _LITERT_TYPE_TO_NP.get(supported[0], np.float32)
    return np.float32


def read_output(buffer, requirements: dict) -> np.ndarray:
    output_dtype = _pick_output_dtype(requirements)
    buffer_size = requirements.get("buffer_size", 0)
    itemsize = np.dtype(output_dtype).itemsize
    num_elements = buffer_size // itemsize if itemsize else buffer_size
    if num_elements == 0:
        raise ValueError("Output buffer size is zero")
    return buffer.read(num_elements, output_dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def load_labels(synsets_path: str, metadata_path: str, output_size: int) -> list[str] | None:
    """Build ImageNet labels as 'synset human_label', matching main.py format."""
    if not os.path.exists(synsets_path) or not os.path.exists(metadata_path):
        return None

    with open(synsets_path, encoding="utf-8") as f:
        synsets = [line.strip() for line in f if line.strip()]

    # Handle models with background class (1001 outputs)
    if output_size == len(synsets) + 1:
        synsets = ["background"] + synsets
    if output_size != len(synsets):
        print(
            f"Warning: label file does not match output size {output_size}. "
            "Falling back to class indices.",
            file=sys.stderr,
        )
        return None

    metadata: dict[str, str] = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            synset, _, label = line.strip().partition("\t")
            if synset and label:
                metadata[synset] = label

    labels = []
    for synset in synsets:
        label = metadata.get(synset, synset)
        labels.append(f"{synset} {label}" if label != synset else synset)
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.path.join(os.getcwd(), "mobilenet_v2.tflite"),
        help="Path to the .tflite model file.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(os.getcwd(), "imagenet_lsvrc_2015_synsets.txt"),
        help="Path to ImageNet synset list.",
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join(os.getcwd(), "imagenet_metadata.txt"),
        help="Path to ImageNet synset-to-label metadata.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        return 1
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    model = CompiledModel.from_file(args.model)

    sig = 0
    input_height, input_width, channels_first = _infer_input_layout(model, sig)
    image_array = preprocess_image(args.image, input_height, input_width, channels_first)

    input_buffers = model.create_input_buffers(sig)
    output_buffers = model.create_output_buffers(sig)
    input_buffers[0].write(image_array)
    model.run_by_index(sig, input_buffers, output_buffers)

    out_req = model.get_output_buffer_requirements(0, sig)
    raw_output = read_output(output_buffers[0], out_req)
    probs = softmax(raw_output.reshape(-1))

    labels = load_labels(args.labels, args.metadata, probs.size)
    top_k = min(args.top_k, probs.size)
    top_indices = np.argsort(probs)[-top_k:][::-1]

    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
        print(f"{rank}: {label} ({probs[idx]:.6f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
