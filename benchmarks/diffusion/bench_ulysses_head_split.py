# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure Ulysses head splitting through the regular single-request Omni entry point.

Pass a JSON workload with engine, prompt and sampling fields. Run the native
arm first, then the fixed and auto arms with the same workload and output
directory. Each arm starts its own engine. Warm-up, RGB hashing and waveform
comparison are outside the reported warm-request times.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def media(outputs, video):
    if not video:
        image = extract_images_from_outputs(outputs)[0].convert("RGB")
        return np.asarray(image), None
    result = outputs[0]
    payload = result.images
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    audio = (result.multimodal_output or {}).get("audio")
    if isinstance(payload, tuple):
        payload, audio = payload
    elif isinstance(payload, dict):
        audio = payload.get("audio", audio)
        payload = payload.get("frames", payload.get("video"))
    if audio is None:
        raise ValueError("Expected the MiniMax-H3 T2VA waveform")
    frames = np.asarray(payload)
    if frames.dtype.kind == "f" and frames.min() >= 0 and frames.max() <= 1.01:
        frames = np.clip(np.rint(frames * 255), 0, 255).astype(np.uint8)
    return frames, np.asarray(audio, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="native", help="native, auto or a fixed ratio such as 3:5")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    case = json.loads(args.workload.read_text())
    engine_kwargs = dict(case["engine"])
    engine_kwargs.pop("ulysses_head_split", None)
    if args.policy != "native":
        engine_kwargs["ulysses_head_split"] = args.policy
    args.output.mkdir(exist_ok=True, parents=True)
    video = "video" in case["prompt"]["modalities"]
    engine = Omni(**engine_kwargs)
    rows = []
    recorded = set()
    try:
        for i in range(-1, args.rounds):
            seed = args.seeds[max(0, i) % len(args.seeds)]
            sampling = OmniDiffusionSamplingParams(
                **case["sampling"], seed=seed, generator=torch.Generator().manual_seed(seed), generator_device="cpu"
            )
            # Multimodal preprocessing replaces file paths with decoded images in place.
            prompt = copy.deepcopy(case["prompt"])
            start = time.perf_counter()
            outputs = engine.generate(prompt, sampling, use_tqdm=False)
            seconds = time.perf_counter() - start
            frames, audio = media(outputs, video)
            digest = hashlib.sha256(np.ascontiguousarray(frames)).hexdigest()
            reference = args.output / f"seed-{seed}.sha256"
            waveform = args.output / f"seed-{seed}-audio.npy"
            if args.policy == "native" and seed not in recorded:
                recorded.add(seed)
                reference.write_text(digest)
                if audio is not None:
                    np.save(waveform, audio)
            exact = reference.read_text().strip() == digest
            error = 0.0
            if audio is not None:
                expected = np.load(waveform)
                if expected.shape != audio.shape:
                    raise ValueError("Waveform shape differs from native")
                error = float(np.sqrt(np.mean(np.square(audio - expected), dtype=np.float64)))
            row = dict(round=i, seed=seed, seconds=seconds, rgb_exact=exact, audio_rmse=error)
            print(json.dumps(row), flush=True)
            rows.append(row)
    finally:
        engine.close()
    report = dict(
        policy=args.policy,
        workload=case,
        warm_median_seconds=median(r["seconds"] for r in rows[1:]),
        requests=rows,
    )
    (args.output / f"{args.policy}.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["rgb_exact"] and r["audio_rmse"] <= 1e-4 for r in rows):
        raise SystemExit("Output differs from the native quality reference")


if __name__ == "__main__":
    main()
