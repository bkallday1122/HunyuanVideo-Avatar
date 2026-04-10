"""
RunPod Serverless Handler — HunyuanVideo-Avatar

Takes a reference image + audio, produces a talking avatar video.
Supports animated/cartoon characters. 704x768 output, 25fps, ~5s clips.
Weights stored on RunPod Network Volume (/runpod-volume/).
"""

import base64
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

# Reduce CUDA allocator fragmentation on 24GB cards during long Hunyuan runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Stub torch.xpu before anything imports diffusers
import torch
if not hasattr(torch, "xpu"):
    import types
    torch.xpu = types.ModuleType("torch.xpu")
    torch.xpu.empty_cache = lambda: None

import runpod

HUNYUAN_DIR = "/app/hunyuan"
VOLUME_DIR = "/runpod-volume"
WEIGHTS_DIR = os.path.join(VOLUME_DIR, "hunyuan-weights")
FP8_CKPT = os.path.join(
    WEIGHTS_DIR, "ckpts", "hunyuan-video-t2v-720p",
    "transformers", "mp_rank_00_model_states_fp8.pt"
)
TEXT_ENCODER_2_REQUIRED_FILES = [
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/config.json"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/merges.txt"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/special_tokens_map.json"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer_config.json"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/vocab.json"),
]
LEGACY_TEXT_ENCODER_2_TOKENIZER_FILES = {
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer/merges.txt"): os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/merges.txt"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer/special_tokens_map.json"): os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/special_tokens_map.json"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer/tokenizer_config.json"): os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer_config.json"),
    os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/tokenizer/vocab.json"): os.path.join(WEIGHTS_DIR, "ckpts/text_encoder_2/vocab.json"),
}
PROFILE_PRESETS = {
    "full": {"sample_n_frames": 129, "image_size": 704, "infer_steps": 30, "cfg_scale": 7.5},
    "balanced": {"sample_n_frames": 97, "image_size": 576, "infer_steps": 24, "cfg_scale": 7.2},
    "low_vram": {"sample_n_frames": 49, "image_size": 448, "infer_steps": 18, "cfg_scale": 6.8},
    "failsafe": {"sample_n_frames": 33, "image_size": 384, "infer_steps": 15, "cfg_scale": 6.5},
}

print("[handler] Loading HunyuanVideo-Avatar...", flush=True)
_load_start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[handler] Device: {device}, CUDA: {torch.version.cuda}", flush=True)

# Download weights to network volume on first startup
def _ensure_weights():
    """Download model weights to network volume if not present."""
    FILES = [
        # Config files first (tiny, fast)
        "ckpts/hunyuan-video-t2v-720p/vae/config.json",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt",
        "ckpts/whisper-tiny/config.json",
        "ckpts/whisper-tiny/tokenizer.json",
        "ckpts/whisper-tiny/vocab.json",
        "ckpts/whisper-tiny/preprocessor_config.json",
        "ckpts/llava_llama_image/config.json",
        "ckpts/llava_llama_image/model.safetensors.index.json",
        "ckpts/llava_llama_image/tokenizer.json",
        "ckpts/llava_llama_image/tokenizer_config.json",
        "ckpts/llava_llama_image/special_tokens_map.json",
        "ckpts/llava_llama_image/preprocessor_config.json",
        "ckpts/text_encoder_2/config.json",
        "ckpts/text_encoder_2/merges.txt",
        "ckpts/text_encoder_2/special_tokens_map.json",
        "ckpts/text_encoder_2/tokenizer_config.json",
        "ckpts/text_encoder_2/vocab.json",
        # Medium files
        "ckpts/whisper-tiny/model.safetensors",
        "ckpts/det_align/detface.pt",
        "ckpts/stable_syncnet.pt",
        # Large files last
        "ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        "ckpts/text_encoder_2/model.safetensors",
        "ckpts/llava_llama_image/model-00001-of-00004.safetensors",
        "ckpts/llava_llama_image/model-00002-of-00004.safetensors",
        "ckpts/llava_llama_image/model-00003-of-00004.safetensors",
        "ckpts/llava_llama_image/model-00004-of-00004.safetensors",
        # Biggest file last (~25GB)
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
    ]
    CRITICAL_FILES = [os.path.join(WEIGHTS_DIR, fpath) for fpath in FILES]
    def _looks_like_html(f):
        if not os.path.exists(f) or os.path.getsize(f) <= 0:
            return False
        with open(f, "rb") as handle:
            head = handle.read(256).lstrip().lower()
        return head.startswith(b"<!doctype") or head.startswith(b"<html")

    def _valid(f):
        if not os.path.exists(f) or os.path.getsize(f) < 10:
            return False
        if _looks_like_html(f):
            return False
        # FP8 checkpoint must be at least 20GB (corrupted partial downloads are smaller)
        if f == FP8_CKPT and os.path.getsize(f) < 20 * 1024**3:
            return False
        if f.endswith(".json"):
            try:
                with open(f, "r", encoding="utf-8") as handle:
                    json.load(handle)
            except Exception:
                return False
        return True
    def _missing_files():
        return [f for f in CRITICAL_FILES if not _valid(f)]

    missing = _missing_files()
    if not missing:
        ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
        print(f"[handler] All weights found (FP8: {ckpt_gb:.1f}GB)", flush=True)
        return True
    print(f"[handler] Missing {len(missing)} files: {[os.path.basename(f) for f in missing]}", flush=True)

    print(f"[handler] Weights not found — downloading to {WEIGHTS_DIR}...", flush=True)
    if not os.path.exists(VOLUME_DIR):
        print(f"[handler] ERROR: Network volume not mounted at {VOLUME_DIR}", flush=True)
        return False

    try:
        import shutil
        repaired = 0
        for legacy_path, current_path in LEGACY_TEXT_ENCODER_2_TOKENIZER_FILES.items():
            if _valid(current_path):
                continue
            if _valid(legacy_path):
                os.makedirs(os.path.dirname(current_path), exist_ok=True)
                shutil.move(legacy_path, current_path)
                repaired += 1
            elif os.path.exists(legacy_path):
                print(f"[handler] Removing invalid legacy tokenizer file {legacy_path}", flush=True)
                os.remove(legacy_path)
        if repaired:
            print(f"[handler] Repaired {repaired} legacy text_encoder_2 tokenizer files", flush=True)

        missing = _missing_files()
        if not missing:
            ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
            print(f"[handler] All weights found after repair (FP8: {ckpt_gb:.1f}GB)", flush=True)
            return True

        # If the FP8 checkpoint is invalid, the weights dir is junk from failed downloads.
        if os.path.exists(WEIGHTS_DIR) and not _valid(FP8_CKPT):
            # Check actual size with du
            du = subprocess.run(["du", "-sm", WEIGHTS_DIR], capture_output=True, text=True, timeout=30)
            used_mb = int(du.stdout.split()[0]) if du.returncode == 0 else 0
            print(f"[handler] Existing weights dir: {used_mb}MB — cleaning (no valid checkpoint)...", flush=True)
            shutil.rmtree(WEIGHTS_DIR, ignore_errors=True)
            # Also clean any HF cache on volume
            hf_cache = os.path.join(VOLUME_DIR, ".cache")
            if os.path.exists(hf_cache):
                shutil.rmtree(hf_cache, ignore_errors=True)
                print(f"[handler] Cleaned HF cache too", flush=True)
        elif os.path.exists(WEIGHTS_DIR):
            missing_names = [os.path.relpath(path, WEIGHTS_DIR) for path in missing]
            print(f"[handler] Preserving existing weights dir and downloading missing files: {missing_names}", flush=True)

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        # Verify writable
        test_file = os.path.join(WEIGHTS_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"[handler] Volume writable: {WEIGHTS_DIR}", flush=True)
        print("[handler] Downloading weights via curl...", flush=True)

        BASE = "https://huggingface.co/tencent/HunyuanVideo-Avatar/resolve/main"

        for i, fpath in enumerate(FILES):
            dest = os.path.join(WEIGHTS_DIR, fpath)
            if os.path.exists(dest):
                if _valid(dest):
                    continue
                print(f"  Removing invalid existing file {fpath}", flush=True)
                os.remove(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = f"{BASE}/{fpath}"
            print(f"  [{i+1}/{len(FILES)}] {fpath}...", flush=True)
            result = subprocess.run(
                ["curl", "-L", "--retry", "3", "--retry-delay", "5",
                 "--progress-bar", "-o", dest, url],
                capture_output=True, timeout=1800,
            )
            dest_size = os.path.getsize(dest) if os.path.exists(dest) else 0
            if dest_size > 0 and not _valid(dest):
                print(f"  Got invalid file for {fpath} — removing", flush=True)
                os.remove(dest)
                dest_size = 0
            if result.returncode != 0 or dest_size < 10:
                stderr = (result.stderr or b"").decode()[-300:]
                stdout = (result.stdout or b"").decode()[-200:]
                print(f"  FAILED {fpath}: exit={result.returncode}", flush=True)
                print(f"    stderr: {stderr}", flush=True)
                print(f"    stdout: {stdout}", flush=True)
                print(f"    dest exists: {os.path.exists(dest)}, dest dir: {os.path.exists(os.path.dirname(dest))}", flush=True)
                if os.path.exists(dest):
                    os.remove(dest)

        missing = _missing_files()
        if not missing:
            ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
            print(f"[handler] Download complete: FP8 checkpoint {ckpt_gb:.1f}GB", flush=True)
            return True
        else:
            print(f"[handler] Critical files still missing after download: {[os.path.basename(f) for f in missing]}", flush=True)
            return False
    except Exception as e:
        print(f"[handler] Weight download failed: {e}", flush=True)
        return False

_weights_ok = _ensure_weights()

# Symlink weights dir so the model code finds them
weights_link = os.path.join(HUNYUAN_DIR, "weights")
if _weights_ok and not os.path.exists(weights_link):
    os.symlink(WEIGHTS_DIR, weights_link)
elif _weights_ok and os.path.isdir(weights_link) and not os.path.islink(weights_link):
    # Remove empty weights dir and symlink
    import shutil
    shutil.rmtree(weights_link)
    os.symlink(WEIGHTS_DIR, weights_link)

# Set MODEL_BASE for the inference script
os.environ["MODEL_BASE"] = WEIGHTS_DIR

# Check VRAM
vram_gb = None
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[handler] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB", flush=True)
    if vram_gb < 40:
        os.environ["CPU_OFFLOAD"] = "1"
        print(f"[handler] CPU offload enabled (VRAM < 40GB)", flush=True)

_load_elapsed = time.time() - _load_start
print(f"[handler] Ready in {_load_elapsed:.1f}s", flush=True)


def _normalize_frame_count(value, default=129):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    value = max(33, min(value, 400))
    return max(33, ((value - 1) // 4) * 4 + 1)


def _normalize_image_size(value, default=704):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    value = max(256, min(value, 704))
    return max(256, (value // 64) * 64)


def _normalize_infer_steps(value, default=30):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(8, min(value, 50))


def _normalize_cfg_scale(value, default=7.5):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(1.0, min(value, 12.0))


def _preset_profile(name):
    profile = PROFILE_PRESETS[name].copy()
    profile["name"] = name
    return profile


def _default_profile_order():
    if vram_gb is None:
        return ["balanced", "low_vram", "failsafe"]
    if vram_gb < 28:
        return ["low_vram", "failsafe"]
    if vram_gb < 40:
        return ["balanced", "low_vram", "failsafe"]
    return ["full", "balanced", "low_vram", "failsafe"]


def _is_more_conservative(candidate, baseline):
    return (
        candidate["sample_n_frames"] <= baseline["sample_n_frames"]
        and candidate["image_size"] <= baseline["image_size"]
        and candidate["infer_steps"] <= baseline["infer_steps"]
    )


def _build_attempt_profiles(job_input):
    order = _default_profile_order()
    requested_profile_name = job_input.get("quality_profile")
    if requested_profile_name not in PROFILE_PRESETS:
        requested_profile_name = order[0]

    requested_profile = _preset_profile(requested_profile_name)
    if any(key in job_input for key in ("sample_n_frames", "image_size", "infer_steps", "cfg_scale")):
        requested_profile["name"] = "custom"
        requested_profile["sample_n_frames"] = _normalize_frame_count(
            job_input.get("sample_n_frames", requested_profile["sample_n_frames"]),
            default=requested_profile["sample_n_frames"],
        )
        requested_profile["image_size"] = _normalize_image_size(
            job_input.get("image_size", requested_profile["image_size"]),
            default=requested_profile["image_size"],
        )
        requested_profile["infer_steps"] = _normalize_infer_steps(
            job_input.get("infer_steps", requested_profile["infer_steps"]),
            default=requested_profile["infer_steps"],
        )
        requested_profile["cfg_scale"] = _normalize_cfg_scale(
            job_input.get("cfg_scale", requested_profile["cfg_scale"]),
            default=requested_profile["cfg_scale"],
        )

    profiles = [requested_profile]
    if job_input.get("allow_profile_fallbacks", True):
        for name in order:
            candidate = _preset_profile(name)
            if candidate["name"] == requested_profile_name and requested_profile["name"] != "custom":
                continue
            if not _is_more_conservative(candidate, requested_profile):
                continue
            if any(
                existing["sample_n_frames"] == candidate["sample_n_frames"]
                and existing["image_size"] == candidate["image_size"]
                and existing["infer_steps"] == candidate["infer_steps"]
                for existing in profiles
            ):
                continue
            profiles.append(candidate)
    return profiles


def _resource_error(returncode, stdout_text, stderr_text):
    combined = f"{stdout_text}\n{stderr_text}".lower()
    return (
        returncode in (-9, 137)
        or "out of memory" in combined
        or "cuda error" in combined
        or "resource exhausted" in combined
        or "killed" in combined
    )


def _find_output_video(results_dir):
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith(".mp4"):
                output_path = os.path.join(root, fname)
                if os.path.getsize(output_path) >= 1000:
                    return output_path
    return None


def handler(job):
    if not _weights_ok:
        return {"error": "Model weights not available. Check network volume."}

    job_input = job["input"]
    start = time.time()

    with tempfile.TemporaryDirectory(prefix="hunyuan_") as tmpdir:
        image_path = os.path.join(tmpdir, "ref.png")
        audio_path = os.path.join(tmpdir, "audio.wav")
        csv_path = os.path.join(tmpdir, "input.csv")
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Get inputs
        image_b64 = job_input.get("image_b64")
        image_url = job_input.get("image_url")
        audio_b64 = job_input.get("audio_b64")
        audio_url = job_input.get("audio_url")
        prompt = job_input.get("prompt", "a person speaking naturally to camera")
        seed = int(job_input.get("seed", 128))

        if not audio_b64 and not audio_url:
            return {"error": "Provide audio_b64 or audio_url"}
        if not image_b64 and not image_url:
            return {"error": "Provide image_b64 or image_url"}

        # Save inputs
        try:
            import urllib.request
            if image_url:
                print(f"[hunyuan] Downloading image from URL...", flush=True)
                urllib.request.urlretrieve(image_url, image_path)
            else:
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_b64))

            if audio_url:
                print(f"[hunyuan] Downloading audio from URL...", flush=True)
                urllib.request.urlretrieve(audio_url, audio_path)
            else:
                with open(audio_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))
        except Exception as e:
            return {"error": f"Failed to get inputs: {e}"}

        img_size = os.path.getsize(image_path) / 1024
        audio_size = os.path.getsize(audio_path) / 1024 / 1024
        print(f"[hunyuan] image={img_size:.0f}KB audio={audio_size:.1f}MB", flush=True)

        # Convert audio to WAV 16kHz mono if needed
        wav_path = os.path.join(tmpdir, "audio_16k.wav")
        conv = subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", wav_path,
        ], capture_output=True, timeout=30)
        if conv.returncode == 0 and os.path.exists(wav_path):
            audio_path = wav_path

        # Write CSV input
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
            writer.writerow(["output", image_path, audio_path, prompt, "25"])

        profiles = _build_attempt_profiles(job_input)
        output_path = None
        used_profile = None
        last_error = None

        # Run inference
        try:
            for attempt_idx, profile in enumerate(profiles, start=1):
                attempt_dir = os.path.join(results_dir, f"{attempt_idx:02d}_{profile['name']}")
                os.makedirs(attempt_dir, exist_ok=True)
                cmd = [
                    sys.executable, "hymm_sp/sample_gpu_poor.py",
                    "--input", csv_path,
                    "--ckpt", FP8_CKPT,
                    "--sample-n-frames", str(profile["sample_n_frames"]),
                    "--seed", str(seed),
                    "--image-size", str(profile["image_size"]),
                    "--cfg-scale", str(profile["cfg_scale"]),
                    "--infer-steps", str(profile["infer_steps"]),
                    "--use-deepcache", "1",
                    "--flow-shift-eval-video", "5.0",
                    "--save-path", attempt_dir,
                    "--use-fp8",
                    "--infer-min",
                ]

                if os.environ.get("CPU_OFFLOAD") == "1":
                    cmd.append("--cpu-offload")

                print(
                    f"[hunyuan] Attempt {attempt_idx}/{len(profiles)} "
                    f"profile={profile['name']} frames={profile['sample_n_frames']} "
                    f"size={profile['image_size']} steps={profile['infer_steps']}",
                    flush=True,
                )
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=HUNYUAN_DIR, timeout=3600,
                    env={**os.environ, "PYTHONPATH": HUNYUAN_DIR,
                         "MODEL_BASE": WEIGHTS_DIR,
                         "DISABLE_SP": "1"},
                )
                stderr_tail = (proc.stderr or "")[-1000:]
                stdout_tail = (proc.stdout or "")[-500:]
                if proc.returncode == 0:
                    output_path = _find_output_video(attempt_dir)
                    if output_path:
                        used_profile = profile
                        break
                    last_error = f"HunyuanVideo produced no output for profile {profile['name']}"
                else:
                    print(f"[hunyuan] FAILED (exit {proc.returncode})", flush=True)
                    print(f"[hunyuan] stderr: {stderr_tail}", flush=True)
                    print(f"[hunyuan] stdout: {stdout_tail}", flush=True)
                    last_error = f"HunyuanVideo failed (exit {proc.returncode}): {(stderr_tail or stdout_tail)[-500:]}"

                if used_profile:
                    break

                if attempt_idx < len(profiles) and _resource_error(proc.returncode, stdout_tail, stderr_tail):
                    print("[hunyuan] Retrying with a lower-memory profile...", flush=True)
                    continue
                break
        except subprocess.TimeoutExpired:
            return {"error": "HunyuanVideo timed out (60 min)"}
        except Exception as e:
            return {"error": f"HunyuanVideo error: {e}"}

        if not output_path:
            return {"error": last_error or "HunyuanVideo produced no output"}

        output_size = os.path.getsize(output_path) / 1024 / 1024
        elapsed = time.time() - start
        print(
            f"[hunyuan] Done profile={used_profile['name']}: {output_size:.1f}MB in {elapsed:.1f}s",
            flush=True,
        )

        # Compress if needed for 20MB RunPod limit
        final_path = output_path
        MAX_FILE_MB = 14
        if output_size > MAX_FILE_MB:
            compressed = os.path.join(tmpdir, "output_compressed.mp4")
            crf = 26 if output_size > 40 else 23
            comp = subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
                "-c:a", "copy", compressed,
            ], capture_output=True, timeout=120)
            if comp.returncode == 0 and os.path.exists(compressed):
                comp_size = os.path.getsize(compressed) / 1024 / 1024
                print(f"[hunyuan] Compressed: {output_size:.1f}MB -> {comp_size:.1f}MB", flush=True)
                final_path = compressed

        final_size = os.path.getsize(final_path) / 1024 / 1024
        with open(final_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "video_b64": video_b64,
            "duration_sec": round(elapsed, 1),
            "output_size_mb": round(final_size, 1),
            "raw_output_size_mb": round(output_size, 1),
            "quality_profile": used_profile["name"],
            "sample_n_frames": used_profile["sample_n_frames"],
            "image_size": used_profile["image_size"],
            "infer_steps": used_profile["infer_steps"],
            "cfg_scale": used_profile["cfg_scale"],
        }


runpod.serverless.start({"handler": handler})
