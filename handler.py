"""
RunPod Serverless Handler — HunyuanVideo-Avatar

Takes a reference image + audio, produces a talking avatar video.
Supports animated/cartoon characters. 704x768 output, 25fps, ~5s clips.
Weights stored on RunPod Network Volume (/runpod-volume/).
"""

import base64
import csv
import os
import subprocess
import sys
import tempfile
import time

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

print("[handler] Loading HunyuanVideo-Avatar...", flush=True)
_load_start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[handler] Device: {device}, CUDA: {torch.version.cuda}", flush=True)

# Download weights to network volume on first startup
def _ensure_weights():
    """Download model weights to network volume if not present."""
    if os.path.exists(FP8_CKPT):
        ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
        print(f"[handler] FP8 checkpoint found: {ckpt_gb:.1f}GB", flush=True)
        return True

    print(f"[handler] Weights not found — downloading to {WEIGHTS_DIR}...", flush=True)
    if not os.path.exists(VOLUME_DIR):
        print(f"[handler] ERROR: Network volume not mounted at {VOLUME_DIR}", flush=True)
        return False

    try:
        # If weights dir exists but FP8 checkpoint is missing, it's junk from failed downloads
        import shutil
        if os.path.exists(WEIGHTS_DIR):
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

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        # Verify writable
        test_file = os.path.join(WEIGHTS_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"[handler] Volume writable: {WEIGHTS_DIR}", flush=True)
        print("[handler] Downloading weights via curl...", flush=True)

        BASE = "https://huggingface.co/tencent/HunyuanVideo-Avatar/resolve/main"
        # Download small config files FIRST, then large model files
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
            "ckpts/text_encoder_2/tokenizer/merges.txt",
            "ckpts/text_encoder_2/tokenizer/special_tokens_map.json",
            "ckpts/text_encoder_2/tokenizer/tokenizer_config.json",
            "ckpts/text_encoder_2/tokenizer/vocab.json",
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

        for i, fpath in enumerate(FILES):
            dest = os.path.join(WEIGHTS_DIR, fpath)
            if os.path.exists(dest) and os.path.getsize(dest) > 100:
                continue  # Already downloaded
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = f"{BASE}/{fpath}"
            print(f"  [{i+1}/{len(FILES)}] {fpath}...", flush=True)
            result = subprocess.run(
                ["curl", "-L", "--retry", "3", "--retry-delay", "5",
                 "--progress-bar", "-o", dest, url],
                capture_output=True, timeout=1800,
            )
            dest_size = os.path.getsize(dest) if os.path.exists(dest) else 0
            # Check for HTML error pages (HuggingFace returns HTML on errors)
            if dest_size > 0 and dest_size < 5000 and fpath.endswith(('.pt', '.safetensors')):
                with open(dest, 'rb') as check:
                    head = check.read(20)
                if b'<!DOCTYPE' in head or b'<html' in head:
                    print(f"  Got HTML instead of model file for {fpath} — removing", flush=True)
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

        if os.path.exists(FP8_CKPT) and os.path.getsize(FP8_CKPT) > 1000000:
            ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
            print(f"[handler] Download complete: FP8 checkpoint {ckpt_gb:.1f}GB", flush=True)
            return True
        else:
            print(f"[handler] FP8 checkpoint missing or too small after download!", flush=True)
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
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[handler] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB", flush=True)
    if vram_gb < 40:
        os.environ["CPU_OFFLOAD"] = "1"
        print(f"[handler] CPU offload enabled (VRAM < 40GB)", flush=True)

_load_elapsed = time.time() - _load_start
print(f"[handler] Ready in {_load_elapsed:.1f}s", flush=True)


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
        infer_steps = int(job_input.get("infer_steps", 30))
        seed = int(job_input.get("seed", 128))
        cfg_scale = float(job_input.get("cfg_scale", 7.5))

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

        # Run inference
        try:
            cmd = [
                sys.executable, "hymm_sp/sample_gpu_poor.py",
                "--input", csv_path,
                "--ckpt", FP8_CKPT,
                "--sample-n-frames", "129",
                "--seed", str(seed),
                "--image-size", "704",
                "--cfg-scale", str(cfg_scale),
                "--infer-steps", str(infer_steps),
                "--use-deepcache", "1",
                "--flow-shift-eval-video", "5.0",
                "--save-path", results_dir,
                "--use-fp8",
                "--infer-min",
            ]

            if os.environ.get("CPU_OFFLOAD") == "1":
                cmd.append("--cpu-offload")

            print(f"[hunyuan] Running inference ({infer_steps} steps)...", flush=True)
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=HUNYUAN_DIR, timeout=3600,
                env={**os.environ, "PYTHONPATH": HUNYUAN_DIR,
                     "MODEL_BASE": WEIGHTS_DIR,
                     "DISABLE_SP": "1"},
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-1000:]
                stdout_tail = (proc.stdout or "")[-500:]
                print(f"[hunyuan] FAILED (exit {proc.returncode})", flush=True)
                print(f"[hunyuan] stderr: {stderr_tail}", flush=True)
                print(f"[hunyuan] stdout: {stdout_tail}", flush=True)
                return {"error": f"HunyuanVideo failed (exit {proc.returncode}): {stderr_tail[-500:]}"}
        except subprocess.TimeoutExpired:
            return {"error": "HunyuanVideo timed out (60 min)"}
        except Exception as e:
            return {"error": f"HunyuanVideo error: {e}"}

        # Find output video
        output_path = None
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f.endswith(".mp4"):
                    output_path = os.path.join(root, f)
                    break
            if output_path:
                break

        if not output_path or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            return {"error": "HunyuanVideo produced no output"}

        output_size = os.path.getsize(output_path) / 1024 / 1024
        elapsed = time.time() - start
        print(f"[hunyuan] Done: {output_size:.1f}MB in {elapsed:.1f}s", flush=True)

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

        with open(final_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "video_b64": video_b64,
            "duration_sec": round(elapsed, 1),
            "output_size_mb": round(output_size, 1),
        }


runpod.serverless.start({"handler": handler})
