"""
RunPod Serverless Handler — HunyuanVideo-Avatar

Takes a reference image + audio, produces a talking avatar video.
Supports animated/cartoon characters. 704x768 output, 25fps, ~5s clips.
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
FP8_CKPT = os.path.join(
    HUNYUAN_DIR, "weights", "ckpts", "hunyuan-video-t2v-720p",
    "transformers", "mp_rank_00_model_states_fp8.pt"
)

print("[handler] Loading HunyuanVideo-Avatar...", flush=True)
_load_start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[handler] Device: {device}, CUDA: {torch.version.cuda}", flush=True)

# Check weights
if os.path.exists(FP8_CKPT):
    ckpt_gb = os.path.getsize(FP8_CKPT) / 1024 / 1024 / 1024
    print(f"[handler] FP8 checkpoint: {ckpt_gb:.1f}GB", flush=True)
else:
    print(f"[handler] WARNING: FP8 checkpoint not found at {FP8_CKPT}", flush=True)
    weights_dir = os.path.join(HUNYUAN_DIR, "weights")
    if os.path.exists(weights_dir):
        for root, dirs, files in os.walk(weights_dir):
            for f in files:
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp) / 1024 / 1024
                if sz > 10:
                    print(f"  {fp} ({sz:.0f}MB)", flush=True)

# Check VRAM
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"[handler] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB", flush=True)
    # Enable CPU offload for GPUs under 40GB
    if vram_gb < 40:
        os.environ["CPU_OFFLOAD"] = "1"
        print(f"[handler] CPU offload enabled (VRAM < 40GB)", flush=True)

_load_elapsed = time.time() - _load_start
print(f"[handler] Ready in {_load_elapsed:.1f}s", flush=True)


def handler(job):
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

            # Add CPU offload for smaller GPUs
            if os.environ.get("CPU_OFFLOAD") == "1":
                cmd.append("--cpu-offload")

            print(f"[hunyuan] Running inference ({infer_steps} steps)...", flush=True)
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=HUNYUAN_DIR, timeout=3600,
                env={**os.environ, "PYTHONPATH": HUNYUAN_DIR,
                     "MODEL_BASE": os.path.join(HUNYUAN_DIR, "weights"),
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
