"""Audio utilities: HTTP 206 range streaming, Opus 24 kbps transcoding, waveform peaks, and sample-exact edge re-cutting."""
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from labelling_tool import config

MEDIA_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".amr": "audio/amr",
}

CUT_SR = 16000
PEAK_DB = -1.0
WAVE_SR = 8000
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def safe_data_path(rel_path: str) -> Path:
    """Resolve a relative path strictly inside labelling_data/."""
    if not rel_path:
        raise HTTPException(404, "Missing audio path")
    p = (config.DATA_DIR / rel_path).resolve()
    base = config.DATA_DIR.resolve()
    if not p.is_relative_to(base) or not p.is_file():
        raise HTTPException(404, "Audio file not found")
    return p


def media_type_for(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def range_file_response(request: Request, file_path: Path, media_type: str | None = None) -> Response:
    """Serve an audio file with full HTTP 206 Partial Content support so HTML5 <audio> seeking works cleanly."""
    if not file_path.is_file():
        raise HTTPException(404, "Audio file not found")
    st = file_path.stat()
    size = st.st_size
    ctype = media_type or media_type_for(file_path)
    etag = f'"{st.st_mtime_ns:x}-{size:x}"'
    base_headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Cache-Control": "private, max-age=3600",
    }

    range_hdr = request.headers.get("range")
    if not range_hdr:
        def iter_all():
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            iter_all(),
            status_code=200,
            media_type=ctype,
            headers={**base_headers, "Content-Length": str(size)},
        )

    m = _RANGE_RE.fullmatch(range_hdr.strip())
    if not m:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    s_str, e_str = m.groups()
    if s_str == "" and e_str == "":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    if s_str == "":
        suffix = int(e_str)
        if suffix <= 0:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(s_str)
        end = int(e_str) if e_str else size - 1
        end = min(end, size - 1)

    if start >= size or end < start or size == 0:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    length = end - start + 1

    def iter_range():
        remaining = length
        with open(file_path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type=ctype,
        headers={
            **base_headers,
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        },
    )


def ensure_opus(src_path: Path, cache_key: str) -> Path:
    """Transcode src_path to ~24 kbps VoIP Opus (.ogg) cached under .opus_cache/, re-encoding if source changed."""
    config.OPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9._-]", "_", cache_key)
    st = src_path.stat()
    sig = f"{st.st_mtime_ns}_{st.st_size}"
    dst = config.OPUS_CACHE_DIR / f"{safe_key}.ogg"
    sig_file = config.OPUS_CACHE_DIR / f"{safe_key}.sig"
    if dst.is_file() and sig_file.is_file():
        try:
            if sig_file.read_text(encoding="utf-8").strip() == sig:
                return dst
        except OSError:
            pass

    tmp = config.OPUS_CACHE_DIR / f".{safe_key}.{os.getpid()}.tmp.ogg"
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-i", str(src_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "24k", "-vbr", "on", "-application", "voip",
        "-f", "ogg", str(tmp),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        err = res.stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"ffmpeg opus failed: {err or res.returncode}")
    os.replace(tmp, dst)
    sig_file.write_text(sig, encoding="utf-8")
    return dst


def invalidate_opus_cache(cache_key: str):
    safe_key = re.sub(r"[^A-Za-z0-9._-]", "_", cache_key)
    for ext in (".ogg", ".sig"):
        (config.OPUS_CACHE_DIR / f"{safe_key}{ext}").unlink(missing_ok=True)


def compute_waveform(call_id: str, audio_path: Path, bins: int = 900) -> dict:
    """Compute and cache normalized RMS peak envelope + peak amplitude for the call waveform canvas."""
    config.WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", call_id)
    cache_path = config.WAVEFORM_CACHE_DIR / f"{safe_id}.json"
    st = audio_path.stat()
    sig = f"{st.st_mtime_ns}:{st.st_size}:{bins}"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("sig") == sig:
                return cached
        except (OSError, ValueError):
            pass

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", str(WAVE_SR), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"ffmpeg waveform failed: {err or proc.returncode}")

    samples = np.frombuffer(proc.stdout, np.float32)
    n = len(samples)
    duration_s = round(n / WAVE_SR, 3)
    if n == 0:
        out = {"sig": sig, "call_id": call_id, "duration_s": 0.0, "peak_amp": 0.0, "peaks": [0.0] * bins}
    else:
        peak_amp = float(np.abs(samples).max())
        actual_bins = min(bins, n)
        step = n / actual_bins
        rms_vals = []
        for i in range(actual_bins):
            s0 = int(i * step)
            s1 = max(s0 + 1, int((i + 1) * step))
            seg = samples[s0:s1]
            # Mix RMS and peak slightly so short plosives & pauses both show clearly
            rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
            mx = float(np.abs(seg).max())
            rms_vals.append(0.65 * rms + 0.35 * mx)
        mx_all = max(rms_vals) or 1.0
        peaks = [round(v / mx_all, 4) for v in rms_vals]
        out = {
            "sig": sig,
            "call_id": call_id,
            "duration_s": duration_s,
            "peak_amp": peak_amp,
            "peaks": peaks,
        }

    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, cache_path)
    return out


def decode_16k_mono(audio_path: Path, start_s: float | None = None, end_s: float | None = None) -> np.ndarray:
    """Decode audio (or exact window via seek) to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1"]
    if start_s is not None and start_s > 2.0:
        pre = max(0.0, start_s - 2.0)
        rem = start_s - pre
        cmd += ["-ss", f"{pre:.4f}", "-i", str(audio_path), "-ss", f"{rem:.4f}"]
    else:
        cmd += ["-i", str(audio_path)]
        if start_s is not None and start_s > 0:
            cmd += ["-ss", f"{start_s:.4f}"]
    if end_s is not None:
        dur = max(0.05, end_s - (start_s or 0.0))
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(CUT_SR), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"ffmpeg decode failed: {err or res.returncode}")
    return np.frombuffer(res.stdout, np.float32)


def recut_piece_audio(
    call_audio_path: Path,
    start_s: float,
    end_s: float,
    dst_flac_path: Path,
    call_duration_s: float = 0.0,
    peak_amp: float | None = None,
) -> Path:
    """Cut [start_s, end_s] from full call audio to 16 kHz PCM_16 FLAC normalized to -1 dB peak.
    Matches audio_dataset_collector/segment_audio.py output format.
    """
    dst_flac_path.parent.mkdir(parents=True, exist_ok=True)
    if call_duration_s <= 1800.0:
        # Exact sample slicing avoids any VBR seek variance on normal call lengths
        full = decode_16k_mono(call_audio_path)
        s0 = max(0, int(round(start_s * CUT_SR)))
        s1 = min(len(full), int(round(end_s * CUT_SR)))
        if s1 <= s0:
            raise ValueError(f"Empty audio slice [{start_s}, {end_s}]")
        seg = full[s0:s1]
        peak = float(np.abs(full).max()) or float(peak_amp or 0.0)
    else:
        seg = decode_16k_mono(call_audio_path, start_s=start_s, end_s=end_s)
        peak = float(peak_amp or 0.0) or float(np.abs(seg).max())

    if len(seg) < CUT_SR // 10:
        raise ValueError(f"Cut piece too short ({len(seg) / CUT_SR:.2f}s)")
    gain = 10 ** (PEAK_DB / 20) / peak if peak > 0 else 1.0
    normalized = np.clip(seg * gain, -1.0, 1.0)
    tmp = dst_flac_path.parent / (dst_flac_path.name + ".part.flac")
    sf.write(str(tmp), normalized, CUT_SR, format="FLAC", subtype="PCM_16")
    os.replace(tmp, dst_flac_path)
    return dst_flac_path
