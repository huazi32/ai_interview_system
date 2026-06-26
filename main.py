import os
import gc
import json
import base64
os.environ['GLOG_minloglevel'] = '2'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
import logging

logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import metrics
from pathlib import Path
import socket
import time

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
import subprocess
import uuid
import inspect
import re
import math

# ====================== 文件夹创建 ======================
UPLOAD_FOLDER = Path("uploads")
AUDIO_FOLDER = Path("audio_files")
UPLOAD_FOLDER.mkdir(exist_ok=True)
AUDIO_FOLDER.mkdir(exist_ok=True)


# ====================== 工具函数 ======================
def is_port_in_use(port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        res = s.connect_ex(("127.0.0.1", port))
        s.close()
        return res == 0
    except:
        return False


def free_port(port: int):
    """释放占用端口的旧进程。"""
    try:
        ps_cmd = (
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
            f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/PID", pid],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        if pids:
            time.sleep(1)
    except Exception:
        pass


def pick_available_port(preferred: int, scan_end: int = 28089) -> int:
    """优先 preferred，被占用则依次尝试后续端口。"""
    free_port(preferred)
    if not is_port_in_use(preferred):
        return preferred
    for p in range(preferred + 1, scan_end + 1):
        if not is_port_in_use(p):
            return p
    return preferred


# ====================== 音频格式转换函数（适配FunASR标准格式） ======================
def convert_audio_to_standard(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg音频转换失败：{result.stderr}")
        if not os.path.exists(output_path):
            raise Exception("音频转换失败，输出文件不存在")
        return output_path
    except Exception as e:
        raise Exception(f"音频转换失败：{str(e)}")


# ====================== 视频提取音频函数 ======================
def extract_audio_from_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise Exception(f"视频文件不存在：{video_path}")

    audio_name = f"ext_{uuid.uuid4().hex[:8]}.wav"
    audio_path = str(AUDIO_FOLDER / audio_name)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg执行失败：{result.stderr}")
        if not os.path.exists(audio_path):
            raise Exception("音频提取失败，输出文件不存在")
        return audio_path
    except Exception as e:
        raise Exception(f"提取音频失败：{str(e)}")


def _ffmpeg_exe() -> str:
    bundled = Path(__file__).resolve().parent / ".runtime" / "Scripts" / "ffmpeg.exe"
    return str(bundled) if bundled.exists() else "ffmpeg"


def enhance_morning_speech_audio(audio_path: str) -> str:
    enhanced_name = f"morning_speech_{uuid.uuid4().hex[:8]}.wav"
    enhanced_path = str(AUDIO_FOLDER / enhanced_name)
    filters = (
        "highpass=f=120,"
        "lowpass=f=3800,"
        "afftdn=nf=-25,"
        "dynaudnorm=f=150:g=15:p=0.95,"
        "volume=2.2,"
        "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-45dB:"
        "stop_periods=-1:stop_duration=0.35:stop_threshold=-45dB"
    )
    command = [
        _ffmpeg_exe(),
        "-i", audio_path,
        "-af", filters,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        enhanced_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0 or not os.path.exists(enhanced_path):
        raise Exception(f"晨读人声增强失败：{result.stderr}")
    return enhanced_path


def _has_effective_transcript(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "失败" in text or "error" in lowered or "exception" in lowered:
        return False
    return len(_normalize_reading_text(text)) >= 4


morning_funasr_model = None


def transcribe_morning_reading_with_funasr(audio_path: str) -> str:
    global morning_funasr_model
    from funasr import AutoModel

    model_root = Path(__file__).resolve().parent / "models" / "models" / "iic"
    asr_model = model_root / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model = model_root / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model = model_root / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

    if morning_funasr_model is None:
        morning_funasr_model = AutoModel(
            model=str(asr_model) if asr_model.exists() else "paraformer-zh",
            vad_model=str(vad_model) if vad_model.exists() else "fsmn-vad",
            punc_model=str(punc_model) if punc_model.exists() else "ct-punc-c",
            disable_update=True,
        )
    result = morning_funasr_model.generate(input=audio_path, batch_size_s=60, disable_pbar=True)
    if not result:
        return ""
    text = "".join(str(item.get("text", "")) for item in result if isinstance(item, dict))
    return _clean_ocr_text(text)


def transcribe_morning_reading_with_whisper(audio_path: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8", local_files_only=True)
    segments, _ = model.transcribe(
        audio_path,
        language="zh",
        vad_filter=True,
        beam_size=5,
        initial_prompt="这是一段学生晨读课文、朗读原文、普通话中文朗读，请准确转写学生实际读出的中文内容。",
    )
    return _clean_ocr_text("".join(seg.text.strip() for seg in segments).strip())


async def transcribe_morning_reading_audio(audio_paths) -> tuple[str, str]:
    fallback_text = ""
    notes = []
    for index, audio_path in enumerate(audio_paths):
        if not audio_path or not os.path.exists(audio_path):
            continue
        text = await safe_call(audio_to_text, audio_path)
        if text and not fallback_text:
            fallback_text = text
        if _has_effective_transcript(text):
            note = "已优先识别人声增强后的晨读音频。" if index == 0 else "增强音频未识别稳定，已回退识别原始视频音频。"
            return text, note
        if text:
            notes.append(f"通用转写第 {index + 1} 路返回：{text[:80]}")

    for index, audio_path in enumerate(audio_paths):
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            text = transcribe_morning_reading_with_funasr(audio_path)
            if text and not fallback_text:
                fallback_text = text
            if _has_effective_transcript(text):
                note = "通用转写不稳定，已启用晨读专用中文 ASR 兜底识别。"
                return text, note
            if text:
                notes.append(f"晨读 ASR 第 {index + 1} 路返回：{text[:80]}")
        except Exception as e:
            notes.append(f"晨读 ASR 第 {index + 1} 路失败：{str(e)[:120]}")

    for index, audio_path in enumerate(audio_paths):
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            text = transcribe_morning_reading_with_whisper(audio_path)
            if text and not fallback_text:
                fallback_text = text
            if _has_effective_transcript(text):
                note = "通用转写不稳定，已启用晨读专用 Whisper 中文兜底识别。"
                return text, note
            if text:
                notes.append(f"Whisper 第 {index + 1} 路返回：{text[:80]}")
        except Exception as e:
            notes.append(f"Whisper 第 {index + 1} 路失败：{str(e)[:120]}")

    detail = "\n".join(notes[-3:])
    if detail:
        detail = "\n识别尝试记录：" + detail
    return fallback_text, "未能稳定识别到清晰朗读内容，请靠近麦克风、降低周围人声后重新录制。" + detail


def _make_unrecognized_transcript_note(transcript: str, note: str) -> str:
    if _has_effective_transcript(transcript):
        return transcript
    return "未识别到朗读内容"


def _get_dashscope_api_key() -> str:
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key and not env_key.startswith("sk-ws-"):
        return env_key
    try:
        from config import settings as app_settings
        configured_key = getattr(app_settings, "DASHSCOPE_API_KEY", "")
    except Exception:
        configured_key = ""
    if configured_key and not configured_key.startswith("sk-ws-"):
        return configured_key
    return "sk-1cf6af7eb2ba48288687d78e12969c0b"


def _extract_multimodal_text(resp) -> str:
    try:
        content = resp.output.choices[0].message.content
    except Exception:
        content = getattr(getattr(resp, "output", None), "text", "")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()


def _resize_pil_for_ocr(img, max_side: int = 2200, min_side: int = 1200):
    from PIL import Image

    max_current = max(img.size)
    if max_current <= 0:
        return img

    scale = 1.0
    if max_current > max_side:
        scale = max_side / max_current
    elif max_current < min_side:
        scale = min_side / max_current

    if abs(scale - 1.0) < 0.01:
        return img

    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.LANCZOS)


def prepare_ocr_image(image_path: str) -> str:
    try:
        from PIL import Image, ImageOps
    except Exception as e:
        raise Exception(f"当前环境无法处理图片：{str(e)}")

    src = Path(image_path)
    prepared_path = str(UPLOAD_FOLDER / f"ocr_ready_{uuid.uuid4().hex[:8]}.jpg")
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = _resize_pil_for_ocr(img)
            img.save(prepared_path, "JPEG", quality=92, optimize=True)
    except Exception as e:
        raise Exception(f"图片无法打开或格式异常，请重新截图/拍照后上传：{str(e)}")
    return prepared_path


def _read_cv_image(image_path: str):
    import cv2
    import numpy as np

    data = np.fromfile(str(Path(image_path)), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _save_cv_ocr_variant(image, suffix: str) -> str:
    import cv2

    out_path = UPLOAD_FOLDER / f"ocr_{suffix}_{uuid.uuid4().hex[:8]}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise Exception("增强图片保存失败")
    encoded.tofile(str(out_path))
    return str(out_path)


def _resize_cv_for_ocr(image, max_side: int = 2200, min_side: int = 1200):
    import cv2

    h, w = image.shape[:2]
    max_current = max(h, w)
    if max_current <= 0:
        return image

    scale = 1.0
    if max_current > max_side:
        scale = max_side / max_current
    elif max_current < min_side:
        scale = min_side / max_current

    if abs(scale - 1.0) < 0.01:
        return image

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def _order_points(pts):
    import numpy as np

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    import cv2
    import numpy as np

    rect = _order_points(pts.astype("float32"))
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _detect_document_region(image):
    import cv2

    h, w = image.shape[:2]
    image_area = h * w
    if image_area <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours_info = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > image_area * 0.22:
            warped = _four_point_transform(image, approx.reshape(4, 2))
            if warped is not None and warped.size > 0:
                return warped
    return image


def _deskew_cv_image(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 80:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.4 or abs(angle) > 12:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _color_contrast_cv(image):
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)


def _gray_sharp_cv(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    return cv2.addWeighted(enhanced, 1.7, blur, -0.7, 0)


def _binary_cv(image):
    import cv2

    gray = _gray_sharp_cv(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    block_size = 35
    if min(gray.shape[:2]) < 900:
        block_size = 25
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        11,
    )


def prepare_ocr_image_variants(image_path: str) -> list:
    variants = [{"path": prepare_ocr_image(image_path), "label": "原图清晰化"}]

    try:
        base = _read_cv_image(variants[0]["path"])
        if base is None:
            return variants

        document = _detect_document_region(base)
        document = _resize_cv_for_ocr(document)
        document = _deskew_cv_image(document)
        variants.append({"path": _save_cv_ocr_variant(document, "deskew"), "label": "自动裁边拉正"})

        variants.append({
            "path": _save_cv_ocr_variant(_color_contrast_cv(document), "contrast"),
            "label": "色彩对比增强",
        })
        variants.append({
            "path": _save_cv_ocr_variant(_gray_sharp_cv(document), "gray"),
            "label": "灰度锐化增强",
        })
        variants.append({
            "path": _save_cv_ocr_variant(_binary_cv(document), "binary"),
            "label": "黑白高对比增强",
        })
    except Exception:
        return variants

    return variants


def _clean_ocr_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", text)
    text = re.sub(r"([，。！？；：])\s+", r"\1", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def recognize_image_with_windows_ocr(image_path: str) -> str:
    if os.name != "nt":
        raise Exception("当前系统不支持 Windows 本地 OCR")

    ps_body = r'''
param([string]$ImagePath)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($AsyncOperation, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $task = $asTask.Invoke($null, @($AsyncOperation))
    $task.Wait()
    $task.Result
}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
try {
    $lang = [Windows.Globalization.Language]::new('zh-Hans')
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
} catch {
    $engine = $null
}
if ($null -eq $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
if ($null -eq $engine) {
    throw '当前 Windows 未安装可用 OCR 语言包'
}
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$result.Text
'''
    quoted_path = str(Path(image_path).resolve()).replace("'", "''")
    command = f"& {{{ps_body}}} '{quoted_path}'"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=45,
    )
    if result.returncode != 0:
        raise Exception((result.stderr or result.stdout or "Windows 本地 OCR 调用失败").strip())

    text = _clean_ocr_text(result.stdout)
    if not text:
        raise Exception("Windows 本地 OCR 未识别到文字")
    return text


def _score_ocr_candidate(text: str) -> int:
    cleaned = _clean_ocr_text(text)
    compact = re.sub(r"\s+", "", cleaned)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", cleaned)
    content_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned)
    punctuation = re.findall(r"[，。！？；：,.!?;:]", cleaned)
    suspicious_chars = re.findall(r"[□�#@$^*_+=<>\\|~`]", cleaned)
    unusual_chars = re.findall(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？；：,.!?;:、（）()《》“”\"'：\-—/·%+]", cleaned)
    lines = [line for line in cleaned.splitlines() if line.strip()]

    content_count = len(content_chars)
    chinese_count = len(chinese_chars)
    chinese_ratio = chinese_count / max(content_count, 1)

    score = 0
    score += min(content_count, 220) * 2
    score += min(chinese_count, 220) * 2
    score += int(chinese_ratio * 80)
    score += min(len(lines), 8) * 10
    score += min(len(punctuation), 24) * 3

    if content_count < 6:
        score -= 120
    if chinese_ratio < 0.35:
        score -= 80
    if len(lines) <= 1 and content_count >= 120:
        score -= 12
    if re.search(r"(.)\1{6,}", compact):
        score -= 40

    score -= len(suspicious_chars) * 24
    score -= len(unusual_chars) * 9
    return score


def recognize_image_with_windows_ocr_best(ocr_images) -> dict:
    candidates = []
    errors = []

    for item in ocr_images or []:
        if isinstance(item, dict):
            image_path = item.get("path")
            label = item.get("label") or "增强图片"
        else:
            image_path = str(item)
            label = "增强图片"

        if not image_path or not os.path.exists(image_path):
            continue

        try:
            text = recognize_image_with_windows_ocr(image_path)
            candidates.append({
                "text": text,
                "score": _score_ocr_candidate(text),
                "variant_label": label,
            })
        except Exception as e:
            errors.append(f"{label}: {str(e)}")

    if not candidates:
        detail = "；".join(errors[-3:])
        raise Exception(f"Windows 本地 OCR 多版本识别均未识别到文字{f'：{detail}' if detail else ''}")

    best = max(candidates, key=lambda item: item["score"])
    return {
        "text": best["text"],
        "variant_label": best["variant_label"],
        "candidate_score": best["score"],
        "candidate_count": len(candidates),
    }


def _is_dashscope_account_error(text: str) -> bool:
    text = text or ""
    lowered = text.lower()
    return (
        "overdue-payment" in lowered
        or "access denied" in lowered
        or "account is in good standing" in lowered
    )


def recognize_morning_reference_image(image_path: str, fallback_images=None) -> dict:
    fallback_images = fallback_images or [{"path": image_path, "label": "原图清晰化"}]

    def local_ocr_result(source_note: str) -> dict:
        local_result = recognize_image_with_windows_ocr_best(fallback_images)
        return {
            "text": local_result["text"],
            "source": "windows_ocr_enhanced",
            "source_label": "Windows 本地 OCR（多版本增强）",
            "source_note": source_note,
            "variant_label": local_result.get("variant_label", ""),
            "candidate_count": local_result.get("candidate_count", 0),
        }

    try:
        import dashscope
        from dashscope import MultiModalConversation
    except Exception as e:
        return local_ocr_result(
            f"当前环境缺少 DashScope 图片识别组件，已改用本地 OCR 多版本增强：{str(e)}。"
            "系统会自动裁边、拉正、增强对比度并选择最佳识别结果；手写中文仍建议人工校对。"
        )

    api_key = _get_dashscope_api_key()
    dashscope.api_key = api_key
    image_url = Path(image_path).resolve().as_uri()
    prompt_text = (
        "请识别图片中的晨读原文。只输出图片里可见的正文文字，"
        "尽量保留自然段和标点，不要解释，不要添加图片中不存在的内容。"
    )

    def build_messages(image_value: str):
        return [
            {
                "role": "user",
                "content": [
                    {"image": image_value},
                    {"text": prompt_text},
                ],
            }
        ]

    def call_vision(image_value: str):
        return MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=build_messages(image_value),
            api_key=api_key,
            result_format="message",
            temperature=0.0,
            request_timeout=120,
        )

    try:
        resp = call_vision(image_url)
    except Exception as e:
        err_text = str(e)
        if _is_dashscope_account_error(err_text):
            return local_ocr_result(
                "百炼/DashScope 账号欠费或状态异常，已改用 Windows 本地 OCR 多版本增强。"
                "系统已尝试原图、裁边拉正、灰度锐化和黑白高对比等版本；手写中文仍建议人工校对。"
            )
        if "10054" in err_text or "Connection aborted" in err_text or "ConnectionResetError" in err_text:
            try:
                with open(image_path, "rb") as img_f:
                    image_b64 = base64.b64encode(img_f.read()).decode("ascii")
                resp = call_vision(f"data:image/jpeg;base64,{image_b64}")
            except Exception as fallback_e:
                fallback_text = str(fallback_e)
                if _is_dashscope_account_error(fallback_text):
                    return local_ocr_result(
                        "百炼/DashScope 账号欠费或状态异常，已改用 Windows 本地 OCR 多版本增强。"
                        "系统已尝试原图、裁边拉正、灰度锐化和黑白高对比等版本；手写中文仍建议人工校对。"
                    )
                raise Exception("远程图片识别连接被中断。请检查网络/代理是否允许访问 DashScope，或稍后重试；系统已自动压缩图片并尝试备用上传。")
        else:
            raise
    if getattr(resp, "status_code", 200) != 200:
        message = getattr(resp, "message", "图片文字识别服务调用失败")
        if _is_dashscope_account_error(message):
            return local_ocr_result(
                "百炼/DashScope 账号欠费或状态异常，已改用 Windows 本地 OCR 多版本增强。"
                "系统已尝试原图、裁边拉正、灰度锐化和黑白高对比等版本；手写中文仍建议人工校对。"
            )
        raise Exception(message)

    text = _extract_multimodal_text(resp)
    text = re.sub(r"^\s*(识别结果|正文文字|晨读原文)[:：]\s*", "", text).strip()
    text = _clean_ocr_text(text)
    if not text:
        raise Exception("未识别到图片中的晨读原文，请换一张更清晰、文字更完整的照片。")
    return {
        "text": text,
        "source": "dashscope",
        "source_label": "百炼视觉 OCR",
        "source_note": "使用百炼视觉模型识别原文照片。",
    }


def evaluate_morning_reference_text(text: str, ocr_source: str = "", source_note: str = "") -> dict:
    raw_text = text or ""
    visible_text = raw_text.strip()
    compact_text = re.sub(r"\s+", "", visible_text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", visible_text)
    content_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", visible_text)
    punctuation = re.findall(r"[，。！？；：,.!?;:]", visible_text)
    suspicious_chars = re.findall(r"[□�#@$^*_+=<>\\|~`]", visible_text)
    unusual_chars = re.findall(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？；：,.!?;:、（）()《》“”\"'：\-—/·%+]", visible_text)
    paragraph_count = len([p for p in re.split(r"\n+", visible_text) if p.strip()])
    length = len(content_chars)

    score = 48
    if length >= 80:
        score += 18
    elif length >= 40:
        score += 12
    elif length >= 15:
        score += 6
    else:
        score -= 12

    chinese_ratio = len(chinese_chars) / max(length, 1)
    if chinese_ratio >= 0.7:
        score += 12
    elif chinese_ratio >= 0.45:
        score += 6
    else:
        score -= 14

    if punctuation:
        score += 6
    else:
        score -= 6
    if paragraph_count >= 2:
        score += 5
    if suspicious_chars:
        score -= min(30, len(suspicious_chars) * 6)
    if unusual_chars:
        score -= min(20, len(unusual_chars) * 3)
    if re.search(r"(.)\1{8,}", compact_text):
        score -= 10
    if ocr_source == "windows_ocr":
        score -= 28
    elif ocr_source == "windows_ocr_enhanced":
        score -= 16

    score = max(0, min(100, int(round(score))))
    if score >= 85:
        level = "识别质量较好"
    elif score >= 70:
        level = "基本可用"
    elif score >= 55:
        level = "建议校对"
    else:
        level = "建议重拍"

    suggestions = []
    if source_note:
        suggestions.append(source_note)
    if ocr_source == "windows_ocr_enhanced":
        suggestions.append("当前是本地 OCR 多版本增强结果，已比单张原图识别更稳；手写内容仍可能出现错字、漏字，建议简单校对后再用于晨读评分。")
    elif ocr_source == "windows_ocr":
        suggestions.append("当前是本地 OCR 兜底结果，手写内容很容易出现错字、漏字和乱码，建议人工校对或修复百炼账号后重新识别。")
    if length < 40:
        suggestions.append("识别文本偏短，建议确认原文是否拍全。")
    if not punctuation:
        suggestions.append("标点较少，朗读内容匹配可能不够精细，可简单补充标点。")
    if suspicious_chars or unusual_chars:
        suggestions.append("存在疑似乱码或异常符号，建议人工检查后再评分。")
    if chinese_ratio < 0.45:
        suggestions.append("中文正文占比较低，请确认照片中是否包含清晰的晨读原文。")
    if paragraph_count <= 1 and length >= 120:
        suggestions.append("长文本只有一个段落，可按原文段落适当换行，便于核对。")
    if not suggestions:
        suggestions.append("文本结构较完整，可直接用于晨读内容匹配评分。")

    return {
        "score": score,
        "level": level,
        "char_count": length,
        "paragraph_count": paragraph_count,
        "punctuation_count": len(punctuation),
        "suspicious_count": len(suspicious_chars),
        "unusual_count": len(unusual_chars),
        "report": (
            f"{level}，共识别约 {length} 个有效字符，"
            f"{paragraph_count} 个段落，{len(punctuation)} 个标点，"
            f"{len(suspicious_chars) + len(unusual_chars)} 个疑似异常字符。"
        ),
        "criteria": (
            "评分标准：这是 OCR 文本可用性评分，不是原文正确率。"
            "主要看有效字数、中文正文占比、标点和段落完整度、异常符号/乱码数量、识别来源可靠性；"
            "如果使用 Windows 本地 OCR，会因手写识别准确率较低而降权；多版本增强结果的降权会比单张原图更小。"
        ),
        "suggestions": "\n".join(f"{i + 1}. {item}" for i, item in enumerate(suggestions)),
    }


# ====================== 录用结果生成函数 ======================
def generate_hire_result(score_result: dict, interview_text: str, ai_report: str) -> dict:
    import dashscope
    from dashscope import Generation
    dashscope.api_key = "sk-1cf6af7eb2ba48288687d78e12969c0b"

    total_score = score_result["总分"]

    if total_score >= 90:
        hire_level = "✅ 优先录用"
    elif total_score >= 80:
        hire_level = "📌 拟录用"
    elif total_score >= 70:
        hire_level = "📋 储备录用"
    else:
        hire_level = "❌ 暂不录用"

    prompt = f"""
你是专业的互联网公司招聘面试官，针对CV算法工程师岗位，根据面试者的面试表现，生成专业、真实、和分数完全匹配的录用判定报告。
严格遵守以下规则：
1.  必须严格根据面试总分、面试回答原文、AI点评生成，绝对不能出现和分数矛盾的内容
2.  如果总分低于70分（暂不录用），必须重点说明暂不录用的核心理由，指出面试回答的核心问题，不能写正面的录用理由
3.  分4个固定模块输出：录用评级、面试总分、录用核心理由、核心优势（仅高分有）、待改进建议
4.  语言专业、简洁，符合企业招聘面试报告规范，只输出报告内容，不要多余解释

面试者信息：
- 面试总分：{total_score}分
- 录用评级：{hire_level}
- 面试回答原文：{interview_text}
- AI点评报告：{ai_report}
    """

    try:
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        hire_report = resp.output.text.strip()
        return {
            "hire_level": hire_level,
            "hire_report": hire_report,
            "total_score": total_score
        }
    except Exception as e:
        if total_score >= 70:
            fallback_report = f"""
【录用评级】{hire_level}
【面试总分】{total_score}分

【录用核心理由】
面试表现符合CV算法工程师岗位基础要求，具备对应场景的项目落地经验，对算法选型、工程化部署有基础理解，匹配岗位招聘需求。

【核心优势】
1.  有对应项目的落地经验，对CV算法全流程有基础认知
2.  专业术语使用准确，技术方向和岗位匹配
3.  对三性六讲的表达框架有基本掌握

【待改进建议】
1.  优化回答的结构化表达，补充项目的核心细节和数据成果
2.  强化算法选型逻辑、工程化落地难点的细节描述
3.  提升回答的完整性，避免内容过于简略
            """.strip()
        else:
            fallback_report = f"""
【录用评级】{hire_level}
【面试总分】{total_score}分

【暂不录用核心理由】
本次面试表现未达到CV算法工程师岗位的基础要求，面试回答内容过于简略，未体现出对应岗位所需的项目经验、技术能力和专业认知，无法证明具备岗位要求的算法落地能力和工程化经验。

【核心问题】
1.  回答内容严重不足，未对面试问题做出有效回应，无法体现专业能力
2.  未补充任何项目细节、技术细节，无法证明相关经验
3.  回答不符合面试表达的基本要求，内容完整性严重不足

【待改进建议】
1.  补充完整的项目经历描述，覆盖需求分析、算法选型、工程化落地、成果数据全流程
2.  强化CV算法相关的专业知识储备，提升回答的专业度
3.  优化面试表达的结构化，针对问题做出完整、有细节的回应
            """.strip()

        return {
            "hire_level": hire_level,
            "hire_report": fallback_report,
            "total_score": total_score
        }


# ====================== 核心导入 ======================
from core.asr_service import audio_to_text
from core.llm_service import generate_ai_review, generate_multimodal_ai_review, score_three_six_dimensions, \
    generate_three_practice_questions, evaluate_practice_answer
from core.video_visual_service import analyze_video_multimodal


async def safe_call(func, *args):
    if inspect.iscoroutinefunction(func):
        return await func(*args)
    return func(*args)


# ====================== 训练会话管理 ======================
train_sessions = {}

# ====================== FastAPI服务 ======================
app = FastAPI(title="面试AI复盘系统")
metrics.init_monitor()

from api.live_interview_router import router as live_interview_router
app.include_router(live_interview_router)

_LIVE_HTML = Path(__file__).resolve().parent / "templates" / "live_interview.html"


@app.get("/live", response_class=HTMLResponse, tags=["live-interview"])
async def live_interview_page_main():
    return HTMLResponse(content=_LIVE_HTML.read_text(encoding="utf-8"))
# ====================== TTS 语音朗读接口（只给练习题用） ======================
@app.get("/api/tts")
async def tts_api(text: str):
    try:
        import edge_tts
        voice = "zh-CN-XiaoxiaoNeural"
        communicate = edge_tts.Communicate(text, voice, rate="+0%")

        async def gen():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(gen(), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": str(e)})

# ====================== 录音音频转文字接口 ======================
@app.post("/api/record-audio-to-text")
async def record_audio_to_text(file: UploadFile = File(...)):
    temp_files = []
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        temp_raw_audio = str(UPLOAD_FOLDER / f"record_raw_{uuid.uuid4().hex[:8]}{file_ext}")
        temp_standard_audio = str(AUDIO_FOLDER / f"record_std_{uuid.uuid4().hex[:8]}.wav")
        temp_files.append(temp_raw_audio)
        temp_files.append(temp_standard_audio)

        with open(temp_raw_audio, "wb") as f:
            f.write(await file.read())

        convert_audio_to_standard(temp_raw_audio, temp_standard_audio)
        text = await safe_call(audio_to_text, temp_standard_audio)

        return {
            "code": 200,
            "text": text,
            "msg": "转写完成"
        }

    except Exception as e:
        return {"code": 500, "msg": f"录音转文字失败：{str(e)}"}
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        gc.collect()


@app.post("/api/morning-reading-ocr")
async def morning_reading_ocr(file: UploadFile = File(...)):
    temp_image = None
    prepared_images = []
    try:
        file_ext = os.path.splitext(file.filename or "")[-1].lower() or ".jpg"
        if file_ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            return {"code": 400, "msg": "请上传 JPG、PNG、WEBP 或 BMP 格式的原文照片"}

        temp_image = str(UPLOAD_FOLDER / f"morning_ref_{uuid.uuid4().hex[:8]}{file_ext}")
        with open(temp_image, "wb") as f:
            f.write(await file.read())

        prepared_images = prepare_ocr_image_variants(temp_image)
        ocr_result = recognize_morning_reference_image(prepared_images[0]["path"], prepared_images)
        text = ocr_result["text"]
        evaluation = evaluate_morning_reference_text(
            text,
            ocr_result.get("source", ""),
            ocr_result.get("source_note", ""),
        )
        evaluation["source"] = ocr_result.get("source", "")
        evaluation["source_label"] = ocr_result.get("source_label", "")
        evaluation["variant_label"] = ocr_result.get("variant_label", "")
        evaluation["candidate_count"] = ocr_result.get("candidate_count", 0)
        return {"code": 200, "text": text, "evaluation": evaluation, "msg": "晨读原文识别完成"}
    except Exception as e:
        return {"code": 500, "msg": f"晨读原文图片识别失败：{str(e)}"}
    finally:
        cleanup_files = [temp_image] + [item.get("path") for item in prepared_images if isinstance(item, dict)]
        for f in dict.fromkeys(cleanup_files):
            if f and os.path.exists(f):
                os.remove(f)
        gc.collect()


# ====================== 录音文本直接分析接口 ======================
@app.post("/api/text_train_score")
async def text_train_score(interview_text: str = Form(...)):
    try:
        if not interview_text or len(interview_text.strip()) < 10:
            return {"code": 400, "msg": "面试内容太短，请重新录制"}

        ai_report = await safe_call(generate_ai_review, interview_text)
        score_result = score_three_six_dimensions(interview_text)
        total_score = score_result["总分"]
        hire_result = generate_hire_result(score_result, interview_text, ai_report)

        need_train = False
        session_id = None
        current_question = None
        if total_score <= 85:
            need_train = True
            questions = generate_three_practice_questions(interview_text)
            session_id = str(uuid.uuid4())
            train_sessions[session_id] = {
                "questions": questions["questions"],
                "done": 0,
                "history": []
            }
            current_question = questions["questions"][0]

        return {
            "code": 200,
            "transcript": interview_text,
            "ai_report": ai_report,
            "three_six_score": score_result,
            "hire_result": hire_result,
            "need_train": need_train,
            "session_id": session_id,
            "current_question": current_question,
            "msg": f"分析完成，得分{total_score}分"
        }

    except Exception as e:
        return {"code": 500, "msg": f"分析失败：{str(e)}"}
    finally:
        gc.collect()


# ---------------------- 音频分析接口 ----------------------
@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        temp_path = str(UPLOAD_FOLDER / f"{uuid.uuid4()}{file_ext}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        text = await safe_call(audio_to_text, temp_path)
        ai_report = await safe_call(generate_ai_review, text)

        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

        return {
            "code": 200,
            "data": {
                "score": "详见AI报告",
                "transcript": text,
                "analysis": ai_report,
                "suggestions": ai_report,
                "high_score_answer": ai_report
            }
        }

    except Exception as e:
        return {"code": 500, "msg": f"音频分析失败：{str(e)}"}


# ---------------------- 一键提取音频并AI点评 ----------------------
@app.post("/api/extract-and-analyze")
async def extract_and_analyze(file: UploadFile = File(...)):
    try:
        temp_video = str(UPLOAD_FOLDER / f"{uuid.uuid4()}.mp4")
        with open(temp_video, "wb") as f:
            f.write(await file.read())

        audio_path = extract_audio_from_video(temp_video)
        text = await safe_call(audio_to_text, audio_path)
        ai_report = await safe_call(generate_ai_review, text)

        os.remove(temp_video)
        gc.collect()

        return {
            "code": 200,
            "data": {
                "score": "详见AI报告",
                "transcript": text,
                "analysis": ai_report,
                "suggestions": ai_report,
                "high_score_answer": ai_report
            },
            "audio_saved_path": f"✅ 音频已保存到：{audio_path}"
        }
    except Exception as e:
        return {"code": 500, "msg": f"失败：{str(e)}"}


# ---------------------- 多模态分析接口 ----------------------
@app.post("/api/multimodal-analysis")
async def multimodal_analysis(file: UploadFile = File(...)):
    try:
        temp_video = str(UPLOAD_FOLDER / f"{uuid.uuid4()}.mp4")
        with open(temp_video, "wb") as f:
            f.write(await file.read())

        audio_path = extract_audio_from_video(temp_video)
        text = await safe_call(audio_to_text, audio_path)
        face_pose_result = await safe_call(analyze_video_multimodal, temp_video)
        ai_report = await safe_call(generate_multimodal_ai_review, text, face_pose_result)

        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        gc.collect()

        return {
            "code": 200,
            "result": face_pose_result,
            "ai_report": ai_report,
            "transcript": text
        }
    except Exception as e:
        return {"code": 500, "msg": f"多模态分析失败：{str(e)}"}


# ====================== 视频分析+智能训练接口 ======================
def _clamp_score(value, low=0, high=100):
    return max(low, min(high, int(round(value))))


def _normalize_reading_text(text: str) -> str:
    import re
    return re.sub(r"[\W_]+", "", text or "", flags=re.UNICODE).lower()


def _reading_similarity(source_text: str, transcript: str) -> float:
    import difflib
    source = _normalize_reading_text(source_text)
    spoken = _normalize_reading_text(transcript)
    if not source or not spoken:
        return 0.0
    return difflib.SequenceMatcher(None, source, spoken).ratio()


def _analyze_audio_delivery(audio_path: str) -> dict:
    from pydub import AudioSegment, silence

    audio = AudioSegment.from_file(audio_path)
    duration_sec = max(len(audio) / 1000, 0.1)
    dbfs = audio.dBFS if audio.dBFS != float("-inf") else -60
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=700,
        silence_thresh=max(dbfs - 14, -45),
    )
    silent_ms = sum(end - start for start, end in silent_ranges)
    speech_ratio = max(0.0, min(1.0, 1 - silent_ms / max(len(audio), 1)))

    volume_score = 90 - abs(dbfs + 20) * 1.8
    continuity_score = 55 + speech_ratio * 45
    duration_score = 95 if duration_sec >= 20 else 65 + duration_sec * 1.5

    return {
        "duration_sec": round(duration_sec, 1),
        "dbfs": round(dbfs, 1),
        "speech_ratio": round(speech_ratio, 2),
        "silent_count": len(silent_ranges),
        "volume_score": _clamp_score(volume_score),
        "continuity_score": _clamp_score(continuity_score),
        "duration_score": _clamp_score(duration_score),
    }


def _score_morning_reading_audio(transcript: str, reference_text: str, audio_metrics: dict) -> dict:
    clean_transcript = _normalize_reading_text(transcript)
    text_length = len(clean_transcript)
    similarity = _reading_similarity(reference_text, transcript)

    if reference_text.strip():
        content_score = 55 + similarity * 45
        content_report = (
            f"与晨读原文匹配度约 {round(similarity * 100, 1)}%。"
            if text_length else "未识别到有效朗读内容。"
        )
    else:
        content_score = 55 + min(text_length / 120, 1) * 35
        content_report = "未填写晨读原文，已根据转写完整度、语句长度和表达连贯性进行评分。"

    speed = text_length / max(audio_metrics.get("duration_sec", 0.1), 0.1)
    if speed < 1.2:
        speed_score = 72
        pace_comment = "语速偏慢，可适当提高朗读节奏。"
    elif speed > 5.5:
        speed_score = 76
        pace_comment = "语速偏快，建议放慢并保留停顿。"
    else:
        speed_score = 90
        pace_comment = "语速整体自然，节奏较稳定。"

    voice_score = (
        audio_metrics["volume_score"] * 0.35
        + audio_metrics["continuity_score"] * 0.35
        + speed_score * 0.2
        + audio_metrics["duration_score"] * 0.1
    )
    final_score = voice_score * 0.45 + content_score * 0.45 + speed_score * 0.1

    suggestions = [
        pace_comment,
        "朗读时保持音量稳定，句末停顿要清楚。",
        "重点词句可以略微加强语气，避免整段声音过平。",
    ]
    if reference_text.strip() and similarity < 0.75:
        suggestions.insert(0, "朗读内容与原文有明显差异，建议先对照原文练习准确度。")
    if audio_metrics["speech_ratio"] < 0.55:
        suggestions.insert(0, "录音中停顿或静音较多，建议减少长时间空白。")

    return {
        "total_score": _clamp_score(final_score),
        "voice_score": _clamp_score(voice_score),
        "content_score": _clamp_score(content_score),
        "fluency_score": _clamp_score(speed_score),
        "transcript": transcript,
        "voice_report": (
            f"音量均值约 {audio_metrics['dbfs']} dBFS，朗读有效占比约 "
            f"{round(audio_metrics['speech_ratio'] * 100, 1)}%，{pace_comment}"
        ),
        "content_report": content_report,
        "suggestions": "\n".join(f"{i + 1}. {item}" for i, item in enumerate(suggestions)),
        "metrics": audio_metrics,
    }


def _visual_value(face_pose_result: dict, preferred_key: str, fallback_index: int) -> str:
    if preferred_key in face_pose_result:
        return str(face_pose_result.get(preferred_key) or "")
    values = list(face_pose_result.values())
    if len(values) > fallback_index:
        return str(values[fallback_index] or "")
    return ""


def _score_visual_text(text: str, positive_words, negative_words, base=82) -> int:
    score = base
    for word in positive_words:
        if word in text:
            score += 4
    for word in negative_words:
        if word in text:
            score -= 8
    if "未检测" in text:
        score = min(score, 60)
    return _clamp_score(score)


def _dist2d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def analyze_morning_live_frame(image_bytes: bytes) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"face_detected": False, "eye_closed": False, "status": "未读取到摄像头画面"}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.45,
    ) as face_mesh:
        res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return {"face_detected": False, "eye_closed": False, "status": "未检测到人脸，请面向摄像头"}

    lm = res.multi_face_landmarks[0].landmark
    left_ear = (_dist2d(lm[159], lm[145]) + _dist2d(lm[158], lm[153])) / max(_dist2d(lm[33], lm[133]) * 2, 1e-6)
    right_ear = (_dist2d(lm[386], lm[374]) + _dist2d(lm[385], lm[380])) / max(_dist2d(lm[362], lm[263]) * 2, 1e-6)
    eye_ratio = round((left_ear + right_ear) / 2, 3)
    eye_closed = eye_ratio < 0.18
    status = "检测到闭眼或犯困迹象" if eye_closed else "眼睛状态正常"

    return {
        "face_detected": True,
        "eye_closed": eye_closed,
        "eye_ratio": eye_ratio,
        "status": status,
    }


def _score_morning_reading_video(face_pose_result: dict) -> dict:
    expression = _visual_value(face_pose_result, "表情状态", 0)
    posture = _visual_value(face_pose_result, "肢体动作", 1)

    appearance_score = _score_visual_text(
        expression,
        ["自信", "微笑", "自然", "专注", "充足", "平稳", "面向镜头"],
        ["紧张", "严肃", "游离", "不足", "未检测"],
        84,
    )
    posture_score = _score_visual_text(
        posture,
        ["端正", "稳定", "自然", "专注", "投入", "平稳"],
        ["僵硬", "拘谨", "松散", "不够", "未检测"],
        84,
    )
    total_score = _clamp_score(appearance_score * 0.48 + posture_score * 0.52)

    suggestions = []
    if appearance_score < 80:
        suggestions.append("朗读时眼神尽量面向镜头，表情保持自然放松。")
    else:
        suggestions.append("精神面貌较好，可以继续保持稳定的镜头交流。")
    if posture_score < 80:
        suggestions.append("站姿或坐姿需要更端正，肩颈放松，身体不要频繁晃动。")
    else:
        suggestions.append("姿态整体稳定，适合晨读展示场景。")
    suggestions.append("建议录制时保持上半身完整入镜，光线从正面照射。")

    return {
        "total_score": total_score,
        "appearance_score": appearance_score,
        "posture_score": posture_score,
        "appearance_report": expression,
        "posture_report": posture,
        "suggestions": "\n".join(f"{i + 1}. {item}" for i, item in enumerate(suggestions)),
        "raw_result": face_pose_result,
    }


@app.post("/api/morning-reading-audio")
async def morning_reading_audio(
        file: UploadFile = File(...),
        reference_text: str = Form("")
):
    temp_files = []
    try:
        file_ext = os.path.splitext(file.filename)[-1] or ".mp4"
        temp_video = str(UPLOAD_FOLDER / f"morning_audio_video_{uuid.uuid4().hex[:8]}{file_ext}")
        temp_files.append(temp_video)

        with open(temp_video, "wb") as f:
            f.write(await file.read())

        try:
            audio_path = extract_audio_from_video(temp_video)
        except Exception as e:
            raise Exception(f"请上传带声音的 MP4 视频，当前视频未能提取到朗读音频：{str(e)}")
        temp_files.append(audio_path)

        enhanced_audio_path = None
        enhance_note = ""
        try:
            enhanced_audio_path = enhance_morning_speech_audio(audio_path)
            temp_files.append(enhanced_audio_path)
        except Exception as e:
            enhanced_audio_path = None
            enhance_note = f"人声增强未启用，已直接识别原始音频：{str(e)}"

        transcript, transcript_note = await transcribe_morning_reading_audio(
            [enhanced_audio_path, audio_path]
        )
        if enhance_note:
            transcript_note = enhance_note + "\n" + transcript_note
        scoring_audio_path = enhanced_audio_path if enhanced_audio_path and os.path.exists(enhanced_audio_path) else audio_path
        audio_metrics = _analyze_audio_delivery(scoring_audio_path)
        result = _score_morning_reading_audio(transcript, reference_text, audio_metrics)
        result["transcript_note"] = transcript_note
        if not _has_effective_transcript(transcript):
            result["transcript"] = "未识别到朗读内容"
            result["content_report"] = "音频中检测到声音，但语音模型未能转成稳定文字；请确认朗读声足够清晰、普通话靠近麦克风，并减少背景人声。"
        result["voice_report"] = f"{result['voice_report']}\n{transcript_note}"

        return {"code": 200, "data": result}
    except Exception as e:
        return {"code": 500, "msg": f"晨读视频声音评分失败：{str(e)}"}
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        gc.collect()


@app.post("/api/morning-reading-video")
async def morning_reading_video(file: UploadFile = File(...)):
    temp_video = None
    try:
        file_ext = os.path.splitext(file.filename)[-1] or ".mp4"
        temp_video = str(UPLOAD_FOLDER / f"morning_video_{uuid.uuid4().hex[:8]}{file_ext}")
        with open(temp_video, "wb") as f:
            f.write(await file.read())

        face_pose_result = await safe_call(analyze_video_multimodal, temp_video)
        result = _score_morning_reading_video(face_pose_result)
        return {"code": 200, "data": result}
    except Exception as e:
        return {"code": 500, "msg": f"晨读视频评分失败：{str(e)}"}
    finally:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
        gc.collect()


@app.post("/api/morning-live-frame")
async def morning_live_frame(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return {"code": 400, "msg": "未收到摄像头画面"}
        result = analyze_morning_live_frame(image_bytes)
        return {"code": 200, "data": result}
    except Exception as e:
        return {"code": 500, "msg": f"晨读实时检测失败：{str(e)}"}


@app.post("/api/video_train_score")
async def video_train_score(file: UploadFile = File(...)):
    try:
        temp_video = str(UPLOAD_FOLDER / f"{uuid.uuid4()}.mp4")
        with open(temp_video, "wb") as f:
            f.write(await file.read())

        audio_path = extract_audio_from_video(temp_video)
        text = await safe_call(audio_to_text, audio_path)
        ai_report = await safe_call(generate_ai_review, text)
        os.remove(temp_video)

        score_result = score_three_six_dimensions(text)
        total_score = score_result["总分"]
        hire_result = generate_hire_result(score_result, text, ai_report)

        if total_score > 85:
            return {
                "code": 200,
                "transcript": text,
                "ai_report": ai_report,
                "three_six_score": score_result,
                "hire_result": hire_result,
                "need_train": False,
                "msg": f"优秀！得分{total_score}分，无需练习"
            }

        questions = generate_three_practice_questions(text)
        session_id = str(uuid.uuid4())
        train_sessions[session_id] = {
            "questions": questions["questions"],
            "done": 0,
            "history": []
        }

        return {
            "code": 200,
            "transcript": text,
            "ai_report": ai_report,
            "three_six_score": score_result,
            "hire_result": hire_result,
            "need_train": True,
            "session_id": session_id,
            "current_question": questions["questions"][0],
            "msg": f"得分{total_score}分，开始3道强化练习"
        }

    except Exception as e:
        return {"code": 500, "msg": f"训练分析失败：{str(e)}"}
    finally:
        gc.collect()
        metrics.show_full_report()


# ====================== 练习题提交接口 ======================
@app.post("/api/train_submit")
async def train_submit(
        session_id: str = Form(...),
        user_answer: str = Form(...)
):
    if session_id not in train_sessions:
        return {"code": 400, "msg": "会话已结束或不存在"}

    sess = train_sessions[session_id]
    done = sess["done"]

    if done >= 3:
        del train_sessions[session_id]
        return {"code": 400, "msg": "会话已结束或不存在"}

    question = sess["questions"][done]
    evaluation = evaluate_practice_answer(question, user_answer)

    sess["history"].append({
        "题目": question,
        "你的回答": user_answer,
        "AI评价": evaluation
    })

    sess["done"] += 1
    new_done = sess["done"]

    if new_done >= 3:
        del train_sessions[session_id]
        return {
            "code": 200,
            "status": "finish",
            "history": sess["history"],
            "evaluation": evaluation,
            "msg": "完成全部3道练习题，已退出"
        }

    return {
        "code": 200,
        "status": "continue",
        "current_done": new_done,
        "last_evaluation": evaluation,
        "next_question": sess["questions"][new_done],
        "history": sess["history"]
    }


# ---------------------- 前端界面 ----------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>面试AI复盘系统</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;font-family:Microsoft YaHei}
        body{background:#f7f9fc;padding:40px 20px}
        .container{max-width:1400px;margin:0 auto}
        .title{text-align:center;font-size:32px;color:#2c3e50;margin-bottom:12px}
        .subtitle{text-align:center;color:#7f8c8d;font-size:16px;margin-bottom:40px}
        .card{background:white;border-radius:16px;padding:40px;box-shadow:0 4px 20px rgba(0,0,0,0.08);margin-bottom:30px}
        .card h2{font-size:22px;color:#2c3e50;margin-bottom:20px}
        .double-box{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
        .upload-box{border:2px dashed #bdc3c7;border-radius:12px;padding:40px 20px;text-align:center;cursor:pointer;transition:0.3s}
        .upload-box:hover{border-color:#27ae60;background:#fafbfc}
        .upload-box input{display:none}
        .upload-box label{cursor:pointer;color:#27ae60;font-weight:bold;font-size:16px}
        .file-name{margin-top:10px;color:#2c3e50;font-size:14px}
        .btn-submit{width:100%;padding:16px;background:#27ae60;color:white;border:none;border-radius:12px;font-size:18px;font-weight:bold;cursor:pointer;margin-top:10px}
        .btn-submit:hover{background:#219653}
        .btn-submit:disabled{background:#95a5a6;cursor:not-allowed}
        .result-box{margin-top:20px;padding:24px;background:#f1f9f7;border-radius:12px;display:none}
        .result-box h3{color:#27ae60;margin-bottom:16px;font-size:20px}
        .result-section{margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #d1e7dd}
        .result-section h4{color:#2c3e50;margin-bottom:8px;font-size:16px}
        .result-section p{color:#34495e;line-height:1.6;font-size:14px;white-space:pre-wrap}
        .score{font-size:24px;font-weight:bold;color:#27ae60}
        .loading{display:inline-block;width:20px;height:20px;border:3px solid #f3f3f3;border-top:3px solid #fff;border-radius:50%;animation:spin 1s linear infinite}
        @keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}

        .record-control-box{margin-top:20px;padding:20px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc}
        .record-title{font-size:16px;color:#2c3e50;margin-bottom:12px;font-weight:bold}
        .record-btns-row{display:flex;gap:12px;align-items:center;margin-bottom:15px}
        .btn-record{padding:10px 20px;border:none;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer}
        .btn-start{background:#7c3aed;color:white}
        .btn-start:hover{background:#6d28d9}
        .btn-stop{background:#ef4444;color:white}
        .btn-stop:hover{background:#dc2626}
        .btn-cancel{background:#9ca3af;color:white}
        .btn-cancel:hover{background:#6b7280}
        .record-time{font-size:20px;font-weight:bold;color:#2c3e50}
        .record-time.recording{color:#ef4444}
        .record-text-box{margin-top:15px;display:block}
        .record-text-box textarea{width:100%;height:120px;padding:12px;border-radius:8px;border:1px solid #cbd5e1;font-size:14px;resize:vertical}
        .hire-box{background:#fffbeb;border:1px solid #fcd34d;border-radius:12px;padding:20px;margin-top:15px}
        .hire-level{font-size:22px;font-weight:bold;margin-bottom:10px}
        .answer-row {
            display: flex;
            gap: 8px;
            margin-top: 10px;
            align-items: flex-start;
        }
        #extractAnswerInput {
            flex: 1;
            height: 80px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .mic-btn {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: #059669;
            color: #fff;
            font-size: 18px;
            border: none;
            cursor: pointer;
        }
        .mic-btn.rec {
            background: #e53e3e;
            animation: pulse 1s infinite;
        }
        .cancel-mic-btn {
            width: 44px;
            height: 44px;
            border-radius: 8px;
            background: #9ca3af;
            color: #fff;
            font-size: 14px;
            border: none;
            cursor: pointer;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }
        .play-eval-btn {
            margin-top: 8px;
            padding: 6px 14px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .morning-card {
            margin-top: 30px;
            padding: 24px;
            border-radius: 14px;
            background: #f8fbff;
            border: 1px solid #dbeafe;
        }
        .morning-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 16px;
        }
        .morning-title h3 {
            margin: 0;
            color: #1e3a5f;
        }
        .morning-title span {
            color: #64748b;
            font-size: 13px;
        }
        .morning-ref {
            margin-bottom: 18px;
        }
        .morning-ref-grid {
            display: grid;
            grid-template-columns: minmax(220px, 0.75fr) minmax(0, 1.25fr);
            gap: 14px;
            align-items: stretch;
        }
        .morning-ref label {
            display: block;
            color: #334155;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .morning-ref textarea {
            width: 100%;
            min-height: 86px;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            resize: vertical;
            font-size: 14px;
        }
        .morning-ref-note {
            color: #64748b;
            font-size: 12px;
            margin: 8px 0 0;
            line-height: 1.5;
        }
        .morning-ocr-btn {
            margin-top: 10px;
            padding: 12px 18px;
            font-size: 15px;
        }
        .morning-image-preview {
            margin-top: 14px;
        }
        .morning-image-preview-title {
            color: #334155;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .morning-image-preview-frame {
            min-height: 360px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .morning-image-preview-frame img {
            display: none;
            width: 100%;
            height: 100%;
            max-height: 520px;
            object-fit: contain;
            background: #ffffff;
        }
        .morning-image-preview-empty {
            color: #94a3b8;
            font-size: 13px;
            padding: 18px;
            text-align: center;
        }
        .ocr-eval-box {
            display: none;
            margin-top: 10px;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
        }
        .ocr-eval-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 8px;
        }
        .ocr-eval-head strong {
            color: #1e3a5f;
            font-size: 14px;
        }
        .ocr-eval-score {
            color: #2563eb;
            font-weight: bold;
            font-size: 18px;
            white-space: nowrap;
        }
        .ocr-eval-box p {
            color: #334155;
            font-size: 13px;
            line-height: 1.6;
            margin: 0;
            white-space: pre-wrap;
        }
        .live-monitor {
            margin: 18px 0;
            padding: 14px;
            border-radius: 12px;
            border: 1px solid #c7d2fe;
            background: #f8fafc;
            transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        }
        .live-monitor.alerting {
            border-color: #ef4444;
            background: #fff1f2;
            box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.18), 0 14px 36px rgba(153, 27, 27, 0.18);
            animation: liveAlertPulse 0.9s ease-in-out infinite;
        }
        .live-monitor-row {
            display: grid;
            grid-template-columns: 220px minmax(0, 1fr);
            gap: 14px;
            align-items: stretch;
        }
        .live-monitor video {
            width: 100%;
            aspect-ratio: 4 / 3;
            border-radius: 10px;
            background: #0f172a;
            object-fit: cover;
        }
        .live-monitor-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .live-monitor-actions button {
            width: auto;
            margin-top: 0;
            padding: 10px 16px;
            font-size: 14px;
        }
        .monitor-status {
            padding: 10px 12px;
            border-radius: 10px;
            background: #e0f2fe;
            color: #0c4a6e;
            font-weight: bold;
            line-height: 1.5;
        }
        .monitor-status.warn {
            background: #fef3c7;
            color: #92400e;
        }
        .monitor-status.danger {
            background: #fee2e2;
            color: #991b1b;
        }
        .volume-meter {
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: #e2e8f0;
            margin: 10px 0 6px;
        }
        .volume-meter span {
            display: block;
            width: 0%;
            height: 100%;
            background: #22c55e;
            transition: width 0.2s ease;
        }
        .volume-meter span.low {
            background: #ef4444;
        }
        body.morning-live-screen-alert::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 9998;
            pointer-events: none;
            background: rgba(220, 38, 38, 0.18);
            animation: screenAlertFlash 0.9s ease-in-out infinite;
        }
        body.morning-live-screen-alert::after {
            content: "晨读提醒";
            position: fixed;
            top: 18px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            padding: 10px 18px;
            border-radius: 999px;
            background: #dc2626;
            color: #fff;
            font-weight: 800;
            letter-spacing: 0;
            box-shadow: 0 12px 30px rgba(127, 29, 29, 0.28);
            pointer-events: none;
        }
        @keyframes screenAlertFlash {
            0%, 100% { opacity: 0.28; }
            50% { opacity: 0.82; }
        }
        @keyframes liveAlertPulse {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-1px); }
        }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }
        .score-pill {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 14px 12px;
            text-align: center;
        }
        .score-pill strong {
            display: block;
            color: #2563eb;
            font-size: 24px;
            line-height: 1;
        }
        .score-pill span {
            display: block;
            color: #64748b;
            font-size: 12px;
            margin-top: 6px;
        }
        @media (max-width: 900px) {
            .double-box { grid-template-columns: 1fr; }
            .morning-ref-grid { grid-template-columns: 1fr; }
            .live-monitor-row { grid-template-columns: 1fr; }
            .morning-image-preview-frame { min-height: 260px; }
            .score-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎓 面试AI复盘系统</h1>
        <p class="subtitle">AI 自动分析面试表现，生成专业复盘报告，快速提升面试能力</p>

        <div class="card" style="background:linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);color:#e2e8f0;border:2px solid #2563eb;margin-bottom:30px;">
            <h2 style="color:#fff;margin-bottom:12px;">🎥 实时面对面 AI 面试（百炼 WebSocket）</h2>
            <p style="font-size:15px;line-height:1.7;margin-bottom:18px;color:#cbd5e1;">
                摄像头 + 麦克风与 AI 面试官实时对话，结束后可下载 Markdown 面试总结。<br>
                需先在 <code style="background:#334155;padding:2px 6px;border-radius:4px;">config/settings.py</code> 填写百炼 APP ID 与 Workspace ID。
            </p>
            <a href="/live" style="display:inline-block;padding:14px 28px;background:#2563eb;color:#fff;font-size:18px;font-weight:bold;border-radius:12px;text-decoration:none;">进入实时面试 →</a>
        </div>

        <div class="card">
            <h2>📤 开始面试复盘</h2>
            <div class="double-box">
                <div>
                    <h3 style="margin-bottom:10px;">🎙️ 上传音频</h3>
                    <form id="uploadForm">
                        <div class="upload-box" id="audioUploadBox">
                            <input type="file" id="audioFile" accept="audio/*" required>
                            <label for="audioFile">点击或拖拽上传音频</label>
                            <p class="file-name" id="audioFileName">未选择文件</p>
                        </div>
                        <button type="submit" class="btn-submit" id="audioAnalyzeBtn" disabled>🚀 开始AI分析</button>
                    </form>
                </div>

                <div>
                    <h3 style="margin-bottom:10px;">🎥 视频多模态分析</h3>
                    <div class="upload-box" id="videoUploadBox">
                        <input type="file" id="modalVideoFile" accept="video/*">
                        <label for="modalVideoFile">点击或拖拽上传视频</label>
                        <p class="file-name" id="modalVideoFileName">未选择文件</p>
                    </div>
                    <button class="btn-submit" id="modalAnalyzeBtn">🎨 多模态分析</button>
                </div>
            </div>

            <div style="margin-top:30px;">
                <h3 style="margin-bottom:10px;">📽️ 视频提取音频并AI点评（带三性六讲+智能训练+录用判定）</h3>
                <div class="upload-box" id="extractUploadBox">
                    <input type="file" id="extractVideoFile" accept="video/*">
                    <label for="extractVideoFile">点击或拖拽上传视频</label>
                    <p class="file-name" id="extractVideoFileName">未选择文件</p>
                </div>

                <div class="record-control-box">
                    <div class="record-title">🎤 实时录音面试（无需上传视频，直接口述分析）</div>
                    <div class="record-btns-row">
                        <span class="record-time" id="recordTime">00:00</span>
                        <button class="btn-record btn-start" id="startRecordBtn">开始录音</button>
                        <button class="btn-record btn-stop" id="stopRecordBtn" disabled>停止录音</button>
                        <button class="btn-record btn-cancel" id="cancelRecordBtn" disabled>取消录音</button>
                    </div>
                    <div class="record-text-box" id="recordTextBox">
                        <textarea id="recordTextResult" placeholder="录音转写结果将显示在这里，可手动编辑..." readonly></textarea>
                    </div>
                </div>

                <button class="btn-submit" id="extractAnalyzeBtn">✅ 一键提取音频并AI点评</button>
            </div>

            <div class="morning-card">
                <div class="morning-title">
                    <h3>晨读评分</h3>
                    <span>评估朗读声音、朗读内容、精神面貌和站姿</span>
                </div>
                <div class="morning-ref">
                    <div class="morning-ref-grid">
                        <div>
                            <label for="morningReferenceImage">晨读原文照片</label>
                            <div class="upload-box" id="morningReferenceUploadBox">
                                <input type="file" id="morningReferenceImage" accept="image/*,.jpg,.jpeg,.png,.webp,.bmp">
                                <label for="morningReferenceImage">点击或拖拽上传原文照片</label>
                                <p class="file-name" id="morningReferenceImageName">未选择文件</p>
                            </div>
                            <button class="btn-submit morning-ocr-btn" id="morningOcrBtn" disabled>识别晨读原文</button>
                            <p class="morning-ref-note">上传课本、讲义或打印材料照片后自动识别文字，不需要手打。</p>
                            <div class="morning-image-preview" id="morningReferencePreviewBox">
                                <div class="morning-image-preview-title">原图预览</div>
                                <div class="morning-image-preview-frame">
                                    <span class="morning-image-preview-empty" id="morningReferencePreviewEmpty">上传后在这里显示原图</span>
                                    <img id="morningReferencePreview" alt="晨读原文原图预览">
                                </div>
                            </div>
                        </div>
                        <div>
                            <label for="morningReferenceText">识别出的晨读原文（可编辑）</label>
                            <textarea id="morningReferenceText" placeholder="识别后的原文会显示在这里，可简单检查和修改后再进行内容评分..."></textarea>
                            <p class="morning-ref-note">声音与内容评分会用这里的文字和视频转写内容进行匹配。</p>
                            <div class="ocr-eval-box" id="morningOcrEvalBox">
                                <div class="ocr-eval-head">
                                    <strong id="morningOcrEvalLevel">文本识别评价</strong>
                                    <span class="ocr-eval-score" id="morningOcrEvalScore">--</span>
                                </div>
                                <p id="morningOcrEvalReport">上传原文照片并识别后生成评价。</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="live-monitor">
                    <div class="morning-title" style="margin-bottom:10px;">
                        <h3>实时晨读监测</h3>
                        <span>摄像头检测犯困闭眼，麦克风检测声音过小</span>
                    </div>
                    <div class="live-monitor-row">
                        <video id="morningLiveVideo" autoplay muted playsinline></video>
                        <div>
                            <div class="live-monitor-actions">
                                <button class="btn-submit" id="morningLiveStartBtn">开始实时检测</button>
                                <button class="btn-submit" id="morningLiveStopBtn" disabled>停止检测</button>
                            </div>
                            <div class="monitor-status" id="morningLiveStatus">等待开启摄像头和麦克风。</div>
                            <div class="volume-meter"><span id="morningVolumeBar"></span></div>
                            <p class="morning-ref-note" id="morningLiveHint">检测中如果连续闭眼或音量过低，会在这里提醒。</p>
                        </div>
                    </div>
                    <canvas id="morningLiveCanvas" width="320" height="240" style="display:none;"></canvas>
                </div>
                <div class="double-box">
                    <div>
                        <h3 style="margin-bottom:10px;">上传晨读视频（提取声音）</h3>
                        <div class="upload-box" id="morningAudioUploadBox">
                            <input type="file" id="morningAudioFile" accept="video/mp4,.mp4">
                            <label for="morningAudioFile">点击或拖拽上传 MP4 视频</label>
                            <p class="file-name" id="morningAudioFileName">未选择文件</p>
                        </div>
                        <button class="btn-submit" id="morningAudioAnalyzeBtn" disabled>开始声音与内容评分</button>
                    </div>
                    <div>
                        <h3 style="margin-bottom:10px;">上传晨读视频</h3>
                        <div class="upload-box" id="morningVideoUploadBox">
                            <input type="file" id="morningVideoFile" accept="video/mp4,.mp4">
                            <label for="morningVideoFile">点击或拖拽上传 MP4 视频</label>
                            <p class="file-name" id="morningVideoFileName">未选择文件</p>
                        </div>
                        <button class="btn-submit" id="morningVideoAnalyzeBtn" disabled>开始精神面貌与站姿评分</button>
                    </div>
                </div>
            </div>

            <div class="result-box" id="audioResultBox">
                <h3>✅ 音频分析完成</h3>
                <div class="result-section"><h4>🎯 面试综合评分</h4><p class="score" id="audioScore">加载中...</p></div>
                <div class="result-section"><h4>📝 语音转文字结果</h4><p id="audioTranscript">加载中...</p></div>
                <div class="result-section"><h4>🧠 AI问题分析</h4><p id="audioAnalysis">加载中...</p></div>
                <div class="result-section"><h4>💡 优化建议</h4><p id="audioSuggestions">加载中...</p></div>
                <div class="result-section"><h4>🏆 高分参考答案</h4><p id="audioHighScore">加载中...</p></div>
            </div>

            <div class="result-box" id="modalResultBox">
                <h3>🧠 多模态分析完成</h3>
                <div class="result-section"><h4>😊 神态表情</h4><p id="modalFace">加载中...</p></div>
                <div class="result-section"><h4>🤸 肢体动作</h4><p id="modalPose">加载中...</p></div>
                <div class="result-section"><h4>📊 AI综合评价</h4><p id="modalEval" style="white-space:pre-wrap">加载中...</p></div>
            </div>

            <div class="result-box" id="morningResultBox">
                <h3>晨读评分结果</h3>
                <div class="score-grid">
                    <div class="score-pill"><strong id="morningTotalScore">--</strong><span>综合分</span></div>
                    <div class="score-pill"><strong id="morningVoiceScore">--</strong><span>声音表现</span></div>
                    <div class="score-pill"><strong id="morningContentScore">--</strong><span>朗读内容</span></div>
                    <div class="score-pill"><strong id="morningAppearanceScore">--</strong><span>精神面貌</span></div>
                    <div class="score-pill"><strong id="morningPostureScore">--</strong><span>站姿体态</span></div>
                </div>
                <div class="result-section"><h4>朗读转写</h4><p id="morningTranscript">等待视频声音评分...</p></div>
                <div class="result-section"><h4>声音评分</h4><p id="morningVoiceReport">等待视频声音评分...</p></div>
                <div class="result-section"><h4>内容评分</h4><p id="morningContentReport">等待视频声音评分...</p></div>
                <div class="result-section"><h4>精神面貌</h4><p id="morningAppearanceReport">等待视频评分...</p></div>
                <div class="result-section"><h4>站姿体态</h4><p id="morningPostureReport">等待视频评分...</p></div>
                <div class="result-section"><h4>提升建议</h4><p id="morningSuggestions">上传晨读视频后生成建议。</p></div>
            </div>

            <div class="result-box" id="extractResultBox">
                <h3>✅ 分析完成</h3>
                <div class="hire-box" id="hireBox" style="display:none;">
                    <div class="hire-level" id="hireLevel">加载中...</div>
                    <div class="result-section" style="border-bottom:none;padding-bottom:0;">
                        <h4>📋 录用判定报告</h4>
                        <p id="hireReport" style="white-space:pre-wrap">加载中...</p>
                    </div>
                </div>
                <div class="result-section"><h4>🎯 三性六讲综合评分</h4><p class="score" id="extractScore">加载中...</p></div>
                <div class="result-section"><h4>📝 语音转文字结果</h4><p id="extractTranscript">加载中...</p></div>
                <div class="result-section"><h4>🧠 AI基础点评</h4><p id="extractAnalysis">加载中...</p></div>
                <div class="result-section"><h4>💡 优化建议</h4><p id="extractSuggestions">加载中...</p></div>
                <div class="result-section"><h4>🏆 高分参考答案</h4><p id="extractHighScore">加载中...</p></div>

                <div class="result-section" id="trainQuestionSection" style="display:none;">
                    <h4>📌 练习题（第1/3题）</h4>
                    <p id="extractCurrentQuestion">加载中...</p>
                    <div class="answer-row">
                        <textarea id="extractAnswerInput" placeholder="请输入或语音回答"></textarea>
                        <button class="mic-btn" id="micBtn">🎤</button>
                        <button class="cancel-mic-btn" id="cancelMicBtn">取消</button>
                    </div>
                    <button class="btn-submit" id="extractSubmitAnswerBtn" style="margin-top:10px;background:#2980b9;">提交答案并获取AI评价</button>
                </div>

                <div class="result-section" id="trainEvalSection" style="display:none;">
                    <h4>💡 上一题AI评价</h4>
                    <p id="extractLastEvaluation">暂无评价</p>
                    <button class="play-eval-btn" id="playEvalBtn">🔊 播放评价</button>
                </div>

                <div class="result-section" id="trainHistorySection" style="display:none;">
                    <h4>📋 本次训练记录</h4>
                    <p id="extractTrainHistory">暂无记录</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let recordTimer = null;
        let recordSeconds = 0;
        let currentAudio = null;
        let answerRecognition = null;
        let isAnswerMicRecording = false;
        let trainSessionId = null;
        let trainCurrentRound = 0;
        let isRecordingMode = false;

        // 只修复这里：语音识别不自动停 + 取消按钮
        if (window.SpeechRecognition || window.webkitSpeechRecognition) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            answerRecognition = new SpeechRecognition();
            answerRecognition.lang = "zh-CN";
            answerRecognition.interimResults = false;
            // 关键修复：不会说完自动停
            answerRecognition.continuous = true;
            answerRecognition.maxAlternatives = 1;

            answerRecognition.onresult = (e) => {
                const txt = e.results[0][0].transcript;
                document.getElementById("extractAnswerInput").value = txt;
            };
        }

        // 停止语音输入
        function stopAnswerMic() {
            isAnswerMicRecording = false;
            document.getElementById("micBtn").classList.remove("rec");
            try { answerRecognition.stop(); } catch (e) {}
        }

        // 取消：停止+清空
        function cancelAnswerMic() {
            stopAnswerMic();
            document.getElementById("extractAnswerInput").value = "";
        }

        // 切换麦克风
        function toggleAnswerMic() {
            if (!answerRecognition) {
                alert("浏览器不支持语音输入，请使用Chrome/Edge");
                return;
            }
            if (isAnswerMicRecording) {
                stopAnswerMic();
            } else {
                isAnswerMicRecording = true;
                document.getElementById("micBtn").classList.add("rec");
                answerRecognition.start();
            }
        }

        // 播放评价
        async function playTTS(text) {
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
            if (!text || text.trim() === '' || text.trim() === '暂无评价') return;
            try {
                const response = await fetch(`/api/tts?text=${encodeURIComponent(text)}`);
                if (!response.ok) throw new Error('TTS生成失败');
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                currentAudio = new Audio(audioUrl);
                currentAudio.play();
            } catch (error) {
                console.error('TTS播放失败:', error);
                alert('语音播放失败：' + error.message);
            }
        }

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60).toString().padStart(2, '0');
            const s = (seconds % 60).toString().padStart(2, '0');
            return `${m}:${s}`;
        }

        document.addEventListener('DOMContentLoaded', function(){
            const startRecordBtn = document.getElementById('startRecordBtn');
            const stopRecordBtn = document.getElementById('stopRecordBtn');
            const cancelRecordBtn = document.getElementById('cancelRecordBtn');
            const recordTime = document.getElementById('recordTime');
            const recordTextResult = document.getElementById('recordTextResult');
            const extractResultBox = document.getElementById('extractResultBox');
            const audioResultBox = document.getElementById('audioResultBox');
            const modalResultBox = document.getElementById('modalResultBox');
            const morningResultBox = document.getElementById('morningResultBox');
            const hireBox = document.getElementById('hireBox');

            const trainQuestionSection = document.getElementById('trainQuestionSection');
            const trainEvalSection = document.getElementById('trainEvalSection');
            const trainHistorySection = document.getElementById('trainHistorySection');
            const extractCurrentQuestion = document.getElementById('extractCurrentQuestion');
            const extractAnswerInput = document.getElementById('extractAnswerInput');
            const extractSubmitAnswerBtn = document.getElementById('extractSubmitAnswerBtn');
            const extractLastEvaluation = document.getElementById('extractLastEvaluation');
            const extractTrainHistory = document.getElementById('extractTrainHistory');
            const micBtn = document.getElementById('micBtn');
            const cancelMicBtn = document.getElementById('cancelMicBtn');
            const playEvalBtn = document.getElementById('playEvalBtn');
            const morningAudioFile = document.getElementById('morningAudioFile');
            const morningAudioFileName = document.getElementById('morningAudioFileName');
            const morningAudioAnalyzeBtn = document.getElementById('morningAudioAnalyzeBtn');
            const morningVideoFile = document.getElementById('morningVideoFile');
            const morningVideoFileName = document.getElementById('morningVideoFileName');
            const morningVideoAnalyzeBtn = document.getElementById('morningVideoAnalyzeBtn');
            const morningReferenceImage = document.getElementById('morningReferenceImage');
            const morningReferenceImageName = document.getElementById('morningReferenceImageName');
            const morningReferencePreview = document.getElementById('morningReferencePreview');
            const morningReferencePreviewEmpty = document.getElementById('morningReferencePreviewEmpty');
            const morningOcrBtn = document.getElementById('morningOcrBtn');
            const morningReferenceText = document.getElementById('morningReferenceText');
            const morningOcrEvalBox = document.getElementById('morningOcrEvalBox');
            const morningOcrEvalLevel = document.getElementById('morningOcrEvalLevel');
            const morningOcrEvalScore = document.getElementById('morningOcrEvalScore');
            const morningOcrEvalReport = document.getElementById('morningOcrEvalReport');
            const morningLiveVideo = document.getElementById('morningLiveVideo');
            const morningLiveCanvas = document.getElementById('morningLiveCanvas');
            const morningLiveStartBtn = document.getElementById('morningLiveStartBtn');
            const morningLiveStopBtn = document.getElementById('morningLiveStopBtn');
            const morningLiveStatus = document.getElementById('morningLiveStatus');
            const morningVolumeBar = document.getElementById('morningVolumeBar');
            const morningLiveHint = document.getElementById('morningLiveHint');
            const morningLiveMonitor = document.querySelector('.live-monitor');

            let morningLiveStream = null;
            let morningAudioContext = null;
            let morningAnalyser = null;
            let morningVolumeTimer = null;
            let morningFrameTimer = null;
            let lowVolumeCount = 0;
            let closedEyeCount = 0;
            let morningLiveAlertReason = '';
            let morningLiveLastBeepAt = 0;
            let morningReferencePreviewUrl = null;

            // 绑定按钮
            micBtn.onclick = toggleAnswerMic;
            cancelMicBtn.onclick = cancelAnswerMic;
            playEvalBtn.onclick = () => playTTS(extractLastEvaluation.textContent);

            function showMorningResultBox() {
                morningResultBox.style.display = 'block';
                audioResultBox.style.display = 'none';
                modalResultBox.style.display = 'none';
                extractResultBox.style.display = 'none';
                hireBox.style.display = 'none';
            }

            function updateMorningTotal() {
                const ids = [
                    'morningVoiceScore',
                    'morningContentScore',
                    'morningAppearanceScore',
                    'morningPostureScore'
                ];
                const values = ids
                    .map(id => parseInt(document.getElementById(id).textContent, 10))
                    .filter(v => Number.isFinite(v));
                if (values.length > 0) {
                    const total = Math.round(values.reduce((a, b) => a + b, 0) / values.length);
                    document.getElementById('morningTotalScore').textContent = total;
                }
            }

            function appendMorningSuggestion(text) {
                const box = document.getElementById('morningSuggestions');
                if (!text) return;
                const old = box.textContent || '';
                if (!old || old === '上传晨读视频后生成建议。') {
                    box.textContent = text;
                } else {
                    box.textContent = old + '\\n' + text;
                }
            }

            function updateMorningOcrEvaluation(evaluation) {
                if (!evaluation) {
                    morningOcrEvalBox.style.display = 'none';
                    return;
                }
                morningOcrEvalBox.style.display = 'block';
                morningOcrEvalLevel.textContent = evaluation.level || '文本识别评价';
                morningOcrEvalScore.textContent = (evaluation.score ?? '--') + '分';
                const details = [
                    evaluation.source_label ? '识别来源：' + evaluation.source_label : '',
                    evaluation.variant_label ? '最佳增强版本：' + evaluation.variant_label : '',
                    evaluation.candidate_count ? '已比较版本数：' + evaluation.candidate_count : '',
                    evaluation.report || '',
                    evaluation.criteria || '',
                    evaluation.suggestions || ''
                ].filter(Boolean).join('\\n');
                morningOcrEvalReport.textContent = details || '已完成文本识别评价。';
            }

            function clearMorningReferencePreview() {
                if (morningReferencePreviewUrl) {
                    URL.revokeObjectURL(morningReferencePreviewUrl);
                    morningReferencePreviewUrl = null;
                }
                morningReferencePreview.removeAttribute('src');
                morningReferencePreview.style.display = 'none';
                morningReferencePreviewEmpty.style.display = 'block';
            }

            function showMorningReferencePreview(file) {
                if (!file) {
                    clearMorningReferencePreview();
                    return;
                }
                if (morningReferencePreviewUrl) {
                    URL.revokeObjectURL(morningReferencePreviewUrl);
                }
                morningReferencePreviewUrl = URL.createObjectURL(file);
                morningReferencePreview.src = morningReferencePreviewUrl;
                morningReferencePreview.style.display = 'block';
                morningReferencePreviewEmpty.style.display = 'none';
            }

            function setMorningLiveStatus(text, level = '') {
                morningLiveStatus.textContent = text;
                morningLiveStatus.className = 'monitor-status' + (level ? ' ' + level : '');
            }

            function playMorningLiveBeep() {
                const now = Date.now();
                if (now - morningLiveLastBeepAt < 3000) return;
                morningLiveLastBeepAt = now;
                try {
                    const Ctx = window.AudioContext || window.webkitAudioContext;
                    if (!Ctx) return;
                    const ctx = new Ctx();
                    const gain = ctx.createGain();
                    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.16, ctx.currentTime + 0.03);
                    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.38);
                    gain.connect(ctx.destination);

                    [0, 0.16].forEach((offset) => {
                        const osc = ctx.createOscillator();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(880, ctx.currentTime + offset);
                        osc.connect(gain);
                        osc.start(ctx.currentTime + offset);
                        osc.stop(ctx.currentTime + offset + 0.1);
                    });
                    setTimeout(() => ctx.close().catch(() => {}), 650);
                } catch (err) {
                    console.warn('提示音播放失败', err);
                }
            }

            function setMorningLiveAlert(active, reason = '') {
                morningLiveAlertReason = active ? reason : '';
                document.body.classList.toggle('morning-live-screen-alert', active);
                if (morningLiveMonitor) {
                    morningLiveMonitor.classList.toggle('alerting', active);
                }
                if (active) playMorningLiveBeep();
            }

            function stopMorningLiveMonitor() {
                if (morningVolumeTimer) clearInterval(morningVolumeTimer);
                if (morningFrameTimer) clearInterval(morningFrameTimer);
                morningVolumeTimer = null;
                morningFrameTimer = null;
                lowVolumeCount = 0;
                closedEyeCount = 0;
                setMorningLiveAlert(false);
                if (morningAudioContext) {
                    morningAudioContext.close().catch(() => {});
                    morningAudioContext = null;
                }
                if (morningLiveStream) {
                    morningLiveStream.getTracks().forEach(track => track.stop());
                    morningLiveStream = null;
                }
                morningLiveVideo.srcObject = null;
                morningVolumeBar.style.width = '0%';
                morningVolumeBar.classList.remove('low');
                morningLiveStartBtn.disabled = false;
                morningLiveStopBtn.disabled = true;
                morningLiveHint.textContent = '检测已停止。';
                setMorningLiveStatus('等待开启摄像头和麦克风。');
            }

            async function sendMorningFrameForCheck() {
                if (!morningLiveStream || !morningLiveVideo.videoWidth) return;
                const ctx = morningLiveCanvas.getContext('2d');
                ctx.drawImage(morningLiveVideo, 0, 0, morningLiveCanvas.width, morningLiveCanvas.height);
                const blob = await new Promise(resolve => morningLiveCanvas.toBlob(resolve, 'image/jpeg', 0.75));
                if (!blob) return;
                const fd = new FormData();
                fd.append('file', blob, 'frame.jpg');
                try {
                    const res = await fetch('/api/morning-live-frame', { method: 'POST', body: fd });
                    const data = await res.json();
                    if (data.code !== 200) return;
                    const r = data.data || {};
                    closedEyeCount = r.eye_closed ? closedEyeCount + 1 : 0;
                    if (!r.face_detected) {
                        setMorningLiveStatus(r.status || '未检测到人脸，请面向摄像头', 'warn');
                    } else if (closedEyeCount >= 2) {
                        setMorningLiveStatus('检测到连续闭眼或犯困，请睁眼并保持精神。', 'danger');
                        morningLiveHint.textContent = '提醒：检测到闭眼或犯困，屏幕已警示。';
                        setMorningLiveAlert(true, 'eye');
                    } else if (lowVolumeCount >= 8) {
                        setMorningLiveStatus('声音偏小，请靠近麦克风或提高朗读音量。', 'warn');
                        setMorningLiveAlert(true, 'voice');
                    } else {
                        setMorningLiveStatus('状态正常：眼睛睁开，声音持续监测中。');
                        setMorningLiveAlert(false);
                    }
                } catch (err) {
                    morningLiveHint.textContent = '摄像头画面检测暂时失败：' + err.message;
                }
            }

            async function startMorningLiveMonitor() {
                try {
                    morningLiveStream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480 },
                        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
                    });
                    morningLiveVideo.srcObject = morningLiveStream;
                    morningAudioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = morningAudioContext.createMediaStreamSource(morningLiveStream);
                    morningAnalyser = morningAudioContext.createAnalyser();
                    morningAnalyser.fftSize = 1024;
                    source.connect(morningAnalyser);
                    const data = new Uint8Array(morningAnalyser.fftSize);

                    morningLiveStartBtn.disabled = true;
                    morningLiveStopBtn.disabled = false;
                    morningLiveHint.textContent = '正在实时检测晨读状态。';
                    setMorningLiveStatus('实时检测已开启。');

                    morningVolumeTimer = setInterval(() => {
                        morningAnalyser.getByteTimeDomainData(data);
                        let sum = 0;
                        for (let i = 0; i < data.length; i++) {
                            const v = (data[i] - 128) / 128;
                            sum += v * v;
                        }
                        const rms = Math.sqrt(sum / data.length);
                        const volume = Math.min(100, Math.round(rms * 420));
                        morningVolumeBar.style.width = volume + '%';
                        morningVolumeBar.classList.toggle('low', volume < 8);
                        lowVolumeCount = volume < 8 ? lowVolumeCount + 1 : 0;
                        if (lowVolumeCount >= 8) {
                            morningLiveHint.textContent = '提醒：声音偏小，朗读可能识别不完整。';
                            setMorningLiveAlert(true, 'voice');
                        } else {
                            morningLiveHint.textContent = '检测中：请保持面向摄像头，朗读音量稳定。';
                            if (morningLiveAlertReason === 'voice' && closedEyeCount < 2) {
                                setMorningLiveAlert(false);
                            }
                        }
                    }, 500);

                    morningFrameTimer = setInterval(sendMorningFrameForCheck, 1800);
                } catch (err) {
                    stopMorningLiveMonitor();
                    alert('实时检测启动失败，请允许摄像头和麦克风权限：' + err.message);
                }
            }

            function bindDropUpload(boxId, inputEl, options = {}) {
                const box = document.getElementById(boxId);
                if (!box || !inputEl) return;
                const accept = options.accept || (() => true);
                const message = options.message || '文件格式不支持';
                ['dragenter', 'dragover'].forEach(eventName => {
                    box.addEventListener(eventName, (e) => {
                        e.preventDefault();
                        box.style.borderColor = '#2563eb';
                        box.style.background = '#eff6ff';
                    });
                });
                ['dragleave', 'drop'].forEach(eventName => {
                    box.addEventListener(eventName, (e) => {
                        e.preventDefault();
                        box.style.borderColor = '#bdc3c7';
                        box.style.background = '';
                    });
                });
                box.addEventListener('drop', (e) => {
                    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                    if (!file) return;
                    if (!accept(file)) {
                        alert(message);
                        return;
                    }
                    const transfer = new DataTransfer();
                    transfer.items.add(file);
                    inputEl.files = transfer.files;
                    inputEl.dispatchEvent(new Event('change', { bubbles: true }));
                });
            }

            const isMp4File = (file) => file && file.name.toLowerCase().endsWith('.mp4');
            const isImageFile = (file) => {
                if (!file) return false;
                const name = file.name.toLowerCase();
                return file.type.startsWith('image/') || ['.jpg', '.jpeg', '.png', '.webp', '.bmp'].some(ext => name.endsWith(ext));
            };

            morningReferenceImage.addEventListener('change', function () {
                if (this.files.length > 0) {
                    if (!isImageFile(this.files[0])) {
                        alert('请上传 JPG、PNG、WEBP 或 BMP 格式的原文照片');
                        this.value = '';
                        morningReferenceImageName.textContent = '未选择文件';
                        morningOcrBtn.disabled = true;
                        clearMorningReferencePreview();
                        return;
                    }
                    morningReferenceImageName.textContent = this.files[0].name;
                    morningOcrBtn.disabled = false;
                    showMorningReferencePreview(this.files[0]);
                    updateMorningOcrEvaluation(null);
                } else {
                    morningReferenceImageName.textContent = '未选择文件';
                    morningOcrBtn.disabled = true;
                    clearMorningReferencePreview();
                    updateMorningOcrEvaluation(null);
                }
            });

            morningAudioFile.addEventListener('change', function () {
                if (this.files.length > 0) {
                    if (!isMp4File(this.files[0])) {
                        alert('请上传 MP4 格式的视频文件');
                        this.value = '';
                        morningAudioFileName.textContent = '未选择文件';
                        morningAudioAnalyzeBtn.disabled = true;
                        return;
                    }
                    morningAudioFileName.textContent = this.files[0].name;
                    morningAudioAnalyzeBtn.disabled = false;
                } else {
                    morningAudioFileName.textContent = '未选择文件';
                    morningAudioAnalyzeBtn.disabled = true;
                }
            });

            morningVideoFile.addEventListener('change', function () {
                if (this.files.length > 0) {
                    if (!isMp4File(this.files[0])) {
                        alert('请上传 MP4 格式的视频文件');
                        this.value = '';
                        morningVideoFileName.textContent = '未选择文件';
                        morningVideoAnalyzeBtn.disabled = true;
                        return;
                    }
                    morningVideoFileName.textContent = this.files[0].name;
                    morningVideoAnalyzeBtn.disabled = false;
                } else {
                    morningVideoFileName.textContent = '未选择文件';
                    morningVideoAnalyzeBtn.disabled = true;
                }
            });

            bindDropUpload('morningReferenceUploadBox', morningReferenceImage, {
                accept: isImageFile,
                message: '请上传 JPG、PNG、WEBP 或 BMP 格式的原文照片'
            });
            bindDropUpload('morningAudioUploadBox', morningAudioFile, {
                accept: isMp4File,
                message: '请上传 MP4 格式的视频文件'
            });
            bindDropUpload('morningVideoUploadBox', morningVideoFile, {
                accept: isMp4File,
                message: '请上传 MP4 格式的视频文件'
            });

            morningLiveStartBtn.addEventListener('click', startMorningLiveMonitor);
            morningLiveStopBtn.addEventListener('click', stopMorningLiveMonitor);
            window.addEventListener('beforeunload', () => {
                stopMorningLiveMonitor();
                clearMorningReferencePreview();
            });

            morningOcrBtn.addEventListener('click', async () => {
                const file = morningReferenceImage.files[0];
                if (!file) { alert('请先选择晨读原文照片'); return; }

                morningOcrBtn.disabled = true;
                morningOcrBtn.innerHTML = '<div class="loading"></div> 识别中...';
                morningReferenceText.value = '正在识别原文照片，请稍候...';
                updateMorningOcrEvaluation({
                    level: '文本识别评价',
                    score: '--',
                    report: '正在评价识别文本质量...',
                    suggestions: ''
                });

                try {
                    const fd = new FormData();
                    fd.append('file', file);
                    const res = await fetch('/api/morning-reading-ocr', { method: 'POST', body: fd });
                    const data = await res.json();
                    if (data.code !== 200) throw new Error(data.msg || '晨读原文识别失败');
                    morningReferenceText.value = data.text || '';
                    updateMorningOcrEvaluation(data.evaluation);
                    if (!morningReferenceText.value.trim()) {
                        alert('没有识别到文字，请换一张更清晰的原文照片');
                    }
                } catch (err) {
                    morningReferenceText.value = '';
                    updateMorningOcrEvaluation(null);
                    alert('晨读原文识别失败：' + err.message);
                } finally {
                    morningOcrBtn.disabled = false;
                    morningOcrBtn.innerHTML = '识别晨读原文';
                }
            });

            morningAudioAnalyzeBtn.addEventListener('click', async () => {
                const file = morningAudioFile.files[0];
                if (!file) { alert('请先选择带声音的晨读 MP4 视频'); return; }

                showMorningResultBox();
                morningAudioAnalyzeBtn.disabled = true;
                morningAudioAnalyzeBtn.innerHTML = '<div class="loading"></div> 评分中...';
                document.getElementById('morningTranscript').textContent = '正在转写和评分...';
                document.getElementById('morningVoiceReport').textContent = '正在分析声音表现...';
                document.getElementById('morningContentReport').textContent = '正在分析朗读内容...';

                try {
                    const fd = new FormData();
                    fd.append('file', file);
                    fd.append('reference_text', morningReferenceText.value || '');
                    const res = await fetch('/api/morning-reading-audio', { method: 'POST', body: fd });
                    const data = await res.json();
                    if (data.code !== 200) throw new Error(data.msg || '晨读视频声音评分失败');
                    const r = data.data;

                    document.getElementById('morningTotalScore').textContent = r.total_score;
                    document.getElementById('morningVoiceScore').textContent = r.voice_score;
                    document.getElementById('morningContentScore').textContent = r.content_score;
                    document.getElementById('morningTranscript').textContent = (r.transcript || '未识别到朗读内容') + (r.transcript_note ? '\\n' + r.transcript_note : '');
                    document.getElementById('morningVoiceReport').textContent = r.voice_report;
                    document.getElementById('morningContentReport').textContent = r.content_report;
                    appendMorningSuggestion(r.suggestions);
                    updateMorningTotal();
                } catch (err) {
                    alert('晨读视频声音评分失败：' + err.message);
                } finally {
                    morningAudioAnalyzeBtn.disabled = false;
                    morningAudioAnalyzeBtn.innerHTML = '开始声音与内容评分';
                }
            });

            morningVideoAnalyzeBtn.addEventListener('click', async () => {
                const file = morningVideoFile.files[0];
                if (!file) { alert('请先选择晨读视频'); return; }

                showMorningResultBox();
                morningVideoAnalyzeBtn.disabled = true;
                morningVideoAnalyzeBtn.innerHTML = '<div class="loading"></div> 评分中...';
                document.getElementById('morningAppearanceReport').textContent = '正在识别精神面貌...';
                document.getElementById('morningPostureReport').textContent = '正在识别站姿体态...';

                try {
                    const fd = new FormData();
                    fd.append('file', file);
                    const res = await fetch('/api/morning-reading-video', { method: 'POST', body: fd });
                    const data = await res.json();
                    if (data.code !== 200) throw new Error(data.msg || '晨读视频评分失败');
                    const r = data.data;

                    document.getElementById('morningTotalScore').textContent = r.total_score;
                    document.getElementById('morningAppearanceScore').textContent = r.appearance_score;
                    document.getElementById('morningPostureScore').textContent = r.posture_score;
                    document.getElementById('morningAppearanceReport').textContent = r.appearance_report || '未检测到有效面部状态';
                    document.getElementById('morningPostureReport').textContent = r.posture_report || '未检测到有效身体姿态';
                    appendMorningSuggestion(r.suggestions);
                    updateMorningTotal();
                } catch (err) {
                    alert('晨读视频评分失败：' + err.message);
                } finally {
                    morningVideoAnalyzeBtn.disabled = false;
                    morningVideoAnalyzeBtn.innerHTML = '开始精神面貌与站姿评分';
                }
            });

            // 大录音逻辑完全不变
            startRecordBtn.addEventListener('click', async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    recordSeconds = 0;

                    mediaRecorder.ondataavailable = (e) => {
                        audioChunks.push(e.data);
                    };

                    mediaRecorder.start();
                    isRecordingMode = true;

                    startRecordBtn.disabled = true;
                    stopRecordBtn.disabled = false;
                    cancelRecordBtn.disabled = false;
                    recordTime.classList.add('recording');

                    recordTextResult.value = "正在录音，停止录音后自动转写...";
                    recordTextResult.readOnly = true;

                    recordTimer = setInterval(() => {
                        recordSeconds++;
                        recordTime.textContent = formatTime(recordSeconds);
                    }, 1000);

                } catch (err) {
                    alert("录音启动失败，请允许麦克风权限");
                    console.error(err);
                }
            });

            stopRecordBtn.addEventListener('click', async () => {
                clearInterval(recordTimer);
                try { mediaRecorder.stop(); } catch (e) {}
                recordTime.classList.remove('recording');

                startRecordBtn.disabled = false;
                stopRecordBtn.disabled = true;
                cancelRecordBtn.disabled = false;

                setTimeout(async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const audioFile = new File([audioBlob], `record_${Date.now()}.webm`, { type: 'audio/webm' });

                    recordTextResult.value = "正在转写中，请稍候...";

                    try {
                        const formData = new FormData();
                        formData.append('file', audioFile);
                        const res = await fetch('/api/record-audio-to-text', { method: 'POST', body: formData });
                        const data = await res.json();

                        if (data.code !== 200) throw new Error(data.msg || '转写失败');
                        recordTextResult.value = data.text;
                        recordTextResult.readOnly = false;

                    } catch (err) {
                        alert('转写失败：' + err.message);
                        recordTextResult.value = "转写失败，请重新录音";
                    }
                }, 500);
            });

            cancelRecordBtn.addEventListener('click', () => {
                clearInterval(recordTimer);
                try { mediaRecorder.stop(); } catch (e) {}
                isRecordingMode = false;
                recordSeconds = 0;
                recordTime.textContent = "00:00";
                recordTextResult.value = "";
                startRecordBtn.disabled = false;
                stopRecordBtn.disabled = true;
                cancelRecordBtn.disabled = true;
                recordTime.classList.remove('recording');
            });

            // 音频上传
            const audioFile = document.getElementById('audioFile');
            const audioAnalyzeBtn = document.getElementById('audioAnalyzeBtn');
            const audioFileName = document.getElementById('audioFileName');
            audioFile.addEventListener('change', function () {
                if (this.files.length > 0) {
                    audioFileName.textContent = this.files[0].name;
                    audioAnalyzeBtn.disabled = false;
                } else {
                    audioFileName.textContent = '未选择文件';
                    audioAnalyzeBtn.disabled = true;
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const file = audioFile.files[0];
                if (!file) return;

                audioAnalyzeBtn.disabled = true;
                audioAnalyzeBtn.innerHTML = '<div class="loading"></div> 分析中...';
                audioResultBox.style.display = 'none';
                modalResultBox.style.display = 'none';
                extractResultBox.style.display = 'none';
                hireBox.style.display = 'none';

                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    const res = await fetch('/api/upload-audio', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (data.code !== 200) throw new Error(data.msg || '分析失败');
                    const r = data.data;

                    document.getElementById('audioScore').textContent = r.score;
                    document.getElementById('audioTranscript').textContent = r.transcript;
                    document.getElementById('audioAnalysis').textContent = r.analysis;
                    document.getElementById('audioSuggestions').textContent = r.suggestions;
                    document.getElementById('audioHighScore').textContent = r.high_score_answer;
                    audioResultBox.style.display = 'block';
                } catch (err) {
                    alert('分析失败：' + err.message);
                } finally {
                    audioAnalyzeBtn.disabled = false;
                    audioAnalyzeBtn.innerHTML = '🚀 开始AI分析';
                }
            });

            // 多模态分析
            const modalVideoFile = document.getElementById('modalVideoFile');
            const modalAnalyzeBtn = document.getElementById('modalAnalyzeBtn');
            const modalVideoFileName = document.getElementById('modalVideoFileName');
            modalVideoFile.addEventListener('change', function () {
                if (this.files.length > 0) {
                    modalVideoFileName.textContent = this.files[0].name;
                }
            });
            modalAnalyzeBtn.addEventListener('click', async () => {
                const file = modalVideoFile.files[0];
                if (!file) { alert('请先选择视频！'); return; }

                modalAnalyzeBtn.disabled = true;
                modalAnalyzeBtn.innerHTML = '<div class="loading"></div> 分析中...';
                modalResultBox.style.display = 'none';
                audioResultBox.style.display = 'none';
                extractResultBox.style.display = 'none';
                hireBox.style.display = 'none';

                const fd = new FormData();
                fd.append('file', file);
                const res = await fetch('/api/multimodal-analysis', { method: 'POST', body: fd });
                const data = await res.json();

                if (data.code === 200) {
                    document.getElementById('modalFace').textContent = data.result.表情状态;
                    document.getElementById('modalPose').textContent = data.result.肢体动作;
                    document.getElementById('modalEval').textContent = data.ai_report;
                    modalResultBox.style.display = 'block';
                } else {
                    alert('❌ 分析失败：' + data.msg);
                }
                modalAnalyzeBtn.disabled = false;
                modalAnalyzeBtn.innerHTML = '🎨 多模态分析';
            });

            // 视频提取音频分析
            const extractVideoFile = document.getElementById('extractVideoFile');
            const extractAnalyzeBtn = document.getElementById('extractAnalyzeBtn');
            const extractVideoFileName = document.getElementById('extractVideoFileName');
            extractVideoFile.addEventListener('change', function(){
                if(this.files.length>0) {
                    isRecordingMode = false;
                    extractVideoFileName.textContent = this.files[0].name;
                }
            });

            extractAnalyzeBtn.addEventListener('click', async ()=>{
                extractResultBox.style.display = 'none';
                audioResultBox.style.display = 'none';
                modalResultBox.style.display = 'none';
                hireBox.style.display = 'none';
                trainQuestionSection.style.display = 'none';
                trainEvalSection.style.display = 'none';
                trainHistorySection.style.display = 'none';
                trainSessionId = null;

                extractAnalyzeBtn.disabled = true;
                extractAnalyzeBtn.innerHTML = '<div class="loading"></div> 分析中...';

                try {
                    let res;
                    if (isRecordingMode) {
                        const interviewText = recordTextResult.value.trim();
                        if (!interviewText || interviewText.length < 10) {
                            alert("请先完成录音，确保面试内容不为空！");
                            return;
                        }
                        const formData = new FormData();
                        formData.append('interview_text', interviewText);
                        res = await fetch('/api/text_train_score', { method: 'POST', body: formData });
                    } else {
                        const file = extractVideoFile.files[0];
                        if (!file) {
                            alert("请先上传视频或录音！");
                            return;
                        }
                        const formData = new FormData();
                        formData.append('file', file);
                        res = await fetch('/api/video_train_score', { method: 'POST', body: formData });
                    }

                    const data = await res.json();
                    if (data.code !== 200) throw new Error(data.msg || '分析失败');

                    document.getElementById('extractScore').textContent = `总分：${data.three_six_score.总分}分`;
                    document.getElementById('extractTranscript').textContent = data.transcript;
                    document.getElementById('extractAnalysis').textContent = data.ai_report;
                    document.getElementById('extractSuggestions').textContent = data.ai_report;
                    document.getElementById('extractHighScore').textContent = data.ai_report;

                    if (data.hire_result) {
                        document.getElementById('hireLevel').textContent = data.hire_result.hire_level;
                        document.getElementById('hireReport').textContent = data.hire_result.hire_report;
                        hireBox.style.display = 'block';
                    }

                    if (data.need_train) {
                        trainQuestionSection.style.display = 'block';
                        trainEvalSection.style.display = 'block';
                        trainHistorySection.style.display = 'block';
                        trainSessionId = data.session_id;
                        extractCurrentQuestion.textContent = data.current_question;
                        extractLastEvaluation.textContent = '暂无评价';
                        extractTrainHistory.textContent = '暂无记录';
                    }
                    extractResultBox.style.display = 'block';
                } catch (err) {
                    alert('分析失败：' + err.message);
                } finally {
                    extractAnalyzeBtn.disabled = false;
                    extractAnalyzeBtn.innerHTML = '✅ 一键提取音频并AI点评';
                }
            });

            // 提交练习题
            extractSubmitAnswerBtn.addEventListener('click', async () => {
                if (!trainSessionId) {
                    alert('请先完成视频/录音分析！');
                    return;
                }
                const answer = extractAnswerInput.value.trim();
                if (!answer) {
                    alert('请输入你的回答！');
                    return;
                }

                extractSubmitAnswerBtn.disabled = true;
                extractSubmitAnswerBtn.textContent = '提交中...';

                try {
                    const formData = new FormData();
                    formData.append('session_id', trainSessionId);
                    formData.append('user_answer', answer);
                    const res = await fetch('/api/train_submit', { method: 'POST', body: formData });
                    const data = await res.json();

                    if (data.code === 400) {
                        alert(data.msg);
                        trainSessionId = null;
                        return;
                    }

                    extractLastEvaluation.textContent = data.last_evaluation || data.evaluation;
                    extractTrainHistory.textContent = JSON.stringify(data.history, null, 2);

                    if (data.status === 'finish') {
                        extractCurrentQuestion.textContent = '✅ 训练完成！';
                        extractSubmitAnswerBtn.disabled = true;
                        trainSessionId = null;
                        alert('完成全部练习');
                    } else {
                        document.querySelector('#trainQuestionSection h4').textContent = `📌 练习题（第${data.current_done + 1}/3题）`;
                        extractCurrentQuestion.textContent = data.next_question;
                        extractAnswerInput.value = '';
                    }
                } catch (err) {
                    alert('提交失败：' + err.message);
                } finally {
                    if (trainSessionId) extractSubmitAnswerBtn.disabled = false;
                    extractSubmitAnswerBtn.textContent = '提交答案并获取AI评价';
                }
            });
        });
    </script>
</body>
</html>
    """


# ====================== 启动服务 ======================
if __name__ == "__main__":
    from config import settings as app_settings

    app_dir = str(Path(__file__).resolve().parent)
    os.chdir(app_dir)
    preferred = getattr(app_settings, "SERVER_PORT", 28080)
    port = pick_available_port(preferred)
    if port != preferred:
        print(f"[提示] 端口 {preferred} 被占用，已改用 {port}", flush=True)
    print(f"[启动] 面试系统 → http://127.0.0.1:{port}/", flush=True)
    print(f"[启动] 实时面试 → http://127.0.0.1:{port}/live", flush=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        reload=False,
    )
