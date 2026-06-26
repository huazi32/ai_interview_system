from fastapi import APIRouter, UploadFile, File
from core.asr_service import audio_to_text
from core.video_visual_service import analyze_video_visual
from core.llm_service import generate_multimodal_ai_review
import subprocess
import uuid
import os

router = APIRouter()

# ======================
# 🔥 视频多模态分析接口（真正打通：语音 + 面部 + 姿态 + 千问）
# ======================
@router.post("/video_multimodal_analysis")
async def video_multimodal_analysis(file: UploadFile = File(...)):
    try:
        # 1. 保存视频
        video_path = f"temp_video_{uuid.uuid4().hex[:8]}.mp4"
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # 2. 提取音频
        audio_path = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1", "-y", audio_path
            ],
            capture_output=True
        )

        # 3. ASR 语音识别
        text = audio_to_text(audio_path)

        # 4. 面部 + 姿态真实分析
        visual = analyze_video_visual(video_path)

        # 5. 多模态大模型生成报告
        report = generate_multimodal_ai_review(
            text=text,
            expression=visual["表情状态"],
            pose=visual["肢体动作"]
        )

        # 清理临时文件
        os.remove(video_path)
        os.remove(audio_path)

        return {
            "code": 200,
            "语音转写": text,
            "面部表情": visual["表情状态"],
            "肢体姿态": visual["肢体动作"],
            "多模态评测报告": report
        }

    except Exception as e:
        return {"code": 500, "msg": str(e)}