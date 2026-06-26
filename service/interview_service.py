from fastapi import UploadFile
from core.asr_service import audio_to_text
from core.llm_service import generate_ai_review

async def analyze_audio(file: UploadFile):
    try:
        # 1. 真实 Whisper 语音识别
        transcript = await audio_to_text(file)

        # 2. 真实 AI 分析
        ai_result = generate_ai_review("面试问题", transcript)

        return {
            "code": 200,
            "msg": "success",
            "data": {
                "score": "88",
                "transcript": transcript,
                "analysis": ai_result,
                "suggestions": ai_result,
                "high_score_answer": ai_result
            }
        }
    except Exception as e:
        return {"code":500,"msg":str(e),"data":None}