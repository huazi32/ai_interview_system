"""百炼实时面对面面试：WebSocket 代理 + MD 报告下载。"""
import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from config import settings
from core.bailian_dialog_bridge import (
    BailianInterviewSession,
    format_bailian_error,
    live_reports,
    live_sessions,
)
from service.live_interview_report import build_interview_markdown
from service.resume_text import normalize_resume_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["live-interview"])

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


@router.get("/api/live/status")
async def live_status():
    """检查百炼配置是否就绪（不发起计费请求）。"""
    app_id = (settings.BAILIAN_APP_ID or "").strip()
    ws_id = (settings.BAILIAN_WORKSPACE_ID or "").strip()
    model = getattr(settings, "BAILIAN_MODEL", "multimodal-dialog")
    ok = bool(app_id and ws_id and not ws_id.startswith("mm_"))
    return {
        "code": 200,
        "ready": ok,
        "app_id_set": bool(app_id),
        "workspace_id_set": bool(ws_id),
        "model": model,
        "hint": None if ok else "请在 config/settings.py 填写 BAILIAN_APP_ID 与 BAILIAN_WORKSPACE_ID",
        "billing_hint": (
            "若出现 BillingAuthError，请在百炼控制台开通「多模态交互开发套件」（按量后付费），"
            "并确保 API Key 来自默认业务空间、账户余额 ≥ 0。"
        ),
    }


@router.get("/api/live/report/{session_id}")
async def download_live_report(session_id: str):
    md = live_reports.get(session_id)
    if not md:
        return Response(content="报告不存在或已过期", status_code=404)
    filename = f"interview_report_{session_id[:8]}.md"
    return Response(
        content=md.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.websocket("/ws/live-interview")
async def live_interview_ws(websocket: WebSocket):
    await websocket.accept()
    session: Optional[BailianInterviewSession] = None
    pump_task = None

    async def pump_to_client(sess: BailianInterviewSession):
        try:
            async for event in sess.pump_events():
                if event.get("type") == "ai_audio":
                    await websocket.send_json(event)
                elif event.get("type") in (
                    "state", "user_text", "ai_text", "error",
                    "started", "responding_started", "responding_ended",
                    "speech_started", "speech_ended", "speech_ready", "speech_stopped",
                    "thinking",
                    "ai_prompt",
                    "input_mode",
                    "connected", "closed", "stopped", "video_ready",
                ):
                    safe = {k: v for k, v in event.items() if k != "payload"}
                    if safe.get("type") == "error":
                        safe["msg"] = format_bailian_error(safe.get("msg", ""))
                    await websocket.send_json(safe)
        except Exception as e:
            logger.exception("pump error: %s", e)
            try:
                await websocket.send_json({"type": "error", "msg": str(e)})
            except Exception:
                pass

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "start":
                job = msg.get("job", "CV算法工程师")
                user_id = msg.get("user_id", "candidate")
                resume_text = normalize_resume_text(msg.get("resume_text", ""))
                session_id = str(uuid.uuid4())
                loop = asyncio.get_event_loop()
                session = BailianInterviewSession(
                    session_id=session_id,
                    job_title=job,
                    user_id=user_id,
                    resume_text=resume_text,
                    loop=loop,
                    input_mode=msg.get("input_mode", ""),
                )
                live_sessions[session_id] = session

                try:
                    await session.run_io(session.start)
                except Exception as e:
                    await websocket.send_json({"type": "error", "msg": format_bailian_error(e)})
                    session.stop()
                    continue

                pump_task = asyncio.create_task(pump_to_client(session))
                await websocket.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "msg": "已连接百炼实时面试，请允许摄像头和麦克风",
                    "has_resume": bool(resume_text),
                    "input_mode": session.get_client_input_mode(),
                })

            elif mtype == "set_input_mode" and session:
                mode = await session.run_io(
                    session.set_client_input_mode, msg.get("mode", "ptt"),
                )
                label = "实时对话" if mode == "realtime" else "按住说话"
                await websocket.send_json({
                    "type": "input_mode",
                    "mode": mode,
                    "msg": f"已切换为{label}模式，对话继续",
                })

            elif mtype == "speech_start" and session:
                if session.get_client_input_mode() == "realtime":
                    await websocket.send_json({"type": "speech_ready"})
                else:
                    session._ptt_open = True
                    await session.run_io(session.start_speech)
                    await websocket.send_json({"type": "speech_ready"})

            elif mtype == "speech_end" and session:
                if session.get_client_input_mode() == "realtime":
                    break
                await session.run_io(session.notify_playback_ended)
                await asyncio.sleep(0.08)
                await session.run_io(session.stop_speech)
                await websocket.send_json({"type": "speech_stopped"})
                await websocket.send_json({"type": "thinking", "msg": "AI 正在理解并准备回复…"})

            elif mtype == "speech_cancel" and session:
                if session.get_client_input_mode() == "realtime":
                    break
                await session.run_io(session.cancel_speech)
                await websocket.send_json({"type": "speech_stopped"})

            elif mtype == "audio" and session:
                pcm = base64.b64decode(msg.get("data", ""))
                await session.run_io(session.send_audio_pcm, pcm)

            elif mtype == "video_frame" and session:
                await session.run_io(session.send_video_frame, msg.get("data", ""))

            elif mtype == "playback_started" and session:
                await session.run_io(session.notify_playback_started)

            elif mtype == "playback_ended" and session:
                await session.run_io(session.notify_playback_ended)

            elif mtype == "end_interview" and session:
                session.stop()
                if pump_task:
                    pump_task.cancel()
                    try:
                        await pump_task
                    except asyncio.CancelledError:
                        pass

                md = build_interview_markdown(
                    session_id=session.session_id,
                    job_title=session.job_title,
                    transcript=session.transcript,
                    started_at=session.started_at,
                    dialog_id=session.dialog_id,
                    resume_text=session.resume_text,
                )
                live_reports[session.session_id] = md

                await websocket.send_json({
                    "type": "report_ready",
                    "session_id": session.session_id,
                    "download_url": f"/api/live/report/{session.session_id}",
                    "preview": md[:2000] + ("..." if len(md) > 2000 else ""),
                })
                session = None
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("ws error: %s", e)
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        if session:
            session.stop()
        if pump_task and not pump_task.done():
            pump_task.cancel()
