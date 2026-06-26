"""百炼 MultiModalDialog WebSocket 桥接：浏览器 ↔ DashScope 实时对话。"""
import asyncio
import base64
import concurrent.futures
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import dashscope
from dashscope.multimodal.multimodal_dialog import MultiModalDialog, MultiModalCallback
from dashscope.multimodal.multimodal_request_params import (
    BizParams,
    ClientInfo,
    Device,
    DialogAttributes,
    Downstream,
    RequestParameters,
    RequestToRespondParameters,
    Upstream,
)
from dashscope.multimodal import dialog_state

from config import settings
from core.asr_term_correction import build_asr_post_processing, correct_user_asr_text

logger = logging.getLogger(__name__)

# 百炼错误码 → 中文说明与处理建议
_BAILIAN_ERROR_HINTS = {
    "BillingAuthError": (
        "百炼「多模态交互开发套件」未开通或未激活。\n"
        "请登录 https://bailian.console.aliyun.com/ → 右上角「免费开通」→ 勾选协议 →「立即购买」（按量后付费）。\n"
        "确保阿里云账户余额 ≥ 0 元，且 API Key 来自默认业务空间。"
    ),
    "AccessDenied.Unpurchased": (
        "未开通阿里云百炼服务。请前往百炼控制台开通服务后再试。"
    ),
    "Model.AccessDenied": (
        "多模态交互仅支持默认业务空间的 API Key。\n"
        "请在百炼控制台 → API-Key 页面，使用「默认业务空间」创建的 Key，"
        "并填写该空间的 Workspace ID。"
    ),
    "InvalidParameter": "请求参数不符合百炼协议，请检查 config/settings.py 与 SDK 版本。",
    "TooManyInterrupt": (
        "向百炼发送控制指令过于频繁（通常是视频帧发太快）。"
        "已自动限速；若仍出现请增大 BAILIAN_VIDEO_FRAME_INTERVAL_MS 或暂时关闭视频。"
    ),
}

FALLBACK_NOT_HEARD = "抱歉，刚才没听清您的声音，麻烦重新说一遍。"
_NO_SPEECH_MARKERS = ("NoSpeechRecognized", "nospeechrecognized", "未识别到语音", "no speech")


def is_asr_no_speech_error(raw: Any) -> bool:
    """是否为 ASR 未识别到语音（含 451 / NoSpeechRecognized）。"""
    text = raw if isinstance(raw, str) else str(raw)
    lower = text.lower()
    if any(m.lower() in lower for m in _NO_SPEECH_MARKERS):
        return True
    try:
        import json
        data = json.loads(text) if text.strip().startswith("{") else None
    except Exception:
        data = None
    if isinstance(data, dict):
        header = data.get("header") or {}
        code = str(
            header.get("error_code")
            or header.get("status_name")
            or header.get("status_code")
            or ""
        )
        if code in ("NoSpeechRecognized", "451") or "nospeech" in code.lower():
            return True
        payload = data.get("payload") or {}
        output = payload.get("output") or {}
        if str(output.get("event", "")).lower() == "error":
            ev_code = str(output.get("code") or output.get("error_code") or "")
            if "nospeech" in ev_code.lower() or ev_code in ("451", "NoSpeechRecognized"):
                return True
    if "451" in text and ("speech" in lower or "recogn" in lower or "nospeech" in lower):
        return True
    return False


def is_recoverable_audio_error(raw: Any) -> bool:
    """可恢复音频类错误：不断开会话，提示用户重说即可。"""
    if is_asr_no_speech_error(raw):
        return True
    text = raw if isinstance(raw, str) else str(raw)
    lower = text.lower()
    return (
        "clientaudiotimeout" in lower
        or "waiting for client audio" in lower
        or "no speech recognized" in lower
    )


def format_bailian_error(raw: Any) -> str:
    """将百炼 SDK 原始错误转为可读中文提示。"""
    text = raw if isinstance(raw, str) else str(raw)
    try:
        import json
        data = json.loads(text) if text.strip().startswith("{") else None
    except Exception:
        data = None

    if isinstance(data, dict):
        header = data.get("header") or {}
        code = header.get("error_code") or header.get("status_name") or ""
        msg = header.get("error_message") or header.get("status_message") or ""
        hint = _BAILIAN_ERROR_HINTS.get(code, "")
        if hint:
            return f"[{code}] {msg or '服务不可用'}\n\n{hint}"
        if code or msg:
            return f"[{code}] {msg}" if code else msg

    for code, hint in _BAILIAN_ERROR_HINTS.items():
        if code in text:
            return f"{text}\n\n{hint}"

    if "TooManyInterrupt" in text:
        return f"[TooManyInterrupt] {_BAILIAN_ERROR_HINTS['TooManyInterrupt']}"

    lower = text.lower()
    if "blocking send" in lower or "thread was interrupted" in lower:
        return (
            "百炼连接发送冲突，WebSocket 已断开。\n"
            "请刷新页面重新开始面试；若反复出现，请在 settings.py 将 BAILIAN_ENABLE_VIDEO 设为 False。"
        )
    if "internal server error" in lower or "internalerror" in lower:
        return (
            "百炼服务内部错误，连接已断开。\n"
            "请刷新页面重新开始；若频繁出现，请检查网络或暂时关闭视频上传（BAILIAN_ENABLE_VIDEO = False）。"
        )
    if "opcode=8" in lower or "response timeout" in lower or "responsetimeout" in lower:
        return "百炼连接超时或已断开，请刷新页面重新开始面试。"
    if "clientaudiotimeout" in lower or "waiting for client audio" in lower:
        return "刚才没有及时收到你的声音，请直接再试一次。"

    if "Invalid payload data" in text or "websocket is closed" in lower:
        return "百炼 WebSocket 已断开（请查看上方第一条错误原因）。"
    if "Connection is already closed" in text:
        return "百炼连接已关闭（请查看上方第一条错误原因）。"
    return text

INTERVIEW_PROMPT = """你是一位拥有十年经验的互联网公司技术面试官，正与候选人进行实时语音面试（面对面说话，不是文字聊天）。

【岗位】应聘岗位：{job_title}。考察重点围绕该岗位的核心技能、项目深度与工程落地能力；若岗位偏 CV，可侧重检测/分割/跟踪/部署；若偏深度学习或多模态，则相应调整，勿偏题。

【输出形式 · 必守】
- 你的每句话都会直接转成语音，必须是口语化中文，像真人面对面交谈
- 单次回复控制在 2～4 短句、约 80 字以内：先承接对方刚才说的（一句），再问一个核心问题（一句）
- 禁止：Markdown、编号列表、JSON、括号旁白、「作为面试官我认为」等书面套话
- 禁止：一次抛出多个问题；若有多个子要点，分多轮逐条追问
- 禁止：压迫式措辞（「请详细说明」「回答不够具体」「还不够」）

【对话节奏】
- 候选人按住说话、说完松手后，你再回复；不要抢话、不要连珠炮
- 流程：开场问候 → 一分钟自我介绍 → 项目/经历深挖 → 专业基础 → 工程与协作 → 收尾
- 全程约 5～8 轮有效问答；信息已充分时自然过渡，勿机械卡在同一话题

【像真人一样追问】
- 上一问若有子要点未答（如只答了数据规模、没答增强策略），下一轮必须先补问未答部分，再换题
- 仅当候选人明确说「没做/不记得/不了解」时，才放弃该子点
- 简历或回答里出现具体项目/指标，但口述太笼统（「调包」「常规流程」）时，追问一个关键点：数据规模、指标数值、你的分工、选型理由、瓶颈与优化
- 每次只追一个点；先简短认可已有信息，再好奇地补问，如：「嗯，类别分布这块清楚了，那数据增强你们当时怎么做的？」
- 若候选人指出「你跳题了」「那个点没追问」，先承认并补问被跳过的内容，再推进

【何时不再追问、换题推进】
- 该点已答得具体（有方法名、数字、步骤），足以判断水平
- 候选人明确表示未参与或不了解
- 同一子话题连续两轮无新信息 → 给台阶，换同项目下一环节或新项目

【角色边界】
- 只提问与点评方向，不代答、不泄露标准答案
- 仅围绕简历、岗位与面试相关内容；遇无关请求，礼貌拉回面试

【收尾】
- 考察充分后，用一句自然口语结束，如：「好的，今天的面试先到这，感谢你参加。」"""


def _build_interview_system_prompt(job_title: str, resume_text: str = "") -> str:
    """组装完整系统 Prompt（岗位 + 可选简历）。"""
    prompt = INTERVIEW_PROMPT.format(job_title=job_title or "算法工程师")
    if resume_text:
        prompt += (
            f"\n\n【候选人简历摘要】\n{resume_text[:2000]}\n\n"
            "请结合简历中的项目、技能与成果针对性提问；不要照读简历，不要假设简历未写明的细节。"
        )
    return prompt


def _extract_payload_text(payload: Any) -> str:
    if not payload:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    output = payload.get("output") or payload
    if not isinstance(output, dict):
        return str(output).strip() if output else ""
    for key in ("text", "content", "transcript", "sentence"):
        val = output.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            nested = val.get("text") or val.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    extra = output.get("extra_info") or payload.get("extra_info") or {}
    if isinstance(extra, dict):
        for key in ("text", "transcript", "content"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


_OPENING_PROMPT_MARKERS = ("请开始面试", "向候选人做简短开场问候")


def _is_internal_prompt_text(text: str) -> bool:
    """过滤开场 trigger 等内部 prompt，避免显示为用户发言。"""
    t = (text or "").strip()
    return any(m in t for m in _OPENING_PROMPT_MARKERS)


def _extract_stream_meta(payload: Any) -> tuple[str, bool]:
    """从百炼 payload 提取 round_id / finished，用于前端单气泡更新。"""
    if not isinstance(payload, dict):
        return "", False
    output = payload.get("output") or {}
    if not isinstance(output, dict):
        return "", False
    msg_id = (
        output.get("round_id")
        or output.get("llm_request_id")
        or output.get("request_id")
        or ""
    )
    return str(msg_id), bool(output.get("finished"))


class InterviewDialogCallback(MultiModalCallback):
    """将百炼 SDK 回调转发到 asyncio 队列（SDK 运行在独立线程）。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        job_title: str = "",
        resume_text: str = "",
    ):
        self._loop = loop
        self._queue = queue
        self.dialog_id: Optional[str] = None
        self._last_user_asr = ""
        self._job_title = job_title or ""
        self._resume_text = resume_text or ""

    def _correct_asr(self, text: str) -> str:
        if not getattr(settings, "BAILIAN_ASR_TERM_CORRECTION", True):
            return text
        return correct_user_asr_text(text, self._job_title, self._resume_text)

    def _emit(self, event: dict):
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(event), self._loop)
        except RuntimeError:
            pass

    def on_connected(self) -> None:
        self._emit({"type": "connected"})

    def on_started(self, dialog_id: str) -> None:
        self.dialog_id = dialog_id
        self._emit({"type": "started", "dialog_id": dialog_id})

    def on_stopped(self) -> None:
        self._emit({"type": "stopped"})

    def on_state_changed(self, state: dialog_state.DialogState) -> None:
        self._emit({"type": "state", "state": state.value if state else "Unknown"})

    def on_speech_audio_data(self, data: bytes) -> None:
        self._emit({
            "type": "ai_audio",
            "data": base64.b64encode(data).decode("ascii"),
            "sample_rate": settings.BAILIAN_DOWNSTREAM_SAMPLE_RATE,
        })

    def on_error(self, error):
        if is_recoverable_audio_error(error):
            self._emit({
                "type": "ai_prompt",
                "text": FALLBACK_NOT_HEARD,
                "reason": "recoverable_audio",
            })
            return
        self._emit({"type": "error", "msg": format_bailian_error(error)})

    def on_responding_started(self):
        self._emit({"type": "responding_started"})

    def on_responding_ended(self, payload):
        self._emit({"type": "responding_ended"})

    def on_speech_started(self):
        self._last_user_asr = ""
        self._emit({"type": "speech_started"})

    def on_speech_content(self, payload):
        text = _extract_payload_text(payload)
        if _is_internal_prompt_text(text):
            return
        msg_id, finished = _extract_stream_meta(payload)
        if text:
            text = self._correct_asr(text)
            self._last_user_asr = text
            self._emit({
                "type": "user_text",
                "text": text,
                "message_id": msg_id,
                "finished": finished,
            })
        else:
            logger.debug("speech_content empty payload keys: %s", list(payload.keys()) if isinstance(payload, dict) else type(payload))

    def on_responding_content(self, payload):
        text = _extract_payload_text(payload)
        msg_id, finished = _extract_stream_meta(payload)
        if text:
            self._emit({
                "type": "ai_text",
                "text": text,
                "message_id": msg_id,
                "finished": finished,
            })

    def on_speech_ended(self):
        self._emit({"type": "speech_ended"})
        last = getattr(self, "_last_user_asr", "")
        if last and not _is_internal_prompt_text(last):
            last = self._correct_asr(last)
            self._emit({
                "type": "user_text",
                "text": last,
                "message_id": "",
                "finished": True,
            })
        self._last_user_asr = ""

    def on_request_accepted(self):
        self._emit({"type": "request_accepted"})

    def on_close(self, close_status_code, close_msg):
        msg = format_bailian_error(close_msg) if close_msg else "连接已关闭"
        self._emit({"type": "closed", "code": close_status_code, "msg": msg})


class BailianInterviewSession:
    """单次实时面试会话。"""

    def __init__(
        self,
        session_id: str,
        job_title: str,
        user_id: str,
        loop: asyncio.AbstractEventLoop,
        resume_text: str = "",
        on_transcript: Optional[Callable[[str, str], None]] = None,
        input_mode: str = "",
    ):
        self.session_id = session_id
        self.job_title = job_title
        self.user_id = user_id
        self.resume_text = (resume_text or "").strip()
        self.started_at = datetime.now()
        self.transcript: List[Dict[str, str]] = []
        self.dialog_id: Optional[str] = None
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._on_transcript = on_transcript
        self._dialog: Optional[MultiModalDialog] = None
        self._callback: Optional[InterviewDialogCallback] = None
        self._video_connected = False
        self._listening = False
        self._closed = False
        self._io_lock = threading.RLock()
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"bailian-{session_id[:8]}",
        )
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_user_fragment = ""
        self._last_ai_fragment = ""
        self._started = False
        self._respond_lock = threading.Lock()
        self._last_respond_at = 0.0
        self._pending_video_b64: Optional[str] = None
        self._video_interval = getattr(settings, "BAILIAN_VIDEO_FRAME_INTERVAL_MS", 1000) / 1000.0
        self._enable_video = getattr(settings, "BAILIAN_ENABLE_VIDEO", True)
        default_mode = (input_mode or getattr(settings, "BAILIAN_AUDIO_MODE", "push2talk")).strip().lower()
        if default_mode in ("realtime", "duplex", "continuous"):
            self._client_input_mode = "realtime"
        else:
            self._client_input_mode = "ptt"
        self._upstream_mode = getattr(settings, "BAILIAN_UPSTREAM_MODE", "duplex")
        self._opening_sent = False
        self._audio_streaming = False
        self._user_turn_open = False
        self._ai_responding = False
        self._ptt_open = False
        self._current_state = ""
        self._nudge_timer: Optional[threading.Timer] = None

    async def run_io(self, func: Callable, *args, **kwargs):
        """百炼 SDK WebSocket 非线程安全：所有读写必须经单线程 + 锁串行化。"""
        if self._closed:
            return None

        def _wrapped():
            with self._io_lock:
                if self._closed:
                    return None
                return func(*args, **kwargs)

        return await self._loop.run_in_executor(self._io_pool, _wrapped)

    def _start_heartbeat(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        interval = getattr(settings, "BAILIAN_HEARTBEAT_INTERVAL_SEC", 25)
        while not self._closed:
            await asyncio.sleep(interval)
            if self._closed or not self._dialog:
                break
            try:
                await self.run_io(self._send_heartbeat)
            except Exception as e:
                logger.debug("heartbeat skipped: %s", e)

    def _send_heartbeat(self):
        with self._io_lock:
            if self._dialog and not self._closed:
                self._dialog.send_heart_beat()

    def _append_transcript(self, role: str, text: str):
        text = text.strip()
        if not text:
            return
        if self.transcript and self.transcript[-1]["role"] == role:
            prev = self.transcript[-1]["text"]
            if text.startswith(prev) or prev.startswith(text) or prev in text:
                self.transcript[-1]["text"] = text
            elif role == "候选人" and self._user_turn_open:
                self.transcript[-1]["text"] = text
            elif role == "面试官" and self._ai_responding:
                self.transcript[-1]["text"] = text
            else:
                self.transcript[-1]["text"] += text
        else:
            self.transcript.append({
                "role": role,
                "text": text,
                "time": datetime.now().strftime("%H:%M:%S"),
            })
        if self._on_transcript:
            self._on_transcript(role, text)

    def _build_interview_prompt(self) -> str:
        return _build_interview_system_prompt(self.job_title, self.resume_text)

    def _build_request_params(self) -> RequestParameters:
        downstream = Downstream(
            sample_rate=settings.BAILIAN_DOWNSTREAM_SAMPLE_RATE,
            intermediate_text="transcript,dialog",
            audio_format="pcm",
        )
        if settings.BAILIAN_VOICE:
            downstream.voice = settings.BAILIAN_VOICE

        upstream = Upstream(
            type="AudioAndVideo" if self._enable_video else "AudioOnly",
            mode=self._upstream_mode if self._upstream_mode in ("duplex", "push2talk", "tap2talk") else "duplex",
            audio_format="pcm",
            sample_rate=settings.BAILIAN_UPSTREAM_SAMPLE_RATE,
        )
        vocab_id = (getattr(settings, "BAILIAN_VOCABULARY_ID", "") or "").strip()
        if vocab_id:
            upstream.vocabulary_id = vocab_id
        if getattr(settings, "BAILIAN_ASR_TERM_CORRECTION", True):
            asr_pp = build_asr_post_processing(self.job_title, self.resume_text)
            if asr_pp:
                upstream.asr_post_processing = asr_pp
        client_info = ClientInfo(
            user_id=self.user_id or "interview_user",
            device=Device(uuid=str(uuid.uuid4())),
        )
        dialog_attributes = DialogAttributes(prompt=self._build_interview_prompt())
        resume_snip = self.resume_text[:2000] if self.resume_text else ""
        biz_params = BizParams(
            user_prompt_params={
                "job_title": self.job_title,
                "position": self.job_title,
                "resume": resume_snip,
            },
        )
        return RequestParameters(
            upstream=upstream,
            downstream=downstream,
            client_info=client_info,
            dialog_attributes=dialog_attributes,
            biz_params=biz_params,
        )

    def start(self):
        dashscope.api_key = settings.DASHSCOPE_API_KEY
        app_id = (settings.BAILIAN_APP_ID or "").strip()
        ws_id = (settings.BAILIAN_WORKSPACE_ID or "").strip()
        if not app_id or not ws_id:
            raise ValueError(
                "请先在 config/settings.py 配置百炼两个 ID：\n"
                "· BAILIAN_APP_ID：应用卡片上的 mm_ 开头 ID\n"
                "· BAILIAN_WORKSPACE_ID：控制台右上角「业务空间 ID」（通常 llm- 开头）\n"
                "二者不能相同，详见 help.aliyun.com/zh/model-studio/obtain-the-app-id-and-workspace-id"
            )
        if app_id == ws_id or ws_id.startswith("mm_"):
            raise ValueError(
                "BAILIAN_WORKSPACE_ID 填错了：你填的是 APP ID（mm_ 开头）。\n"
                "请到百炼控制台右上角点击头像/业务空间 → 复制「业务空间 ID」（llm- 开头），"
                "填到 config/settings.py 的 BAILIAN_WORKSPACE_ID。"
            )

        self._callback = InterviewDialogCallback(
            self._loop,
            self._queue,
            job_title=self.job_title,
            resume_text=self.resume_text,
        )
        self._dialog = MultiModalDialog(
            app_id=settings.BAILIAN_APP_ID,
            workspace_id=settings.BAILIAN_WORKSPACE_ID,
            request_params=self._build_request_params(),
            multimodal_callback=self._callback,
            api_key=settings.DASHSCOPE_API_KEY,
            model=getattr(settings, "BAILIAN_MODEL", "multimodal-dialog"),
        )
        with self._io_lock:
            self._dialog.start(dialog_id=None)

    async def pump_events(self):
        """从百炼回调队列读取并 yield 给 WebSocket。"""
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            etype = event.get("type")
            if etype == "user_text":
                self._append_transcript("候选人", event["text"])
                if event.get("finished"):
                    self._schedule_nudge_if_stuck()
            elif etype == "ai_text":
                self._append_transcript("面试官", event["text"])
            elif etype == "started":
                self.dialog_id = event.get("dialog_id")
                self._started = True
                self._audio_streaming = True
                self._start_heartbeat()
                if not self._opening_sent:
                    self._schedule_opening_greeting()
            elif etype == "speech_started":
                self._user_turn_open = True
            elif etype == "speech_ended":
                self._user_turn_open = False
                self._schedule_nudge_if_stuck()
            elif etype == "responding_started":
                self._ai_responding = True
                if self._nudge_timer:
                    self._nudge_timer.cancel()
                    self._nudge_timer = None
            elif etype == "responding_ended":
                self._ai_responding = False
            elif etype == "state":
                self._current_state = event.get("state") or ""
                self._listening = self._current_state == "Listening"
                logger.info("百炼状态 → %s", self._current_state)
                if (
                    self._enable_video
                    and self._started
                    and self.dialog_id
                    and self._listening
                    and not self._video_connected
                ):
                    if self.connect_video_channel():
                        yield {"type": "video_ready", "msg": "视频通道已连接，开始上传画面"}

            yield event

            if etype in ("closed", "stopped"):
                break

    def _safe_request_to_respond(
        self,
        request_type: str,
        text: str,
        parameters: Optional[RequestToRespondParameters] = None,
        *,
        min_interval: Optional[float] = None,
    ) -> bool:
        """限速调用 request_to_respond，避免 TooManyInterrupt。"""
        if not self._dialog or self._closed:
            return False
        interval = self._video_interval if min_interval is None else min_interval
        with self._io_lock:
            if not self._dialog or self._closed:
                return False
            with self._respond_lock:
                now = time.monotonic()
                elapsed = now - self._last_respond_at
                if elapsed < interval:
                    return False
                self._dialog.request_to_respond(request_type, text, parameters)
                self._last_respond_at = time.monotonic()
                return True

    def _schedule_opening_greeting(self):
        """会话建立后触发 AI 开场白（仅一次）。"""

        def _go():
            if self._closed or self._opening_sent or not self._dialog:
                return
            try:
                ok = self._safe_request_to_respond(
                    "prompt",
                    "请用口语简短问候候选人，说明这是一次语音面试，并邀请其用约一分钟做自我介绍。"
                    + ("可先点出你从简历里注意到的一个项目，再请他展开。" if self.resume_text else ""),
                    min_interval=0.0,
                )
                if ok:
                    self._opening_sent = True
                    logger.info("Opening greeting requested")
            except Exception as e:
                logger.warning("opening greeting failed: %s", e)

        threading.Timer(0.8, _go).start()

    def connect_video_channel(self) -> bool:
        if not self._dialog or self._video_connected or not self.dialog_id:
            return False
        try:
            cmd = [{"action": "connect", "type": "voicechat_video_channel"}]
            params = RequestToRespondParameters(biz_params=BizParams(videos=cmd))
            ok = self._safe_request_to_respond("prompt", "", params, min_interval=0.0)
            if ok:
                self._video_connected = True
                logger.info("Live video channel connect sent")
            return ok
        except Exception as e:
            logger.warning("connect video failed: %s", e)
            return False

    def set_client_input_mode(self, mode: str) -> str:
        """切换客户端输入模式（不断开百炼会话）。返回实际生效的模式。"""
        m = (mode or "").strip().lower()
        if m in ("realtime", "duplex", "continuous"):
            self._client_input_mode = "realtime"
        elif m in ("ptt", "push2talk"):
            self._client_input_mode = "ptt"
        if self._ptt_open:
            try:
                self.cancel_speech()
            except Exception:
                self._ptt_open = False
        logger.info("客户端输入模式 → %s (upstream=%s)", self._client_input_mode, self._upstream_mode)
        return self._client_input_mode

    def get_client_input_mode(self) -> str:
        return self._client_input_mode

    def send_audio_pcm(self, pcm_bytes: bytes):
        if not pcm_bytes or not self._audio_streaming or self._closed:
            return
        if self._client_input_mode == "ptt" and not self._ptt_open:
            return
        with self._io_lock:
            if not self._dialog or self._closed:
                return
            self._dialog.send_audio_data(pcm_bytes)

    def start_speech(self):
        with self._io_lock:
            if self._closed:
                return
            self.notify_playback_ended()
            self._ptt_open = True
            if self._dialog:
                self._dialog.start_speech()

    def stop_speech(self):
        with self._io_lock:
            if self._closed:
                return
            self.notify_playback_ended()
            if self._dialog:
                self._dialog.stop_speech()
            self._ptt_open = False

    def cancel_speech(self):
        """取消本轮语音输入（如无有效声音时）。"""
        if self._nudge_timer:
            self._nudge_timer.cancel()
            self._nudge_timer = None
        with self._io_lock:
            if self._closed:
                self._ptt_open = False
                return
            if not self._dialog:
                self._ptt_open = False
                return
            try:
                cmd = self._dialog.request.generate_common_direction_request(
                    "CancelSpeech",
                    self._dialog.dialog_id,
                )
                self._dialog._send_text_frame(cmd)
            except Exception as e:
                logger.debug("cancel_speech fallback stop: %s", e)
                try:
                    self._dialog.stop_speech()
                except Exception:
                    pass
            self._ptt_open = False

    def _schedule_nudge_if_stuck(self):
        """用户说完后若百炼未自动进入 Responding，主动 prompt 触发 AI 回复。"""
        if self._nudge_timer:
            self._nudge_timer.cancel()

        def _go():
            self._nudge_timer = None
            if self._closed or not self._dialog or self._ai_responding:
                return
            last_user = ""
            for item in reversed(self.transcript):
                if item.get("role") == "候选人":
                    last_user = (item.get("text") or "").strip()
                    break
            if not last_user:
                return
            try:
                ok = self._safe_request_to_respond(
                    "prompt",
                    f"候选人刚才回答：{last_user}。"
                    "请先口语化承接一句，再按面试流程追问下一个问题；"
                    "若上一问有未答子要点，必须先补问该点。",
                    min_interval=0.0,
                )
                logger.info("用户轮次 nudge sent=%s state=%s", ok, self._current_state)
                if ok:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._queue.put({
                                "type": "thinking",
                                "msg": "AI 正在组织回复…",
                            }),
                            self._loop,
                        )
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.warning("nudge failed: %s", e)

        self._nudge_timer = threading.Timer(2.0, _go)
        self._nudge_timer.daemon = True
        self._nudge_timer.start()

    def send_video_frame(self, jpeg_b64: str):
        if not self._dialog or not self._video_connected or not jpeg_b64:
            return
        self._pending_video_b64 = jpeg_b64
        try:
            params = RequestToRespondParameters(
                biz_params=BizParams(
                    videos=[{"type": "base64", "value": self._pending_video_b64}],
                ),
            )
            if self._safe_request_to_respond("prompt", "", params):
                self._pending_video_b64 = None
        except Exception as e:
            logger.debug("video frame send: %s", e)

    def notify_playback_started(self):
        with self._io_lock:
            if self._dialog and not self._closed:
                try:
                    self._dialog.local_responding_started()
                except Exception:
                    pass

    def notify_playback_ended(self):
        with self._io_lock:
            if self._dialog and not self._closed:
                try:
                    self._dialog.local_responding_ended()
                except Exception:
                    pass

    def stop(self):
        self._closed = True
        if self._nudge_timer:
            self._nudge_timer.cancel()
            self._nudge_timer = None
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        with self._io_lock:
            if self._dialog:
                try:
                    self._dialog.stop()
                except Exception:
                    pass
                try:
                    self._dialog.close()
                except Exception:
                    pass
        try:
            self._io_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def get_transcript_plain(self) -> str:
        lines = []
        for item in self.transcript:
            lines.append(f"[{item.get('time', '')}] {item['role']}：{item['text']}")
        return "\n".join(lines)


# 全局会话存储（面试结束后用于生成/下载 MD）
live_sessions: Dict[str, BailianInterviewSession] = {}
live_reports: Dict[str, str] = {}
