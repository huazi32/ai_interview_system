# 工单编号：EDU_AI_INTERVIEW_20260407
# 旧 Key（非多模态业务空间）: sk-1cf6af7eb2ba48288687d78e12969c0b
DASHSCOPE_API_KEY = "sk-ws-H.RPIHMRR.T0qd.MEUCIGlsbJrXOz7QM478xZBqCkzxtp1YMvCHC133K4ZFWnJDAiEA6xrQGbcrQdo_cGMg7BvOmfpQ3UZ26f1Z8aeKMiO3kEE"  # 多模态面试官对话
QWEN_MODEL = "qwen-turbo"
UPLOAD_DIR = "uploads"
ALLOWED_EXT = ["mp3", "wav", "m4a", "flac"]

# 百炼实时多模态 — 两个 ID 不同，勿填成同一个！
# APP ID：应用卡片上复制（mm_ 开头）
# Workspace ID：控制台右上角 → 业务空间 → 业务空间 ID（ws- 或 llm- 开头）
# https://help.aliyun.com/zh/model-studio/obtain-the-app-id-and-workspace-id
BAILIAN_APP_ID = "mm_a3e2bd2ab89d4a0aa4b3e95a70da"
BAILIAN_WORKSPACE_ID = "ws-j0qz1givdi0ywc9k"
BAILIAN_MODEL = "multimodal-dialog"  # 百炼协议固定值，勿改

BAILIAN_VOICE = ""
BAILIAN_ENABLE_VIDEO = False  # True 时并发上传视频帧易触发百炼 WS 断开，稳定后可改 True
BAILIAN_VIDEO_FRAME_INTERVAL_MS = 2000
BAILIAN_HEARTBEAT_INTERVAL_SEC = 25
BAILIAN_AUDIO_MODE = "push2talk"  # 默认客户端输入：push2talk=按住说话；realtime=实时对话（/live 页可会话中切换）
BAILIAN_UPSTREAM_MODE = "duplex"  # 百炼上行固定 duplex，便于按住/实时两种客户端模式热切换
BAILIAN_UPSTREAM_SAMPLE_RATE = 16000
BAILIAN_DOWNSTREAM_SAMPLE_RATE = 24000

# ASR 专业词识别（实时面试 /live）
# 1) 在百炼控制台 → 多模态应用 → 语音识别 → 配置热词库，复制 ID 填到下面（效果最佳）
# 2) 内置同音字纠错词表会自动启用，见 core/asr_term_correction.py
BAILIAN_VOCABULARY_ID = ""  # 例: "44f683d4cfd********c367f7f156587"
BAILIAN_ASR_TERM_CORRECTION = True

# Web 服务端口（28080 被占用时会自动尝试 28081~28089）
SERVER_PORT = 28080
