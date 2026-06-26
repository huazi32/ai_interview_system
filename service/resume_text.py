"""简历文本提取与截断（供实时面试 Prompt 使用）。"""

RESUME_PROMPT_MAX_CHARS = 3500


def normalize_resume_text(raw: str, max_chars: int = RESUME_PROMPT_MAX_CHARS) -> str:
    text = (raw or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n…（简历已截断）"
    return text
