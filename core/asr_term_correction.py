"""面试场景 ASR 同音字/术语纠错：百炼 upstream 词表 + 本地后处理。"""
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from dashscope.multimodal.multimodal_request_params import AsrPostProcessing, ReplaceWord

# (错误识别, 正确写法, match_mode) — partial 用于短语片段
_BASE_REPLACEMENTS: Sequence[Tuple[str, str, str]] = (
    # CV / 深度学习常见同音误识别
    ("捐积", "卷积", "partial"),
    ("卷机", "卷积", "partial"),
    ("神经网落", "神经网络", "partial"),
    ("神经网路", "神经网络", "partial"),
    ("目标检策", "目标检测", "partial"),
    ("目标检则", "目标检测", "partial"),
    ("目标检侧", "目标检测", "partial"),
    ("语义分切", "语义分割", "partial"),
    ("实力分割", "实例分割", "partial"),
    ("实列分割", "实例分割", "partial"),
    ("数据增強", "数据增强", "partial"),
    ("反向播", "反向传播", "partial"),
    ("梯读下降", "梯度下降", "partial"),
    ("提度下降", "梯度下降", "partial"),
    ("过事宜", "过拟合", "partial"),
    ("过以合", "过拟合", "partial"),
    ("政则化", "正则化", "partial"),
    ("正责化", "正则化", "partial"),
    ("注意力机治", "注意力机制", "partial"),
    ("特征融和", "特征融合", "partial"),
    ("特征融活", "特征融合", "partial"),
    ("非极大值抑治", "非极大值抑制", "partial"),
    ("非极大值抑至", "非极大值抑制", "partial"),
    ("预训链", "预训练", "partial"),
    ("预训炼", "预训练", "partial"),
    ("微调模形", "微调模型", "partial"),
    ("微调模性", "微调模型", "partial"),
    ("多模太", "多模态", "partial"),
    ("多模带", "多模态", "partial"),
    ("扩散模形", "扩散模型", "partial"),
    ("骨干网落", "骨干网络", "partial"),
    ("学习绿", "学习率", "partial"),
    ("学习律", "学习率", "partial"),
    ("批大小", "batch size", "partial"),
    ("优楼", "YOLO", "partial"),
    ("优络", "YOLO", "partial"),
    ("瑞思 net", "ResNet", "partial"),
    ("瑞思Net", "ResNet", "partial"),
    ("派托奇", "PyTorch", "partial"),
    ("派 torch", "PyTorch", "partial"),
    ("tensor flow", "TensorFlow", "partial"),
    ("Tensor flow", "TensorFlow", "partial"),
    ("伊波克", "epoch", "partial"),
    ("因爱有恩", "IoU", "partial"),
    ("爱欧U", "IoU", "partial"),
    ("恩 ms", "NMS", "partial"),
    ("恩MS", "NMS", "partial"),
    ("m a p", "mAP", "partial"),
    ("cuda", "CUDA", "partial"),
    ("库达", "CUDA", "partial"),
    ("on nx", "ONNX", "partial"),
    ("tensor rt", "TensorRT", "partial"),
    ("open cv", "OpenCV", "partial"),
    ("batch norm", "BatchNorm", "partial"),
    ("批归一", "BatchNorm", "partial"),
    ("批规范", "BatchNorm", "partial"),
    ("锚狂", "锚框", "partial"),
    ("锚矿", "锚框", "partial"),
    ("迁移学系", "迁移学习", "partial"),
    ("强化学系", "强化学习", "partial"),
    ("大模形", "大模型", "partial"),
    ("大模性", "大模型", "partial"),
    ("推理加素", "推理加速", "partial"),
    ("量化部暑", "量化部署", "partial"),
    ("端侧部暑", "端侧部署", "partial"),
    ("知识蒸留", "知识蒸馏", "partial"),
    ("对比学系", "对比学习", "partial"),
    ("自监督学系", "自监督学习", "partial"),
    ("半监督学系", "半监督学习", "partial"),
)

_EN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+#.\-]{1,}\b")
_ZH_TERM_RE = re.compile(
    r"卷积|神经网络|目标检测|语义分割|实例分割|数据增强|反向传播|梯度下降|"
    r"过拟合|正则化|注意力|Transformer|微调|预训练|多模态|扩散|"
    r"量化|部署|蒸馏|迁移学习|强化学习|特征融合|非极大值抑制"
)


def extract_terms_from_text(*texts: str) -> List[str]:
    """从岗位/简历中提取可能需要保护的英文术语与中文专业词。"""
    seen = set()
    terms: List[str] = []
    for raw in texts:
        if not raw:
            continue
        for m in _EN_TOKEN_RE.findall(raw):
            t = m.strip()
            key = t.lower()
            if key not in seen and len(t) >= 2:
                seen.add(key)
                terms.append(t)
        for m in _ZH_TERM_RE.findall(raw):
            if m not in seen:
                seen.add(m)
                terms.append(m)
    return terms


def _job_title_terms(job_title: str) -> List[str]:
    jt = (job_title or "").strip()
    mapping = {
        "CV": ["OpenCV", "目标检测", "语义分割", "YOLO", "ResNet", "mAP", "IoU", "NMS"],
        "算法": ["PyTorch", "TensorFlow", "神经网络", "梯度下降", "过拟合", "正则化"],
        "多模态": ["CLIP", "ViT", "Transformer", "多模态", "对比学习", "预训练"],
        "深度学习": ["反向传播", "BatchNorm", "Adam", "SGD", "epoch", "CUDA"],
    }
    terms: List[str] = []
    for key, vals in mapping.items():
        if key in jt:
            terms.extend(vals)
    return terms


def build_replace_words(
    job_title: str = "",
    resume_text: str = "",
    extra: Optional[Iterable[str]] = None,
) -> List[ReplaceWord]:
    """构建百炼 Upstream AsrPostProcessing 纠错词表。"""
    words: List[ReplaceWord] = []
    seen = set()
    for src, tgt, mode in _BASE_REPLACEMENTS:
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        words.append(ReplaceWord(source=src, target=tgt, match_mode=mode))

    # 简历/岗位中的英文术语：若被识别成全小写或拆词，尽量拉回标准写法
    dynamic = list(_job_title_terms(job_title))
    dynamic.extend(extract_terms_from_text(resume_text))
    if extra:
        dynamic.extend(extra)
    for term in dynamic:
        if not term or len(term) < 2:
            continue
        lower = term.lower()
        for variant in (lower, lower.replace("-", " "), lower.replace("_", " ")):
            key = (variant, term)
            if key in seen or variant == term:
                continue
            seen.add(key)
            words.append(ReplaceWord(source=variant, target=term, match_mode="partial"))
    return words


def build_asr_post_processing(
    job_title: str = "",
    resume_text: str = "",
) -> Optional[AsrPostProcessing]:
    replace_words = build_replace_words(job_title, resume_text)
    if not replace_words:
        return None
    return AsrPostProcessing(replace_words=replace_words)


def correct_user_asr_text(
    text: str,
    job_title: str = "",
    resume_text: str = "",
) -> str:
    """本地后处理：修正 ASR 同音字（展示与 transcript 使用）。"""
    if not text or not text.strip():
        return text
    out = text
    # 长词优先，避免短词误伤
    rules = sorted(_BASE_REPLACEMENTS, key=lambda x: len(x[0]), reverse=True)
    for src, tgt, mode in rules:
        if mode == "exact":
            if out == src:
                out = tgt
        else:
            out = out.replace(src, tgt)

    # 保护简历/岗位中的英文专名（若 ASR 输出小写变体）
    for term in extract_terms_from_text(resume_text, job_title):
        if len(term) <= 2:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        out = pattern.sub(term, out)
    return out
