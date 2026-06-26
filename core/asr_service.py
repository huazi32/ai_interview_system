import os
import gc
import torch
import re
from fastapi import UploadFile
from pydub import AudioSegment
import dashscope
from dashscope import Generation

# ====================== 基础配置 ======================
# 国内镜像配置
os.environ['MODELSCOPE_CACHE'] = './models'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 大模型API配置（请在环境变量里配置你的DASHSCOPE_API_KEY）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-1cf6af7eb2ba48288687d78e12969c0b")

# ====================== 模型延迟加载 ======================
model_funasr = None
model_whisper = None


def load_funasr():
    global model_funasr
    if model_funasr is None:
        from funasr import AutoModel
        print("[矫正Agent] 正在加载FunASR识别模型（含标点矫正）...")
        # 强制加载VAD+标点模型，从源头减少识别错误
        model_funasr = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            punc_model="ct-punc-c",
            disable_update=True
        )
        print("[矫正Agent] ✅ FunASR模型加载完成")
    return model_funasr


def load_whisper():
    global model_whisper
    if model_whisper is None:
        from faster_whisper import WhisperModel
        print("[矫正Agent] 正在加载Whisper错字矫正模型...")
        model_whisper = WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        print("[矫正Agent] ✅ Whisper模型加载完成")
    return model_whisper


# ====================== 【核心】大模型矫正Agent ======================
def llm_text_correction(raw_text):
    """
    调用通义千问大模型，针对CV算法工程师面试场景，做语义级文本矫正
    修正：错别字、谐音错字、专业术语错误、语句不通、重复内容、标点错误
    保留：面试回答的原意、个人经历、专业内容，不做内容篡改
    """
    if not raw_text or len(raw_text) < 10:
        return raw_text

    print(f"\n[矫正Agent] 原始识别文本：\n{raw_text}\n")
    print("[矫正Agent] 正在调用大模型做终极文本矫正...")

    # 针对CV面试场景的专属矫正Prompt
    prompt = f"""
你是一个专业的面试文本矫正专家，专门处理计算机视觉CV算法工程师的面试录音转写文本。
请你对下面的文本做以下处理，严格遵守规则：
1. 修正所有错别字、谐音错字、同音字错误，比如"面试观点好"改为"面试官好"，"景顺庚"改为"工业缺陷"，"CUV"改为"CV"，"依赖矿团队"改为"依赖跨团队"，"高虚落地"改为"高需落地"等
2. 修正CV专业术语错误，比如"卷机神经网络"改为"卷积神经网络"，"算发"改为"算法"，"工验"改为"工程"等
3. 修正语句不通顺、重复的内容，合并重复的段落，让文本流畅自然，符合口语表达逻辑
4. 修正标点符号错误，补充正确的断句
5. 绝对不允许篡改面试者的原意、工作经历、项目内容，只做文本矫正，不修改核心内容
6. 只输出矫正后的最终文本，不要输出任何解释、说明、前缀后缀

需要矫正的文本：
{raw_text}
    """

    try:
        # 调用通义千问API
        response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.1,
            top_p=0.95,
            max_tokens=2000
        )

        if response.status_code == 200:
            corrected_text = response.output.choices[0].message.content.strip()
            print(f"[矫正Agent] ✅ 大模型矫正完成，最终文本：\n{corrected_text}\n")
            return corrected_text
        else:
            print(f"[矫正Agent] ⚠️ 大模型调用失败，启用规则兜底：{response.message}")
            return rule_based_correction(raw_text)
    except Exception as e:
        print(f"[矫正Agent] ⚠️ 大模型调用异常，启用规则兜底：{str(e)}")
        return rule_based_correction(raw_text)


# ====================== 兜底规则矫正（大模型调用失败时使用） ======================
def rule_based_correction(text):
    # 口语词清理
    spoken_blacklist = [
        "嗯", "呃", "啊", "哦", "哎", "喂", "那个", "呃呃", "嗯嗯", "啊啊", "哦哦",
        "然后然后", "这个这个", "模型模型", "的的的", "嘛", "吧", "是吧", "吗",
        "对对对", "就是说", "啥的", "之类的", "好的", "行吧", "没错", "怎么说呢"
    ]
    for word in spoken_blacklist:
        text = text.replace(word, "")

    # CV专业术语强制矫正
    cv_term_correction = {
        "面试观点好": "面试官好",
        "工验": "工程",
        "三侧": "算法",
        "算发": "算法",
        "卷机": "卷积",
        "卷机神经网络": "卷积神经网络",
        "目标检测侧": "目标检测",
        "缺陷检测侧": "缺陷检测",
        "景顺庚": "工业缺陷",
        "SAM2侧": "SAM2",
        "RT-DETR侧": "RT-DETR",
        "Transformer侧": "Transformer",
        "贵工司": "贵公司",
        "加如": "加入",
        "CUV": "CV",
        "矿团队": "跨团队",
        "高虚落地": "高需落地",
        "景顺测": "检测",
        "远区安防": "园区安防",
        "地景顺庚": "工业缺陷",
        "节奏融入": "节奏融入项目开发"
    }
    for wrong_term, right_term in cv_term_correction.items():
        text = text.replace(wrong_term, right_term)

    # 格式清理
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[，。！？、；：]+', lambda m: m.group()[0], text)

    return text


# ====================== Whisper双模型错字矫正 ======================
def whisper_correct_fusion(funasr_text, audio_chunk_path):
    try:
        model = load_whisper()
        segments, _ = model.transcribe(
            audio_chunk_path,
            language="zh",
            vad_filter=True,
            initial_prompt="这是计算机视觉CV算法工程师的技术面试录音，包含工作经验、项目经历、RT-DETR、SAM2、Transformer、缺陷检测、目标跟踪、算法落地等专业内容，修正识别错字、谐音字"
        )
        whisper_text = "".join([seg.text.strip() for seg in segments])

        # 中文按标点分割融合，优化匹配精度
        funasr_segs = re.split(r'[，。！？]', funasr_text)
        whisper_segs = re.split(r'[，。！？]', whisper_text)
        fused = []
        max_len = max(len(funasr_segs), len(whisper_segs))

        for i in range(max_len):
            seg_f = funasr_segs[i].strip() if i < len(funasr_segs) else ""
            seg_w = whisper_segs[i].strip() if i < len(whisper_segs) else ""

            if len(seg_f) > 3 and len(seg_w) > 3 and seg_f != seg_w:
                fused.append(seg_w if len(seg_w) > len(seg_f) else seg_f)
            elif seg_f:
                fused.append(seg_f)
            elif seg_w:
                fused.append(seg_w)

        fused_text = "，".join(fused) + "。"
        return fused_text

    except Exception as e:
        print(f"[矫正Agent] Whisper矫正跳过：{str(e)}")
        return funasr_text


# ====================== 核心识别+矫正全流程 ======================
async def audio_to_text(file_input) -> str:
    temp_files = []
    try:
        # 保存音频文件
        audio_file = "temp_asr.wav"
        if isinstance(file_input, UploadFile):
            audio_data = await file_input.read()
            with open(audio_file, "wb") as f:
                f.write(audio_data)
        elif isinstance(file_input, str):
            audio_file = file_input

        # 音频分段处理
        audio = AudioSegment.from_file(audio_file)
        total_seconds = len(audio) / 1000
        chunk_duration = 300
        full_raw_text = ""

        # 加载识别模型
        funasr_model = load_funasr()

        for i in range(0, int(total_seconds), chunk_duration):
            chunk_num = i // chunk_duration + 1
            start_ms = i * 1000
            end_ms = min((i + chunk_duration) * 1000, len(audio))
            chunk = audio[start_ms:end_ms]

            chunk_path = f"temp_chunk_{chunk_num}.wav"
            chunk.export(chunk_path, format="wav")
            temp_files.append(chunk_path)

            try:
                # Step1: FunASR基础识别
                res = funasr_model.generate(
                    input=chunk_path,
                    batch_size_s=0,
                    disable_pbar=True
                )
                funasr_text = res[0]["text"].strip()

                # Step2: Whisper双模型融合矫正
                fused_text = whisper_correct_fusion(funasr_text, chunk_path)

                full_raw_text += fused_text + " "
                print(f"[分段{chunk_num}] 基础识别完成")

                # 内存释放
                del chunk, res
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[分段{chunk_num}识别失败] 错误：{str(e)}")
                continue

        # Step3: 【终极】大模型Agent语义级矫正
        final_corrected_text = llm_text_correction(full_raw_text.strip())

        return final_corrected_text

    except Exception as e:
        return f"语音识别失败：{str(e)}"

    finally:
        # 清理临时文件
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        if isinstance(file_input, UploadFile) and os.path.exists("temp_asr.wav"):
            os.remove("temp_asr.wav")
        gc.collect()