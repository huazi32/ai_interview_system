import dashscope
from dashscope import Generation

dashscope.api_key = "sk-1cf6af7eb2ba48288687d78e12969c0b"

def generate_ai_review(user_answer, question=None):
    """
    【音频分析专用】
    user_answer: 面试者的语音转文字内容
    """
    prompt = f"""
你是专业的面试评估官，请严格按照以下结构，根据面试者的回答生成完整的复盘报告，所有内容必须用中文输出，不要省略任何模块：

【面试回答原文】
{user_answer}

请严格按照以下5个模块输出，每个模块用对应标题开头：
1. 【综合评分】：满分100分，给出具体分数，同时说明评分理由
2. 【优点】：分点说明面试回答的优点
3. 【缺点】：分点说明面试回答存在的问题
4. 【优化建议】：分点给出具体、可落地的优化方向
5. 【高分参考答案】：针对这个面试问题，给出一个完整、专业的高分回答示例
"""
    resp = Generation.call(model="qwen-turbo", messages=[{"role":"user","content":prompt}])
    return resp.output.text


def generate_multimodal_ai_review(text, face_pose_dict):
    """
    【仅内部逻辑优化 外部接口100%不动】彻底解决半读半回、套话问题
    函数名、入参、返回格式完全兼容原有代码，不影响任何其他功能
    """
    expression = face_pose_dict.get("表情状态", "自然专注")
    pose = face_pose_dict.get("肢体动作", "站姿端正")

    prompt = f"""
你是一位有10年CV算法团队招聘经验的资深面试官，现在你刚看完面试者的完整面试视频，要给一份**精准、有针对性、完全贴合面试者内容**的面试复盘点评。

【最高优先级铁律 违反直接淘汰】
1.  **绝对禁止半读半回**：必须精准引用面试者语音里**已经说出来的具体技术、项目、模型、场景**，比如面试者提到了FunASR、MediaPipe、YOLOv8，你必须精准提到这些内容，绝对不能泛泛问“你有没有用语音识别模型”这种废话。
2.  **绝对禁止通用套话**：不能说“内容深度不足、缺乏细节”这种空话，必须精准指出“你提到了AI复盘系统，但没说清你在FunASR语音识别模块里做了什么优化”这种具体问题。
3.  **评分必须逻辑自洽**：综合评分0-100分，必须同时结合语音内容的专业度和视觉表现，视觉表现全优的情况下，不能给过低的分数，评分理由必须精准对应内容。
4.  **必须结合视觉细节**：要把视觉识别结果里的具体占比数据自然融入点评，不能生硬堆砌。
5.  **高分参考回答必须100%贴合面试者的真实内容**：必须基于面试者已经提到的项目、技术、模型优化，绝对不能瞎编面试者没提到的技术、模型。

【面试者的完整语音内容（必须逐字认真读，精准引用）】
{text}

【面试者的视频画面表现（必须结合细节）】
- 表情状态：{expression}
- 肢体动作：{pose}

【点评要求】
1.  先给【综合评分】，满分100分，给出具体分数，评分理由必须同时结合语音里的具体技术内容和视觉表现，不能说套话。
2.  然后分3个部分，用自然的段落表达，不要生硬的序号：
    - 【做得好的地方】：必须精准提到面试者语音里的具体项目、技术点，比如“你提到的用FunASR做语音识别、MediaPipe做多模态视觉分析的AI面试复盘系统，方向非常贴合岗位需求”，绝对不能泛泛而谈。
    - 【存在的问题】：必须精准指出语音里的具体问题，比如“你介绍AI复盘系统时，只说了整体功能，没讲清你在语音识别模块里做了什么核心优化、解决了什么难点”，还要结合视觉表现指出问题，不能说空话。
    - 【优化建议】：必须给可落地的、针对这个面试者的具体建议，比如“你可以补充AI复盘系统里，文本矫正模块的Prompt优化细节，还有语音识别准确率的提升数据”，不能给通用模板。
3.  最后给【高分参考回答】：必须完全基于面试者已经提到的项目、技术、场景优化，100%贴合他的真实内容，绝对不能瞎编他没提到的技术、模型。
4.  语言要真诚、专业，像真实面试官的一对一沟通，不要用生硬的书面语、套话。
"""
    try:
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.85,
            timeout=60
        )
        return resp.output.text.strip()
    except Exception as e:
        return f"AI点评生成失败：{str(e)}"

# ====================== 新增功能：三性六讲评分 + 练习题生成 + 答案评价 ======================
import json

# 1. 三性六讲 100分评分标准
def score_three_six_dimensions(user_answer):
    prompt = f"""
你是专业面试考官，严格按照【三性六讲】标准评分，总分100分，仅输出标准JSON，无多余文字。
【评分规则】
三性(50分)：专业性20分 | 逻辑性15分 | 完整性15分
六讲(50分)：讲痛点8分 | 讲方案10分 | 讲落地10分 | 讲数据8分 | 讲价值8分 | 讲反思6分

【面试回答内容】
{user_answer}

输出格式：
{{
    "总分": 分数,
    "三性评分": {{"专业性":分数,"逻辑性":分数,"完整性":分数}},
    "六讲评分": {{"讲痛点":分数,"讲方案":分数,"讲落地":分数,"讲数据":分数,"讲价值":分数,"讲反思":分数}},
    "评价": "简短评价"
}}
"""
    try:
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return json.loads(resp.output.text.strip())
    except:
        return {
            "总分": 0,
            "三性评分": {"专业性":0,"逻辑性":0,"完整性":0},
            "六讲评分": {"讲痛点":0,"讲方案":0,"讲落地":0,"讲数据":0,"讲价值":0,"讲反思":0},
            "评价": "评分失败"
        }

# 2. 根据面试内容生成3道相关性练习题
def generate_three_practice_questions(user_answer):
    prompt = f"""
根据面试者的回答内容，生成3道**强相关**的面试强化练习题，仅输出JSON格式。
【面试内容】
{user_answer}

输出格式：
{{"questions": ["题目1", "题目2", "题目3"]}}
"""
    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.output.text.strip())

# 3. 评价练习题答案
def evaluate_practice_answer(question, user_answer):
    prompt = f"""
作为面试官，对以下答题进行评价：优点、存在问题、改进建议，简洁专业。
题目：{question}
用户回答：{user_answer}
"""
    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.output.text.strip()