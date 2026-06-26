# 工单编号：EDU_AI_INTERVIEW_20260407
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# 面试记录
class InterviewRecord(BaseModel):
    """面试记录数据模型"""
    id: Optional[int] = None
    student_name: str = Field(..., description="学生姓名")
    job_name: str = Field(..., description="岗位名称")
    round_type: str = Field(..., description="面试轮次/形式")
    city: str = Field(..., description="面试城市")
    interview_time: str = Field(..., description="面试时间")
    reporter: str = Field(..., description="上报人")
    report_time: Optional[str] = Field(None, description="上报时间")
    status: str = Field("待完善", description="状态")
    ai_review: Optional[dict] = Field(None, description="AI复盘结果")

# 对话项
class DialogueItem(BaseModel):
    """面试对话项：原始回答+优化回答"""
    speaker: str = Field(..., description="说话人（面试官/学生）")
    content: str = Field(..., description="原始回答")
    optimized: str = Field(..., description="AI优化回答")

# 问题解析
class QuestionAnalysis(BaseModel):
    """面试问题分析模型"""
    question: str = Field(..., description="面试问题")
    score: int = Field(..., description="问题得分（0-10分）")
    strength: str = Field(..., description="回答亮点")
    weakness: str = Field(..., description="回答不足")
    suggestion: str = Field(..., description="改进建议")

# AI复盘返回
class InterviewReviewResponse(BaseModel):
    """面试AI复盘完整响应"""
    total_score: int = Field(..., description="面试总分（0-10分）")
    overall_evaluation: str = Field(..., description="整体评价")
    self_introduction_advice: str = Field(..., description="自我介绍优化建议")
    question_analysis: List[QuestionAnalysis] = Field(..., description="问题分析列表")
    dialogue_history: List[DialogueItem] = Field(..., description="面试对话历史")