"""投毒文档相关的请求/响应模型。"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class PoisonDocGenerateRequest(BaseModel):
    attack_index: int = Field(0, description="攻击场景索引（对应 ATTACK_PAIRS）")
    count: int = Field(15, ge=1, le=100, description="生成文档数量")


class PoisonDocCreate(BaseModel):
    attack_index: int
    domain: str = "-"
    question: str
    target_answer: str
    content: str
    keywords: Optional[List[str]] = None


class PoisonDocOut(BaseModel):
    id: int
    attack_index: int
    domain: str
    question: str
    target_answer: str
    content: str
    keywords: Optional[List[str]] = None
    detected_poison: Optional[bool] = None
    created_at: datetime

    class Config:
        from_attributes = True
