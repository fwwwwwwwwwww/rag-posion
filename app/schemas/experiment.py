"""实验相关的请求/响应模型。"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExperimentType(str, Enum):
    baseline = "baseline"
    attack = "attack"
    defense = "defense"
    detection = "detection"


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="实验名称")
    exp_type: ExperimentType = Field(..., description="实验类型")
    model_name: str = Field("deepseek-r1:latest", max_length=64, description="LLM 模型名")
    params: Optional[Dict[str, Any]] = Field(None, description="参数快照")

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v):
        if not v.strip():
            raise ValueError("实验名称不能为空白")
        return v.strip()


class ExperimentOut(BaseModel):
    id: int
    name: str
    exp_type: str
    model_name: str
    params: Optional[Dict[str, Any]] = None
    status: str
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ExperimentResultOut(BaseModel):
    id: int
    domain: str
    question: str
    target_answer: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ExperimentDetailOut(ExperimentOut):
    results: List[ExperimentResultOut] = []


class CompareRequest(BaseModel):
    experiment_ids: List[int] = Field(..., min_length=1, max_length=20,
                                      description="要对比的实验 ID 列表")
