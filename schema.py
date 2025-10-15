from pydantic import BaseModel, Field
from typing import List

class Adjustment(BaseModel):
    reason: str
    delta: str  # "+10%", "-10%" 등

class BaseTable(BaseModel):
    desc: str

class Reasons(BaseModel):
    base_table: BaseTable
    adjustments: List[Adjustment]

class Law(BaseModel):
    name: str
    key: str

class Precedent(BaseModel):
    source: str
    gist: str

class InputMeta(BaseModel):
    video_name: str
    video_date: str

class Summary(BaseModel):
    A: int
    B: int

class AccidentPayload(BaseModel):
    summary: Summary
    reasons: Reasons
    laws: List[Law] = Field(default_factory=list)
    precedents: List[Precedent] = Field(default_factory=list)
    input: InputMeta