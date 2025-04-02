from pydantic import BaseModel, Field
from typing import Optional

class InputSchema(BaseModel):
    problem: str = Field(..., title="Problem to solve")
    num_thoughts: Optional[int] = Field(3, title="Number of reasoning paths to generate")