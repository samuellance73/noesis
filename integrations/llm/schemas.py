from typing import List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (e.g., 'user', 'assistant').")
    content: str = Field(..., description="The text content of the message.")

class ChatPayload(BaseModel):       
    model: str = Field(..., description="The ID of the model to use.")
    messages: List[ChatMessage] = Field(..., description="The chat history payload.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate.")
    stream: bool = Field(True, description="Enable Server-Sent Events (SSE) streaming.")
