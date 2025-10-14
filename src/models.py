from pydantic import BaseModel

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
