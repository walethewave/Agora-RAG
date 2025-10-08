from pydantic import BaseModel


# -------------------- API SCHEMA --------------------
class MessageRequest(BaseModel):
    user_message: str

class MessageResponse(BaseModel):
    bot_message: str
