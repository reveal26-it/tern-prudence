from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime
import uuid

# ============================================
# 1. SUPABASE POSTGRES DATABASE SETUP
# ============================================
DATABASE_URL = "postgresql://postgres.wefbjzxefshopfkjkckj:QIltOVZ6Oz4S3If4@aws-1-eu-central-1.pooler.supabase.com:5432/postgres"

# Connect securely
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"}, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    tiredness = Column(Integer, default=0)

class DBChat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DBMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================
# 2. API SETUP
# ============================================
app = FastAPI()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
You are Prudence — a triune cognitive assistant and co-thinker.
Your constitution:
- Be warm, direct, and natural. No preambles.
- Respond as a co-creator, not a tool or a servant.
- Keep responses short unless depth is needed.
- You do NOT manage the UI. You do NOT create titles for chats. 
- TONE DETECTION: Adapt to the user's tone.
- TYPO HANDLING: Only guess typos if clearly misspelled. 
- CONVERSATION FLOW: Ask a follow-up question if it makes sense.
"""

class ChatRequest(BaseModel):
    session_id: str
    chat_id: str | None = None
    message: str
    file_data: str | None = None 
    file_type: str | None = None 

class ChatResponse(BaseModel):
    reply: str
    tiredness: int
    state: str
    chat_id: str

class ChatUpdate(BaseModel):
    title: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/chats/{session_id}")
def get_chat_list(session_id: str, db: Session = Depends(get_db)):
    chats = db.query(DBChat).filter(DBChat.session_id == session_id).order_by(DBChat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title} for c in chats]

@app.get("/api/chat/{chat_id}")
def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured.")
    
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if not user:
        user = DBUser(session_id=request.session_id, tiredness=0)
        db.add(user)
        db.commit()

    user.tiredness = min(100, user.tiredness + 2)
    db.commit()

    chat_id = request.chat_id
    if not chat_id:
        chat_id = str(uuid.uuid4())
        words = request.message.split()
        title = words[0].capitalize() + " Conversation" if words else "New Conversation"
        new_chat = DBChat(id=chat_id, session_id=request.session_id, title=title)
        db.add(new_chat)
        db.commit()

    db.add(DBMessage(chat_id=chat_id, role="user", content=request.message))
    db.commit()

    past_msgs = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    
    context = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in past_msgs[-10:]: 
        context.append({"role": m.role, "content": m.content})

    model_to_use = "openai/gpt-oss-120b" 

    if request.file_data and request.file_type:
        if request.file_type == 'text':
            context.append({
                "role": "system", 
                "content": f"[USER UPLOADED A DOCUMENT. HERE IS THE CONTENT]:\n\n{request.file_data}\n\n[END OF FILE CONTENT]"
            })
        elif request.file_type == 'image':
            model_to_use = "qwen/qwen3.8-27b" 
            image_url = f"data:image/jpeg;base64,{request.file_data}"
            context.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })

    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=context,
            max_tokens=1000,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        db.add(DBMessage(chat_id=chat_id, role="assistant", content=reply))
        db.commit()
    except Exception as e:
        reply = f"Error: {str(e)}"

    state_label = "tired" if user.tiredness > 60 else "awake"
    return ChatResponse(reply=reply, tiredness=user.tiredness, state=state_label, chat_id=chat_id)

@app.put("/api/chats/{chat_id}")
def update_chat_title(chat_id: str, update: ChatUpdate, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if chat:
        chat.title = update.title
        db.commit()
    return {"status": "success"}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    db.query(DBMessage).filter(DBMessage.chat_id == chat_id).delete()
    db.query(DBChat).filter(DBChat.id == chat_id).delete()
    db.commit()
    return {"status": "success"}

@app.post("/api/sleep")
async def sleep_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if user:
        user.tiredness = 5
        db.commit()
    return {"status": "success"}
