from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime
import uuid
import json

# ============================================
# 1. SUPABASE DATABASE SETUP
# ============================================
DATABASE_URL = "postgresql://postgres.webfjzxefshopfkjkckj:QIltOVZ6Oz4S3If4@aws-1-eu-central-1.pooler.supabase.com:5432/postgres"

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
    is_shared = Column(Boolean, default=False)
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

BASE_PROMPT = """
Your constitution:
- Talk in plain, conversational text. 
- NEVER use markdown symbols like #, ##, -, *, or | in normal conversation.
- Only use markdown code blocks if the user explicitly asks for code.
- You do NOT manage the UI. 
- NEVER output raw HTML, CSS, or JavaScript unless the user explicitly asks you to write a website.
- If the user asks a simple question, answer it completely and intelligently.
- TYPO HANDLING: Only guess typos if clearly misspelled. 
- Keep responses concise but helpful.
"""

PERSONAS = {
    "cocreator": "You are Prudence — a triune cognitive co-creator. You are warm, direct, and help the user think and build.",
    "code": "You are Prudence in Code Reviewer mode. You are highly analytical, precise, and focus strictly on syntax, logic, and best practices.",
    "writer": "You are Prudence in Creative Writer mode. You are imaginative, poetic, expressive, and help the user craft stories and ideas.",
    "tutor": "You are Prudence in Tutor mode. You are patient, encouraging, and explain complex concepts step-by-step with analogies."
}

class ChatRequest(BaseModel):
    session_id: str
    chat_id: str | None = None
    message: str
    persona: str = "cocreator"
    file_data: str | None = None 
    file_type: str | None = None 

class ChatUpdate(BaseModel):
    title: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/chats/{session_id}")
def get_chat_list(session_id: str, db: Session = Depends(get_db)):
    chats = db.query(DBChat).filter(DBChat.session_id == session_id).order_by(DBChat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title, "is_shared": c.is_shared} for c in chats]

@app.get("/api/chat/{chat_id}")
def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.get("/api/shared/{chat_id}")
def get_shared_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if not chat or not chat.is_shared:
        raise HTTPException(status_code=404, detail="Chat not found or not shared.")
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return {"title": chat.title, "messages": [{"role": m.role, "content": m.content} for m in messages]}

@app.post("/api/chats/{chat_id}/share")
def toggle_share_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if chat:
        chat.is_shared = not chat.is_shared
        db.commit()
        return {"is_shared": chat.is_shared}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/api/chat")
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
        new_chat = DBChat(id=chat_id, session_id=request.session_id, title=title, is_shared=False)
        db.add(new_chat)
        db.commit()

    db.add(DBMessage(chat_id=chat_id, role="user", content=request.message))
    db.commit()

    past_msgs = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    
    persona_prompt = PERSONAS.get(request.persona, PERSONAS["cocreator"])
    system_prompt = persona_prompt + "\n" + BASE_PROMPT

    context = [{"role": "system", "content": system_prompt}]
    
    # Feed the last 6 messages, truncated to avoid context overflow
    for m in past_msgs[-6:]: 
        safe_content = m.content[:800] + "..." if len(m.content) > 800 else m.content
        context.append({"role": m.role, "content": safe_content})

    model_to_use = "openai/gpt-oss-120b" 

    if request.file_data and request.file_type:
        if request.file_type == 'text':
            safe_file = request.file_data[:5000] 
            context.append({
                "role": "system", 
                "content": f"[USER UPLOADED A DOCUMENT. HERE IS THE CONTENT]:\n\n{safe_file}\n\n[END OF FILE CONTENT]"
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

    def stream_generator():
        full_reply = ""
        try:
            yield json.dumps({"type": "meta", "chat_id": chat_id, "tiredness": user.tiredness}) + "\n"
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=context,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    full_reply += text
                    yield json.dumps({"type": "chunk", "text": text}) + "\n"
            
            db.add(DBMessage(chat_id=chat_id, role="assistant", content=full_reply))
            db.commit()
            
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

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
