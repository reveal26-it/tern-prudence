    # --- CROSS-CHAT MEMORY RETRIEVAL ---
    # Find keywords in the user's message (ignoring words shorter than 4 letters)
    keywords = [w for w in request.message.split() if len(w) > 4]
    cross_chat_context = ""
    
    if keywords:
        # Search past messages from OTHER chats in this same session
        past_memories = db.query(DBMessage).join(DBChat).filter(
            DBChat.session_id == request.session_id,
            DBMessage.chat_id != chat_id,
            DBMessage.role == "user", # Only search what the user said
            or_(*[DBMessage.content.ilike(f"%{kw}%") for kw in keywords])
        ).order_by(DBMessage.timestamp.desc()).limit(5).all()

        if past_memories:
            cross_chat_context = "\n\n[SYSTEM NOTE - CROSS-CHAT MEMORY]:\n"
            cross_chat_context += "You have access to these things the user said in PREVIOUS separate conversations:\n"
            for mem in past_memories:
                chat_title = db.query(DBChat).filter(DBChat.id == mem.chat_id).first().title
                cross_chat_context += f"- In the chat '{chat_title}', they said: '{mem.content}'\n"
            cross_chat_context += "Use this information naturally if it is relevant to their current question.\n"
