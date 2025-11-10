from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import requests
import os

load_dotenv()
app = FastAPI()

# SQLite setup
Base = declarative_base()
engine = create_engine("sqlite:///chat_history.db")
Session = sessionmaker(bind=engine)
session = Session()

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    user_input = Column(Text)
    bot_response = Column(Text)
    model_used = Column(Text)

Base.metadata.create_all(engine)

# API keys
PQAI_API_URL = "https://search.projectpq.ai/api/search"
PQAI_API_KEY = os.getenv("PQAI_API_KEY")
GPT4_API_URL = "https://api.openai.com/v1/chat/completions"
GPT4_API_KEY = os.getenv("GPT4_API_KEY")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Dual Chatbot: GPT-4 + PQAI</title>
      <style>
        body { font-family: sans-serif; margin: 2em; }
        .chat-box { max-width: 600px; margin: auto; }
        .message { margin: 1em 0; }
        .user { color: blue; }
        .bot { color: green; }
      </style>
    </head>
    <body>
      <div class="chat-box" id="chatBox"></div>
      <label for="modelSelect">Choose model:</label>
      <select id="modelSelect">
        <option value="gpt4">GPT-4</option>
        <option value="pqai">PQAI</option>
      </select>
      <input type="text" id="userInput" placeholder="Ask something..." autofocus />
      <button onclick="sendMessage()">Send</button>
      <a href="/history">View History</a>

      <script>
        async function sendMessage() {
          const input = document.getElementById("userInput");
          const model = document.getElementById("modelSelect").value;
          const message = input.value;
          if (!message) return;

          const chatBox = document.getElementById("chatBox");
          chatBox.innerHTML += `<div class="message user"><b>You:</b> ${message}</div>`;
          input.value = "";

          const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, model })
          });
          const data = await res.json();
          chatBox.innerHTML += `<div class="message bot"><b>${model.toUpperCase()}:</b> ${data.reply}</div>`;
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")
    model = data.get("model", "gpt4")

    if model == "pqai":
        headers = {
            "Authorization": f"Bearer {PQAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = { "query": user_input, "filters": {} }
        try:
            response = requests.post(PQAI_API_URL, headers=headers, json=payload)
            result = response.json()
            reply = result.get("summary", "No relevant patents found.")
        except Exception as e:
            reply = f"Error contacting PQAI: {str(e)}"

    elif model == "gpt4":
        headers = {
            "Authorization": f"Bearer {GPT4_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": user_input}]
        }
        try:
            response = requests.post(GPT4_API_URL, headers=headers, json=payload)
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"Error contacting GPT-4: {str(e)}"

    else:
        reply = "Invalid model selected."

    chat_entry = Chat(user_input=user_input, bot_response=reply, model_used=model)
    session.add(chat_entry)
    session.commit()

    return JSONResponse(content={"reply": reply})

@app.get("/history", response_class=HTMLResponse)
async def history():
    chats = session.query(Chat).order_by(Chat.id.desc()).limit(20).all()
    html = "<h2>Chat History</h2><ul>"
    for chat in chats:
        html += f"<li><b>Model:</b> {chat.model_used.upper()}<br><b>You:</b> {chat.user_input}<br><b>Bot:</b> {chat.bot_response}</li><hr>"
    html += "</ul><a href='/'>Back to chat</a>"
    return HTMLResponse(content=html)