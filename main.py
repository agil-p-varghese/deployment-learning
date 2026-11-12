import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from types import SimpleNamespace

# Import LangChain's Chat Hugging Face wrapper classes
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

app = FastAPI()

# Enable CORS for your HTML chat UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = SimpleNamespace(primary_model="meta-llama/Llama-3.1-8B-Instruct")

# 1. Initialize the base LLM endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id=getattr(settings, "primary_model", "meta-llama/Llama-3.1-8B-Instruct"),
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# 2. Wrap it with ChatHuggingFace so it handles chat messages & conversational providers properly
primary_endpoint = ChatHuggingFace(llm=llm_endpoint)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "LangChain ChatHuggingFace Backend is running!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise HTTPException(
            status_code=500, 
            detail="HUGGINGFACEHUB_API_TOKEN environment variable is not configured."
        )
    
    try:
        # Pass messages using LangChain message objects
        messages = [HumanMessage(content=request.message)]
        response = primary_endpoint.invoke(messages)
        
        return {"reply": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))