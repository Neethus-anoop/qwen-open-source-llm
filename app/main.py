from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI(
    title="Qwen Local Inference API",
    description="Local Qwen 2.5 inference using Ollama",
    version="1.0.0"
)


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "qwen2.5:0.5b",
        "engine": "Ollama"
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    response = ollama.generate(
        model="qwen2.5:0.5b",
        prompt=request.prompt
    )

    return {
        "prompt": request.prompt,
        "response": response["response"]
    }