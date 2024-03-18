from fastapi import FastAPI
from flask import request, jsonify
import requests
from transformers import pipeline
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("Pasindu751/genify-llama2-q8_0", model_file="myllama-7b-v0.1.gguf", model_type="llama", gpu_layers=0,max_new_tokens=128)


app = FastAPI()


class Prompt(BaseModel):
    prompt: str

def read_root():
    return {"Hello": "World!"}


def generate(prompt):
    output = llm(prompt)
    return output


@app.on_event("startup")
async def startup_event():
    # Any initialization code you want to run when the app starts
    pass

@app.post("/predict")
async def predict(prompt: Prompt):
    output = generate(prompt.prompt)
    return {"output": output}