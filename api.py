from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model setup
model_id = "CompVis/stable-diffusion-v1-4"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe.to(device)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")

@app.get("/")
def generate(prompt: str):
    try:
        with autocast(device):
            image = pipe(prompt, guidance_scale=8.5).images[0]

        # Save image to bytes buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Convert image to base64 string
        imgstr = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return Response(content=imgstr, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
