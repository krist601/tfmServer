from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the summarization model
summarizer = pipeline('summarization', model='adrimj/text_summ_bart')

# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

# Define a Pydantic model for the response body
class SummaryResponse(BaseModel):
    summary: str

# Define the POST endpoint
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: TextRequest):
    summary = summarizer(request.text, max_length=1500, min_length=40, do_sample=False)
    summarized_text = summary[0]['summary_text']
    return SummaryResponse(summary=summarized_text)