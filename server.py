from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
import time
from pyngrok import ngrok
import nest_asyncio
import os
from typing import Optional

from main import MultiSpecializedLanguageModelPipeline

os.environ["NGROK_AUTH_TOKEN"] = "ENTER_YOUR_NGROK_AUTH_TOKEN"

# Initialize nest-asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="LLM Query Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global pipeline instance
pipeline = None

# Request model
class QueryRequest(BaseModel):
    query: str
    max_length: Optional[int] = Field(default=512)
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=0.9)

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    logger.info("Initializing LLM pipeline...")
    pipeline = MultiSpecializedLanguageModelPipeline()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global pipeline
    if pipeline:
        pipeline.clear_resources()
    ngrok.kill()

# Root endpoint to verify API is working
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.post("/process")
async def process_query(request: QueryRequest):
    """Process a query using the LLM pipeline"""
    global pipeline
    start_time = time.time()
    
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        result = pipeline.generate_response(
            query=request.query,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "response": result,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def setup_ngrok():
    """Setup ngrok tunnel"""
    try:
        ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
        if not ngrok_token:
            logger.warning("NGROK_AUTH_TOKEN not found")
            return None
            
        ngrok.set_auth_token(ngrok_token)
        public_url = ngrok.connect(8000)
        logger.info(f"ngrok tunnel established at: {public_url}")
        print(f"\nAPI is accessible at: {public_url}/process")
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {e}")
        raise

if __name__ == "__main__":
    setup_ngrok()
    uvicorn.run(app, host="0.0.0.0", port=8000)