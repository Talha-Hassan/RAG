from contextlib import asynccontextmanager
import time
import os 



from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import ( ChatRequest, ChatResponse, ErrorResponse)
from app.security import analyze
from app.cache import ResponseCache
from app.monitor import get_logger, MetricsCollector 
from app.agent import ProductionAgent


load_dotenv()  # Load environment variables from .env file
logger = get_logger()



@asynccontextmanager
async def lifespan(app: FastAPI):


    global logger, metrics_collector, cache, agent
    settings = get_settings()

    logger.info("Starting up the application...",extra={ "extra_data" :{
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "fallback_model": settings.fallback_model,
    }})

    cache = ResponseCache()
    metrics_collector = MetricsCollector()
    agent = ProductionAgent()

    yield # Control is returned to FastAPI to start serving requests

    logger.info("Shutting down the application...",extra={ "extra_data" :{
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "fallback_model": settings.fallback_model,
    }})


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Therapy Chat API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter

@app.post("/chat" , response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint", tags=["chat"])
async def chat(request: Request, chat_request: ChatRequest):
    start_time = time.time()
    try:
        # Analyze the message for potential injection
        analysis_result = analyze(chat_request.message)
        if analysis_result.is_suspicious:
            logger.warning("Injection detected in user message.", extra={"extra_data": {"message": chat_request.message}})
            raise HTTPException(status_code=400, detail="Potential injection detected in the message.")

        # Check cache for existing response
        cached_response = await cache.get(chat_request.message)
        if cached_response:
            metrics_collector.record_request(latency=time.time() - start_time, tokens_in=len(chat_request.message.split()), tokens_out=len(cached_response.split()), cache_hit=True)
            return ChatResponse(response=cached_response)

        # Generate response using the agent
        # print(f"Processing message: {chat_request.message}")
        response_text = await agent.invoke(chat_request.message)

        # Store the response in cache
        # await cache.set(chat_request.message, response_text)
        try : 
            metrics_collector.record_request(latency=time.time() - start_time, tokens_in=len(chat_request.message.split()), tokens_out=len(response_text.split()), cache_hit=False)
        except Exception as e:
            logger.error("Error recording metrics.", extra={"extra_data": {"error": str(e)}})

        # print(f"Response Text = {response_text['response']}")
        return ChatResponse(response=response_text["response"], thread_id=None, model_used=response_text["model_used"], cached=False, processing_time=time.time() - start_time)

    except RateLimitExceeded as e:
        logger.error("Rate limit exceeded.", extra={"extra_data": {"message": str(e)}})
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    except Exception as e:
        logger.error("An error occurred while processing the chat request.", extra={"extra_data": {"error": str(e)}})
        raise HTTPException(status_code=500, detail="Internal server error.")
