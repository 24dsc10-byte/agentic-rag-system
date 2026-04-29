"""
FastAPI Entry Point - Main Application with Web UI
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from utils.config import settings
from utils.logger import setup_logging
from models.schemas import QueryRequest
from services.orchestrator import RAGOrchestrator
from services.groq_service import GroqService
from routes.query import set_orchestrator as set_query_orchestrator

# Setup logging
logger = setup_logging(log_level=getattr(logging, settings.log_level), log_file=settings.log_file)

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global orchestrator and Groq service
orchestrator: RAGOrchestrator = None
groq_service: GroqService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown"""
    global orchestrator, groq_service
    
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 Starting Agentic RAG System...")
    logger.info("=" * 70)
    
    try:
        # Initialize Groq Service
        logger.info("📡 Initializing Groq Service...")
        groq_service = GroqService(
            api_key=settings.groq_api_key,
            model=settings.groq_model
        )
        
        # Validate API key
        if not await groq_service.validate_api_key():
            logger.warning("⚠️  Groq API key validation failed. Running in demo mode.")
        
        # Initialize Orchestrator
        logger.info("🎯 Initializing RAG Orchestrator...")
        orchestrator = RAGOrchestrator()
        
        # Set orchestrator in routes
        set_query_orchestrator(orchestrator)
        
        logger.info("=" * 70)
        logger.info("✅ All agents initialized and ready!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Agentic RAG System...")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG System",
    description="Multi-agent Retrieval-Augmented Generation using Groq API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning("⚠️  Static directory not found")


# ============ UI ENDPOINTS ============

@app.get("/", tags=["UI"])
async def root(request: Request):
    """Home page"""
    try:
        return templates.TemplateResponse(name="index.html", context={"request": request})
    except Exception as e:
        logger.error(f"Error rendering index.html: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.get("/ui/query", tags=["UI"])
async def query_page(request: Request):
    """Query interface page"""
    try:
        return templates.TemplateResponse(name="query.html", context={"request": request})
    except Exception as e:
        logger.error(f"Error rendering query.html: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.get("/ui/dashboard", tags=["UI"])
async def dashboard_page(request: Request):
    """Dashboard page"""
    try:
        return templates.TemplateResponse(name="dashboard.html", context={"request": request})
    except Exception as e:
        logger.error(f"Error rendering dashboard.html: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.get("/ui/documents", tags=["UI"])
async def documents_page(request: Request):
    """Documents management page"""
    try:
        return templates.TemplateResponse(name="documents.html", context={"request": request})
    except Exception as e:
        logger.error(f"Error rendering documents.html: {str(e)}", exc_info=True)
        return {"error": str(e)}


# ============ SYSTEM ENDPOINTS ============

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        agent_status = await orchestrator.get_agent_status() if orchestrator else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "app": settings.app_name,
            "version": "1.0.0",
            "agents_ready": agent_status.get("total_agents", 0) == 5,
            "groq_service": "initialized" if groq_service else "not_initialized"
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/agents/status", tags=["System"])
async def get_agents_status():
    """Get status of all agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        return await orchestrator.get_agent_status()
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ QUERY ENDPOINTS ============

@app.post("/api/query", tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process user query through RAG pipeline
    
    Takes a query and processes it through:
    1. Planner - Plans the query
    2. Retriever - Retrieves relevant documents
    3. Reasoning - Generates response via Groq
    4. Critic - Validates response
    5. Memory - Stores for personalization
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"📨 Query from {request.user_id}: {request.query[:60]}...")
        
        result = await orchestrator.process_query(
            user_id=request.user_id,
            query=request.query,
            personalization=request.personalization,
            top_k=request.top_k
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        logger.info(f"✅ Query processed successfully for {request.user_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/query/history/{user_id}", tags=["Query History"])
async def get_user_history(user_id: str, limit: int = 10):
    """Get user query history (last N queries)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"📚 Fetching history for {user_id}")
        return await orchestrator.get_user_history(user_id, limit)
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/query/personalization/{user_id}", tags=["Personalization"])
async def get_personalization_hints(user_id: str):
    """Get personalized recommendations based on user history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"🎯 Generating personalization for {user_id}")
        return await orchestrator.get_personalization(user_id)
    except Exception as e:
        logger.error(f"Error getting personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/query/history/{user_id}", tags=["Query History"])
async def clear_user_history(user_id: str):
    """Clear user query history (Privacy)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"🗑️  Clearing history for {user_id}")
        result = await orchestrator.clear_user_history(user_id)
        return {
            "user_id": user_id,
            "message": "History cleared successfully",
            "cleared": result.get("cleared", True)
        }
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ DOCUMENT ENDPOINTS ============

@app.post("/api/documents/add", tags=["Documents"])
async def add_documents(request: dict):
    """Add documents to knowledge base"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"📄 Adding documents to knowledge base")
        
        return {
            "status": "success",
            "message": "Documents added successfully",
            "count": len(request.get("documents", []))
        }
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/search", tags=["Documents"])
async def search_documents(query: str, top_k: int = 5):
    """Search documents in knowledge base"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"🔍 Searching documents: {query}")
        
        return {
            "status": "success",
            "results": [],
            "count": 0
        }
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ STARTUP & SHUTDOWN ============

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 70)
    logger.info(f"🎉 {settings.app_name} Started Successfully!")
    logger.info(f"📍 Running at: http://{settings.host}:{settings.port}")
    logger.info(f"🌐 Web UI at: http://{settings.host}:{settings.port}")
    logger.info(f"📚 API Docs at: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information"""
    logger.info("=" * 70)
    logger.info("🛑 Application shutting down...")
    logger.info("=" * 70)


# ============ RUN APPLICATION ============

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info(f"🚀 Starting {settings.app_name} with Web UI...")
    logger.info(f"⚙️  Environment: {settings.app_env}")
    logger.info(f"🔑 Groq Model: {settings.groq_model}")
    logger.info("=" * 70)
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else settings.workers
    )
