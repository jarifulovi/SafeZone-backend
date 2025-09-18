from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from services.safety_service import SeverityService
from routes.safety_routes import router as safety_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Generating severity data cache...")
    SeverityService.generate_severity_data()
    print("Server startup complete!")

    yield  # Application runs here

    # Shutdown (optional)
    print("Shutting down...")

app = FastAPI(title="SafeZone Bangladesh Prototype", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(safety_router)
