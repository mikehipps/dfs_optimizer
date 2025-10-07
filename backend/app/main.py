from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# pydfs imports (you'll use these in the next steps)
# from pydfs_lineup_optimizer import get_optimizer, Site, Sport

app = FastAPI(title="DFS Optimizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    # v1 placeholder: we’ll replace with full schema later
    site: str = "FANDUEL"
    sport: str = "NFL"
    solver: str = "mip"   # "mip" | "pulp" | "gurobi" (if licensed)
    pool_size: int = 150  # number of lineups to build
    params: Dict[str, Any] = {}  # future: rules, randomness, exposures

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/optimize")
async def optimize(
    settings: str = Form(...),
    csv: UploadFile = File(...)
):
    """
    Multipart endpoint to receive:
    - 'settings' (JSON string, conforms to OptimizeRequest)
    - 'csv' (the merged projections/player pool CSV)
    For now, we just echo back—wire pydfs in the next iteration.
    """
    text = await csv.read()
    size_bytes = len(text)
    return {"received_csv_bytes": size_bytes, "settings": settings}
