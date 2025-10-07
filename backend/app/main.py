from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List
import io, csv, json

# from pydfs_lineup_optimizer import get_optimizer, Site, Sport  # used later

app = FastAPI(title="DFS Optimizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    site: str = "FANDUEL"
    sport: str = "NFL"
    solver: str = "mip"        # "mip" | "pulp" | "gurobi"
    pool_size: int = 150
    params: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"ok": True}

def _parse_csv_summary(binary: bytes) -> Dict[str, Any]:
    """Decode CSV bytes, sniff dialect, return headers/row_count/sample_rows/warnings."""
    warnings: List[str] = []
    # 1) decode safely
    text = binary.decode("utf-8", errors="replace")
    buf = io.StringIO(text)

    # 2) try to sniff dialect
    try:
        sample = text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        buf.seek(0)
    except Exception:
        warnings.append("Could not sniff delimiter; defaulted to comma.")
        dialect = csv.get_dialect("excel")  # comma
        buf.seek(0)

    # 3) read as DictReader
    reader = csv.DictReader(buf, dialect=dialect)
    headers = reader.fieldnames or []

    # 4) collect first 3 rows + count
    sample_rows: List[Dict[str, Any]] = []
    count = 0
    for row in reader:
        if count < 3:
            sample_rows.append(row)
        count += 1

    return {
        "headers": headers,
        "row_count": count,
        "sample_rows": sample_rows,
        "warnings": warnings,
    }

@app.post("/optimize")
async def optimize(settings: str = Form(...), csv: UploadFile = File(...)):
    """
    Multipart endpoint:
    - 'settings': JSON string matching OptimizeRequest
    - 'csv': uploaded projections/player pool CSV

    For now: parse CSV and return a summary; no optimization yet.
    """
    raw = await csv.read()

    # Parse settings safely
    parsed_settings: Dict[str, Any]
    settings_model: Optional[OptimizeRequest] = None
    try:
        settings_json = json.loads(settings)
        settings_model = OptimizeRequest(**settings_json)
        parsed_settings = settings_model.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        parsed_settings = {"settings_raw": settings, "parse_error": str(e)}

    summary = _parse_csv_summary(raw)

    return {
        "ok": True,
        "received_csv_bytes": len(raw),
        "settings": parsed_settings,
        "csv_summary": summary,
    }
