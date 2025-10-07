from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from pathlib import Path
import io, csv as csvmod, json, os

DATA_DIR = Path("/app/data")  # added near the top of the file is fine

app = FastAPI(title="DFS Optimizer API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    site: str = "FANDUEL"
    sport: str = "NFL"
    solver: str = "mip"
    pool_size: int = 150
    params: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"ok": True}

def _parse_csv(binary: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    warnings: List[str] = []
    text = binary.decode("utf-8", errors="replace")
    buf = io.StringIO(text)
    try:
        sample = text[:4096]
        dialect = csvmod.Sniffer().sniff(sample, delimiters=",;\t|")
        buf.seek(0)
    except Exception:
        warnings.append("Could not sniff delimiter; defaulted to comma.")
        dialect = csvmod.get_dialect("excel")
        buf.seek(0)
    reader = csvmod.DictReader(buf, dialect=dialect)
    headers = reader.fieldnames or []
    rows = [row for row in reader]
    summary = {
        "headers": headers,
        "row_count": len(rows),
        "sample_rows": rows[:3],
        "warnings": warnings,
    }
    return rows, summary

def _to_bool(v: Any) -> Optional[bool]:
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y"): return True
    if s in ("0","false","f","no","n"): return False
    return None

def _to_int(v: Any) -> Optional[int]:
    try: return int(float(str(v).replace(",","")))
    except Exception: return None

def _to_float(v: Any) -> Optional[float]:
    try: return float(str(v).replace(",",""))
    except Exception: return None

REQUIRED_KEYS = ["player_id","name","team","position","salary","projection"]
OPTIONAL_NUMERIC = {
    "max_exposure": _to_float,
    "min_exposure": _to_float,
    "projected_ownership": _to_float,
    "min_deviation": _to_float,
    "max_deviation": _to_float,
    "projection_floor": _to_float,
    "projection_ceil": _to_float,
}
OPTIONAL_BOOL = {
    "confirmed_starter": _to_bool,
    "progressive_scale": _to_bool,
}

def _normalize_rows(rows: List[Dict[str, Any]], mapping: Dict[str, str]) -> Dict[str, Any]:
    players: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    csv_headers = set(rows[0].keys()) if rows else set()
    bad_refs = [dst for dst, src in mapping.items() if src and src not in csv_headers]
    if bad_refs:
        errors.append({"row": None, "error": f"Mapping refers to missing columns: {bad_refs}"})

    for i, r in enumerate(rows):
        item: Dict[str, Any] = {}
        err: List[str] = []
        for key in REQUIRED_KEYS:
            src = mapping.get(key)
            val = r.get(src) if src else None
            if key == "salary":
                coerced = _to_int(val)
                if coerced is None: err.append(f"Invalid salary from '{src}': {val!r}")
                item[key] = coerced
            elif key == "projection":
                coerced = _to_float(val)
                if coerced is None: err.append(f"Invalid projection from '{src}': {val!r}")
                item[key] = coerced
            else:
                sval = ("" if val is None else str(val).strip())
                if not sval: err.append(f"Missing {key} from '{src}'")
                item[key] = sval
        for key, caster in OPTIONAL_NUMERIC.items():
            src = mapping.get(key)
            if src: item[key] = caster(r.get(src))
        for key, caster in OPTIONAL_BOOL.items():
            src = mapping.get(key)
            if src: item[key] = caster(r.get(src))
        if err:
            errors.append({"row": i, "error": "; ".join(err)})
            continue
        players.append(item)

    return {
        "players": players,
        "count_ok": len(players),
        "count_invalid": len(errors),
        "errors": errors[:50],
    }

@app.post("/optimize")
async def optimize(
    settings: str = Form(...),
    file: UploadFile = File(..., alias="csv"),
    mapping: Optional[str] = Form(None),
):
    raw = await file.read()

    try:
        settings_json = json.loads(settings)
        settings_model = OptimizeRequest(**settings_json)
        parsed_settings = settings_model.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        parsed_settings = {"settings_raw": settings, "parse_error": str(e)}

    rows, csv_summary = _parse_csv(raw)

    normalized = None
    job = None  # NEW: job metadata
    if mapping:
        try:
            mapping_json = json.loads(mapping)
            if not isinstance(mapping_json, dict):
                raise ValueError("mapping must be a JSON object")
            normalized = _normalize_rows(rows, mapping_json)

            # NEW: persist normalized players if we have any
            players = (normalized or {}).get("players") or []
            if players:
                job_id = str(uuid4())
                job_dir = DATA_DIR / "jobs" / job_id
                job_dir.mkdir(parents=True, exist_ok=True)

                # choose headers = union of keys across players, stable order
                base_headers = ["player_id","name","team","position","salary","projection"]
                extra_headers = sorted({k for p in players for k in p.keys()} - set(base_headers))
                headers = base_headers + extra_headers

                out_csv = job_dir / "pool_input.csv"
                with out_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csvmod.DictWriter(f, fieldnames=headers)
                    w.writeheader()
                    for p in players:
                        w.writerow(p)

                # optional: stash a tiny summary.json too
                summary_path = job_dir / "summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump({
                        "site": parsed_settings.get("site"),
                        "sport": parsed_settings.get("sport"),
                        "count_ok": normalized.get("count_ok"),
                        "count_invalid": normalized.get("count_invalid"),
                    }, f, ensure_ascii=False, indent=2)

                job = {
                    "job_id": job_id,
                    "dir": str(job_dir),
                    "pool_csv": str(out_csv),
                    "summary_json": str(summary_path),
                }
        except Exception as e:
            normalized = {"error": f"Invalid mapping: {e}"}

    return {
        "ok": True,
        "received_csv_bytes": len(raw),
        "settings": parsed_settings,
        "csv_summary": csv_summary,
        "normalized": normalized,
        "job": job,  # NEW: paths + id (None if no mapping or no players)
    }


def _csv_bool_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return "true"
    if s in ("0", "false", "f", "no", "n"):
        return "false"
    return "false"


def _csv_float_or_empty(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(float(v))
    except Exception:
        return ""


def _projection_is_zero(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    try:
        return float(s) == 0.0
    except Exception:
        return False


@app.get("/jobs/{job_id}/pool")
def get_pool(job_id: str):
    job_dir = DATA_DIR / "jobs" / job_id
    pool_csv = job_dir / "pool_input.csv"
    if not pool_csv.exists():
        return {"players": [], "count": 0, "error": "pool_input.csv not found"}

    with pool_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csvmod.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []

    default_cols = ["active", "lock", "exclude", "max_exposure", "min_exposure"]
    for col in default_cols:
        if col not in headers:
            headers.append(col)

    for row in rows:
        # normalize truthy flags and ensure defaults when missing/blank
        active_val = row.get("active")
        if active_val is None or str(active_val).strip() == "":
            row["active"] = "true"
        else:
            row["active"] = _csv_bool_to_str(active_val)

        lock_val = row.get("lock")
        if lock_val is None or str(lock_val).strip() == "":
            row["lock"] = "false"
        else:
            row["lock"] = _csv_bool_to_str(lock_val)

        exclude_val = row.get("exclude")
        if exclude_val is None or str(exclude_val).strip() == "":
            row["exclude"] = "true" if _projection_is_zero(row.get("projection")) else "false"
        else:
            row["exclude"] = _csv_bool_to_str(exclude_val)

        for exposure_key in ("max_exposure", "min_exposure"):
            exposure_val = row.get(exposure_key)
            if exposure_val is None or str(exposure_val).strip() == "":
                row[exposure_key] = ""
            else:
                row[exposure_key] = _csv_float_or_empty(exposure_val)

    return {"players": rows, "count": len(rows)}


@app.post("/jobs/{job_id}/pool")
async def update_pool(job_id: str, body: Dict[str, Any]):
    job_dir = DATA_DIR / "jobs" / job_id
    pool_csv = job_dir / "pool_input.csv"
    if not pool_csv.exists():
        return {"ok": False, "error": "pool_input.csv not found"}

    with pool_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csvmod.DictReader(f)
        current_rows = list(reader)
        headers = reader.fieldnames or []

    default_cols = ["active", "lock", "exclude", "max_exposure", "min_exposure"]
    for col in default_cols:
        if col not in headers:
            headers.append(col)
            for row in current_rows:
                if col == "active":
                    row[col] = "true"
                elif col in ("lock", "exclude"):
                    row[col] = "false"
                else:
                    row[col] = ""

    index = {row.get("player_id"): i for i, row in enumerate(current_rows)}

    updates = body.get("players") or []
    changed = 0
    for update in updates:
        player_id = update.get("player_id")
        if player_id is None or player_id not in index:
            continue

        row = current_rows[index[player_id]]
        row_changed = False

        for key, value in update.items():
            if key == "player_id":
                continue

            if key not in headers:
                headers.append(key)
                for existing_row in current_rows:
                    existing_row.setdefault(key, "")

            if key == "salary":
                try:
                    row[key] = str(int(float(value)))
                    row_changed = True
                except Exception:
                    continue
            elif key == "projection":
                try:
                    row[key] = str(float(value))
                    row_changed = True
                except Exception:
                    continue
            elif key in ("active", "lock", "exclude"):
                row[key] = _csv_bool_to_str(value)
                row_changed = True
            elif key in ("max_exposure", "min_exposure"):
                row[key] = _csv_float_or_empty(value)
                row_changed = True
            else:
                row[key] = "" if value is None else str(value)
                row_changed = True

        if row_changed:
            changed += 1

    with pool_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in current_rows:
            writer.writerow(row)

    return {"ok": True, "changed": changed, "count": len(current_rows)}
