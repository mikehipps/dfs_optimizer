from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict
from uuid import uuid4
from pathlib import Path
import io, csv as csvmod, json, os, re

from pydfs_lineup_optimizer import get_optimizer
from pydfs_lineup_optimizer.player import Player
from pydfs_lineup_optimizer.exceptions import LineupOptimizerException, GenerateLineupException
from pydfs_lineup_optimizer.solvers.mip_solver import MIPSolver
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver

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


class SolveRequest(BaseModel):
    job_id: str
    num_lineups: int = 20
    site: str = "FANDUEL"
    sport: str = "NFL"
    solver: str = "mip"

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


def _parse_positions(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).replace("D/ST", "DST")
    parts = re.split(r"[/,;]+", text)
    positions = [p.strip().upper() for p in parts if p and p.strip()]
    return positions


def _split_name(full_name: str) -> Tuple[str, str]:
    bits = full_name.split()
    if not bits:
        return "", ""
    if len(bits) == 1:
        return bits[0], ""
    return bits[0], " ".join(bits[1:])


def _parse_percentish(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        num = float(text)
    except Exception:
        return None
    if num < 0:
        num = 0.0
    if num > 1.0:
        if num <= 100.0:
            num = num / 100.0
        else:
            num = 1.0
    return num


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _to_float(text)


def _optional_bool(value: Any) -> Optional[bool]:
    result = _to_bool(value)
    return result


def _read_pool_rows(job_id: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    job_dir = DATA_DIR / "jobs" / job_id
    pool_csv = job_dir / "pool_input.csv"
    if not pool_csv.exists():
        raise FileNotFoundError("pool_input.csv not found")

    with pool_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csvmod.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    return rows, headers


def _coerce_pool_player(row: Dict[str, Any], index: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    projection_val = row.get("projection")
    projection = _to_float(projection_val)
    salary = _to_int(row.get("salary"))

    exclude_raw = row.get("exclude")
    active_raw = row.get("active")
    lock_raw = row.get("lock")

    exclude = _optional_bool(exclude_raw)
    if exclude is None:
        exclude = _projection_is_zero(projection_val)

    active = _optional_bool(active_raw)
    if active is None:
        active = not exclude

    lock = bool(_optional_bool(lock_raw) or False)

    if exclude:
        active = False

    if lock and exclude:
        errors.append("player is both locked and excluded")

    if lock and not active:
        # lock should imply active participation
        errors.append("locked player is marked inactive")

    player_id = str(row.get("player_id") or "").strip()
    if not player_id:
        errors.append("missing player_id")

    name = str(row.get("name") or "").strip()
    if not name:
        errors.append("missing name")

    team = str(row.get("team") or "").strip()

    positions = _parse_positions(row.get("position"))
    if not positions:
        errors.append("missing position")

    if salary is None:
        errors.append("invalid salary")

    if projection is None:
        errors.append("invalid projection")

    max_exposure = _parse_percentish(row.get("max_exposure"))
    min_exposure = _parse_percentish(row.get("min_exposure"))
    if max_exposure is not None and min_exposure is not None and min_exposure > max_exposure:
        errors.append("min_exposure greater than max_exposure")

    projected_ownership = _parse_percentish(row.get("projected_ownership"))
    min_deviation = _optional_float(row.get("min_deviation"))
    max_deviation = _optional_float(row.get("max_deviation"))
    projection_floor = _optional_float(row.get("projection_floor"))
    projection_ceil = _optional_float(row.get("projection_ceil"))
    progressive_scale = _optional_float(row.get("progressive_scale"))
    confirmed_starter = _optional_bool(row.get("confirmed_starter"))

    include = (active and not exclude) or lock

    first_name, last_name = _split_name(name)

    if errors:
        return None, errors

    player = {
        "player_id": player_id,
        "first_name": first_name or name,
        "last_name": last_name,
        "full_name": name,
        "team": team,
        "positions": positions,
        "salary": salary,
        "projection": projection,
        "lock": lock,
        "active": active,
        "exclude": exclude,
        "include": include,
        "max_exposure": max_exposure,
        "min_exposure": min_exposure,
        "projected_ownership": projected_ownership,
        "min_deviation": min_deviation,
        "max_deviation": max_deviation,
        "projection_floor": projection_floor,
        "projection_ceil": projection_ceil,
        "progressive_scale": progressive_scale,
        "confirmed_starter": confirmed_starter,
        "row_index": index,
    }
    return player, []


def _prepare_players_for_solver(job_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, int]]:
    rows, _ = _read_pool_rows(job_id)
    players: List[Dict[str, Any]] = []
    locked: List[Dict[str, Any]] = []
    warnings: List[str] = []
    stats = {
        "total_rows": len(rows),
        "eligible": 0,
        "skipped_inactive": 0,
    }

    for idx, row in enumerate(rows, start=2):
        player, errors = _coerce_pool_player(row, idx)
        if errors:
            lock_flag = bool(_optional_bool(row.get("lock")))
            message = f"Row {idx}: {', '.join(errors)}"
            if lock_flag:
                raise ValueError(message)
            warnings.append(message)
            continue

        if not player["include"]:
            stats["skipped_inactive"] += 1
            continue

        players.append(player)
        stats["eligible"] += 1
        if player["lock"]:
            locked.append(player)

    return players, locked, warnings, stats


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
        projection_val = row.get("projection")
        projection_is_zero = _projection_is_zero(projection_val)

        exclude_val = row.get("exclude")
        if exclude_val is None or str(exclude_val).strip() == "":
            # Default: auto-exclude zero projection players
            row["exclude"] = "true" if projection_is_zero else "false"
        else:
            row["exclude"] = _csv_bool_to_str(exclude_val)

        active_val = row.get("active")
        if active_val is None or str(active_val).strip() == "":
            # Default active only when the player is not auto-excluded
            row["active"] = "false" if row.get("exclude") == "true" else "true"
        else:
            row["active"] = _csv_bool_to_str(active_val)
            if row.get("exclude") == "true":
                # Prevent conflicting states when exclude is active
                row["active"] = "false"

        lock_val = row.get("lock")
        if lock_val is None or str(lock_val).strip() == "":
            row["lock"] = "false"
        else:
            row["lock"] = _csv_bool_to_str(lock_val)

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
                projection_val = row.get("projection")
                projection_is_zero = _projection_is_zero(projection_val)
                if col == "active":
                    row[col] = "false" if projection_is_zero else "true"
                elif col == "exclude":
                    row[col] = "true" if projection_is_zero else "false"
                elif col == "lock":
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

    for row in current_rows:
        if row.get("exclude") == "true":
            row["active"] = "false"

    with pool_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in current_rows:
            writer.writerow(row)

    return {"ok": True, "changed": changed, "count": len(current_rows)}


SPORT_ALIASES = {
    "NFL": "FOOTBALL",
    "NCAAF": "COLLEGE_FOOTBALL",
    "CFL": "CANADIAN_FOOTBALL",
    "NBA": "BASKETBALL",
    "WNBA": "WNBA",
    "MLB": "BASEBALL",
    "NHL": "HOCKEY",
    "LOL": "LEAGUE_OF_LEGENDS",
}


ALLOWED_SOLVERS = {
    "mip": MIPSolver,
    "pulp": PuLPSolver,
}


def _resolve_sport(value: str) -> str:
    key = (value or "").upper()
    return SPORT_ALIASES.get(key, key)


def _build_optimizer_player(data: Dict[str, Any]) -> Player:
    return Player(
        player_id=data["player_id"],
        first_name=data["first_name"],
        last_name=data["last_name"],
        positions=data["positions"],
        team=data["team"],
        salary=float(data["salary"]),
        fppg=float(data["projection"]),
        max_exposure=data["max_exposure"],
        min_exposure=data["min_exposure"],
        projected_ownership=data["projected_ownership"],
        min_deviation=data["min_deviation"],
        max_deviation=data["max_deviation"],
        is_confirmed_starter=data["confirmed_starter"],
        fppg_floor=data["projection_floor"],
        fppg_ceil=data["projection_ceil"],
        progressive_scale=data["progressive_scale"],
        original_positions=data["positions"],
    )


def _diagnose_lineup_constraints(optimizer, players: List[Dict[str, Any]], locked: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide visibility into roster requirements versus the available pool."""

    lineup_positions = getattr(getattr(optimizer, "settings", None), "positions", []) or []

    position_counts: Counter[str] = Counter()
    for player in players:
        for pos in player.get("positions", []):
            position_counts[pos] += 1

    locked_counts: Counter[str] = Counter()
    for player in locked:
        for pos in player.get("positions", []):
            locked_counts[pos] += 1

    slot_map = defaultdict(lambda: {"name": None, "allowed": (), "required": 0})
    for slot in lineup_positions:
        allowed = tuple(slot.positions)
        key = (slot.name, allowed)
        entry = slot_map[key]
        entry["name"] = slot.name
        entry["allowed"] = allowed
        entry["required"] += 1

    slot_requirements: List[Dict[str, Any]] = []
    issues: List[str] = []

    for entry in slot_map.values():
        allowed = entry["allowed"]
        allowed_set = set(allowed)
        eligible = sum(1 for player in players if allowed_set & set(player.get("positions", [])))
        locked_for_slot = sum(1 for player in locked if allowed_set & set(player.get("positions", [])))

        item = {
            "name": entry["name"],
            "allowed": list(allowed),
            "required": entry["required"],
            "eligible": eligible,
            "locked": locked_for_slot,
        }
        slot_requirements.append(item)

        if eligible < entry["required"]:
            slot_name = entry["name"] or "/".join(allowed)
            issues.append(f"Not enough eligible players for {slot_name}: need {entry['required']}, have {eligible}")
        if locked_for_slot > entry["required"]:
            slot_name = entry["name"] or "/".join(allowed)
            issues.append(f"Too many locked players for {slot_name}: locked {locked_for_slot}, slots {entry['required']}")

    return {
        "position_counts": dict(position_counts),
        "locked_counts": dict(locked_counts),
        "slot_requirements": slot_requirements,
        "issues": issues,
    }


@app.post("/solve")
async def solve(request: SolveRequest):
    job_id = request.job_id
    try:
        players_data, locked_data, warnings, stats = _prepare_players_for_solver(job_id)
    except FileNotFoundError:
        return {"ok": False, "error": "pool_input.csv not found", "job_id": job_id}
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "job_id": job_id}

    if not players_data:
        return {
            "ok": False,
            "job_id": job_id,
            "error": "No eligible players after applying active/exclude flags.",
            "warnings": warnings,
            "stats": stats,
        }

    solver_key = (request.solver or "mip").lower()
    solver_cls = ALLOWED_SOLVERS.get(solver_key)
    if solver_cls is None:
        return {"ok": False, "job_id": job_id, "error": f"Unsupported solver '{request.solver}'"}

    site_key = (request.site or "FANDUEL").upper()
    sport_key = _resolve_sport(request.sport)

    try:
        optimizer = get_optimizer(site_key, sport_key, solver=solver_cls)
    except Exception as exc:
        return {"ok": False, "job_id": job_id, "error": f"Failed to initialize optimizer: {exc}"}

    optimizer_players: List[Player] = []
    player_lookup: Dict[str, Player] = {}
    for pdata in players_data:
        player_obj = _build_optimizer_player(pdata)
        optimizer_players.append(player_obj)
        player_lookup[pdata["player_id"]] = player_obj

    optimizer.load_players(optimizer_players)

    diagnostics = _diagnose_lineup_constraints(optimizer, players_data, locked_data)
    if diagnostics.get("issues"):
        message = "; ".join(diagnostics["issues"])
        return {
            "ok": False,
            "job_id": job_id,
            "error": f"Lineup constraints invalid: {message}",
            "warnings": warnings,
            "stats": stats,
            "diagnostics": diagnostics,
        }

    locked_ids = {p["player_id"] for p in locked_data}
    for pdata in locked_data:
        player_obj = player_lookup.get(pdata["player_id"])
        if player_obj:
            try:
                optimizer.add_player_to_lineup(player_obj)
            except LineupOptimizerException as exc:
                diagnostics = _diagnose_lineup_constraints(optimizer, players_data, locked_data)
                return {
                    "ok": False,
                    "job_id": job_id,
                    "error": str(exc),
                    "warnings": warnings,
                    "stats": stats,
                    "diagnostics": diagnostics,
                }

    requested = max(1, min(int(request.num_lineups or 1), 150))

    lineups: List[Dict[str, Any]] = []
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    lineups_path = job_dir / "lineups.csv"

    try:
        for idx, lineup in enumerate(optimizer.optimize(requested), start=1):
            lineup_players: List[Dict[str, Any]] = []
            for lp in lineup.lineup:
                lineup_players.append(
                    {
                        "player_id": lp.id,
                        "name": lp.full_name,
                        "team": lp.team,
                        "positions": list(lp.positions),
                        "lineup_position": lp.lineup_position,
                        "salary": int(lp.salary),
                        "projection": round(float(lp.fppg), 3),
                        "used_projection": round(float(lp.used_fppg), 3) if lp.used_fppg is not None else None,
                        "locked": lp.id in locked_ids,
                    }
                )

            lineups.append(
                {
                    "index": idx,
                    "salary": int(lineup.salary_costs),
                    "projection": round(float(lineup.fantasy_points_projection), 3),
                    "projection_actual": round(float(lineup.actual_fantasy_points_projection), 3),
                    "players": lineup_players,
                }
            )
    except GenerateLineupException as exc:
        diagnostics = _diagnose_lineup_constraints(optimizer, players_data, locked_data)
        return {
            "ok": False,
            "job_id": job_id,
            "error": str(exc),
            "warnings": warnings,
            "stats": stats,
            "diagnostics": diagnostics,
        }
    except LineupOptimizerException as exc:
        diagnostics = _diagnose_lineup_constraints(optimizer, players_data, locked_data)
        return {
            "ok": False,
            "job_id": job_id,
            "error": str(exc),
            "warnings": warnings,
            "stats": stats,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        diagnostics = _diagnose_lineup_constraints(optimizer, players_data, locked_data)
        return {
            "ok": False,
            "job_id": job_id,
            "error": f"Solver error: {exc}",
            "warnings": warnings,
            "stats": stats,
            "diagnostics": diagnostics,
        }

    if not lineups:
        return {
            "ok": False,
            "job_id": job_id,
            "error": "Solver did not return any lineups.",
            "warnings": warnings,
            "stats": stats,
        }

    csv_headers = [
        "lineup_index",
        "lineup_salary",
        "lineup_projection",
        "lineup_projection_actual",
        "slot",
        "player_id",
        "name",
        "team",
        "positions",
        "salary",
        "projection",
        "used_projection",
        "locked",
    ]

    with lineups_path.open("w", encoding="utf-8", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for lineup in lineups:
            for player in lineup["players"]:
                writer.writerow(
                    {
                        "lineup_index": lineup["index"],
                        "lineup_salary": lineup["salary"],
                        "lineup_projection": lineup["projection"],
                        "lineup_projection_actual": lineup["projection_actual"],
                        "slot": player["lineup_position"],
                        "player_id": player["player_id"],
                        "name": player["name"],
                        "team": player["team"],
                        "positions": "/".join(player["positions"]),
                        "salary": player["salary"],
                        "projection": player["projection"],
                        "used_projection": player["used_projection"] if player["used_projection"] is not None else "",
                        "locked": "true" if player["locked"] else "false",
                    }
                )

    preview = lineups[:3]
    response = {
        "ok": True,
        "job_id": job_id,
        "requested": requested,
        "generated": len(lineups),
        "lineups": preview,
        "lineups_csv": f"jobs/{job_id}/lineups.csv",
        "warnings": warnings,
        "stats": stats,
        "solver": solver_key,
        "site": site_key,
        "sport": sport_key,
        "locked_players": len(locked_data),
    }
    if len(lineups) > len(preview):
        response["preview_count"] = len(preview)

    return response
