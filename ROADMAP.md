# DFS Optimizer – Roadmap

## Current state (Oct 7, 2025)
- Dockerized 2-service app:
  - **frontend (nginx)** at http://localhost:8085, proxies `/api` → `backend:8000`
  - **backend (FastAPI)** with `/health`, `/optimize`
- CSV flow:
  - Upload CSV → **Load headers** → dropdown mapping → **Build mapping & Optimize**
  - Backend normalizes rows to canonical fields:
    - required: `player_id, name, team, position, salary, projection`
    - optional: exposure/ownership/deviation/floor/ceil/starter/progressive
  - Writes `data/jobs/<job_id>/pool_input.csv` + `summary.json`
- Repo hygiene:
  - `./data` is mounted into container as `/app/data`
  - `data/*` is ignored in Git (optionally keep `data/.gitkeep`)

## Near-term goals
1. **Editable Player Pool (recommended next)**
   - API: `GET /jobs/{job_id}/pool` (returns rows), `POST /jobs/{job_id}/pool` (persist edits)
   - UI: preview table with inline edits for `projection`, `max_exposure`, `min_exposure`; toggles `active`, `lock`, `exclude`; Save button
   - On first write, add columns if missing: `active=true`, `lock=false`, `exclude=false`

2. **Minimal Solve**
   - API: `POST /solve { job_id, num_lineups, site, sport, solver }`
   - Load pool, apply filters/locks/exposures, call pydfs, write `lineups.csv`, return 3 sample lineups

3. **Quality-of-life**
   - Position normalization (NFL/FD: strip `/FLEX`, map `D/ST`→`DST`)
   - LocalStorage: auto-apply last mapping per (Site,Sport)
   - Download links: `/jobs/{job_id}/pool.csv` & `/jobs/{job_id}/lineups.csv`

## Later
- Contest simulation module (10k sims)
- Advanced filters/slices + export to FD format
- Auth + multi-slate library
- Ownership inputs & templates per sport/site

## Tech notes
- Backend writes to `/app/data` (host `./data`)
- Keep `DATA_DIR = Path("/app/data")`
- For Linux perms: set docker-compose backend `user: "${UID}:${GID}"` and export env before `compose up`
