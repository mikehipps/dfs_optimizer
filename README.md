# DFS Optimizer (Dockerized scaffold)

Backend: **FastAPI** + **pydfs-lineup-optimizer** with **CBC** via `python-mip`.
Frontend: Static placeholder served by **nginx**.

## Run

```bash
docker compose up --build
```

## Creating a "return to here" restore point

When you reach a stable milestone and want a quick way to jump back to it, create
either a lightweight tag or a dedicated branch in Git. Both approaches record the
current commit so you can return to it later without disrupting ongoing work.

### Option 1: lightweight tag

```bash
git tag snapshot-2024-xx-yy
git push origin snapshot-2024-xx-yy
```

Tags are ideal for bookmarking a state. Replace `snapshot-2024-xx-yy` with any
descriptive label. To return later, either check out the commit directly or
create a temporary branch from the tag:

```bash
git checkout snapshot-2024-xx-yy    # detached HEAD at the tagged commit
# or
git checkout -b restore snapshot-2024-xx-yy
```

### Option 2: checkpoint branch

```bash
git checkout -b checkpoint/stable-ui
git push -u origin checkpoint/stable-ui
```

Checkpoint branches behave like ordinary branches, so you can merge or cherry
pick from them as needed. When you are ready to return to that snapshot:

```bash
git checkout checkpoint/stable-ui
```

You can remove the branch or tag later with `git branch -D` / `git push origin
--delete` or `git tag -d` / `git push origin :refs/tags/<tag>` once it is no
longer needed.
