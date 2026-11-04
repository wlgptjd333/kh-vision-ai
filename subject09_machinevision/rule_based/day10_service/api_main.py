from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from day10_service.service_core import (
    PRODUCT_DIR,
    append_history,
    inspect_sample,
    latest_result,
    list_samples,
    read_history,
    stats,
)


STATIC_DIR = (
    PROJECT_ROOT
    / "day10_service"
    / "static"
)


app = FastAPI(
    title="Subject9 Machine Vision Service",
    version="1.0.0",
    description=(
        "Day10 FastAPI + Web Dashboard Practice"
    ),
)


class InspectRequest(BaseModel):
    file_name: str


app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


@app.get("/")
def home():
    return FileResponse(
        STATIC_DIR
        / "index.html"
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service":
            "Subject9 Machine Vision Service",
    }


@app.get("/api/samples")
def samples():
    return {
        "samples":
            list_samples()
    }


@app.get(
    "/api/sample-image/{file_name}"
)
def sample_image(
    file_name: str,
):
    if file_name not in list_samples():
        raise HTTPException(
            status_code=404,
            detail="등록되지 않은 Sample입니다.",
        )

    return FileResponse(
        PRODUCT_DIR
        / file_name
    )


@app.post("/api/inspect")
def inspect(
    request: InspectRequest,
):
    try:
        result = inspect_sample(
            request.file_name
        )

        append_history(
            result
        )

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except (
        KeyError,
        ValueError,
        RuntimeError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.get("/api/latest")
def latest():
    result = latest_result()

    return {
        "latest": result
    }


@app.get("/api/history")
def history(
    limit: int = 20,
):
    rows = read_history()

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    return {
        "history":
            rows[-limit:]
    }


@app.get("/api/stats")
def inspection_stats():
    return stats()
