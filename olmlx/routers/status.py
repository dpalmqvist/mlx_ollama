from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from olmlx import __version__
from olmlx.schemas.status import PsResponse, RunningModel, VersionResponse

router = APIRouter()


@router.get("/")
async def root():
    return PlainTextResponse("Ollama is running")


@router.head("/")
async def root_head():
    return PlainTextResponse("Ollama is running")


@router.get("/api/version")
async def version():
    return VersionResponse(version=__version__)


@router.get("/api/ps")
async def ps(request: Request):
    manager = request.app.state.model_manager
    loaded = manager.get_loaded()
    models = []
    for lm in loaded:
        expires = ""
        if lm.expires_at is not None:
            expires = datetime.fromtimestamp(lm.expires_at, tz=timezone.utc).isoformat()
        models.append(
            RunningModel(
                name=lm.name,
                model=lm.hf_path,
                size=lm.size_bytes,
                expires_at=expires,
                size_vram=lm.size_bytes,
                active_refs=lm.active_refs,
            )
        )
    return PsResponse(models=models)
