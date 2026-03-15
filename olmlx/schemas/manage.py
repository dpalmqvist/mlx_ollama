from pydantic import BaseModel, Field


class CopyRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=256)
    destination: str = Field(..., min_length=1, max_length=256)


class DeleteRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)


class CreateRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    modelfile: str | None = None
    stream: bool = True
    path: str | None = None
    quantize: str | None = None


class WarmupRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    keep_alive: str | None = None


class AbortRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)


class UnloadRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
