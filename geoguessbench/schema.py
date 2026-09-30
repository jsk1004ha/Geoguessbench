"""Strict evaluator-side schemas. Dataset objects never enter model prompts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class Node(Strict):
    id: str = Field(min_length=1, max_length=160)
    image: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    panorama: bool = False
    heading: float | None = Field(default=None, ge=0, lt=360)
    sequence: str = ""
    neighbors: list[str] = Field(default_factory=list)
    attribution: str = ""
    source_url: str = ""


class Round(Strict):
    id: str = Field(min_length=1, max_length=160)
    start: str
    split: Literal["train", "dev", "test"] = "test"
    group: str = Field(min_length=1)
    country: str | None = None
    region: str | None = None
    start_heading: float | None = Field(default=None, ge=0, lt=360)


class Manifest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    fixture: bool = False
    provenance: dict = Field(default_factory=dict)
    nodes: list[Node] = Field(min_length=1)
    rounds: list[Round] = Field(min_length=1)


class Protocol(Strict):
    track: Literal["nmpz", "nm", "moving"] = "nmpz"
    split: Literal["train", "dev", "test"] = "test"
    seed: int = 42
    max_actions: int = Field(default=20, ge=0, le=1000)
    max_cost: float = Field(default=40, ge=0, le=10000)
    time_limit_s: float = Field(default=300, gt=0, le=86400)
    width: int = Field(default=960, ge=128, le=2048)
    height: int = Field(default=640, ge=128, le=2048)
    initial_fov: float = Field(default=90, ge=20, le=110)
    lower_band_mask: float = Field(default=0, ge=0, le=0.4)
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)


class Action(Strict):
    type: Literal["turn", "look", "zoom", "move", "submit", "abstain"]
    degrees: float | None = Field(default=None, ge=-180, le=180)
    fov: float | None = Field(default=None, ge=20, le=110)
    exit: int | None = Field(default=None, ge=0, le=100, strict=True)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    confidence_25km: float | None = Field(default=None, ge=0, le=1)
    evidence: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def fields_match_action(self):
        required = {"turn": {"degrees"}, "look": {"degrees"}, "zoom": {"fov"},
                    "move": {"exit"}, "submit": {"lat", "lon"}, "abstain": set()}[self.type]
        allowed = required | ({"confidence_25km"} if self.type == "submit" else set())
        for field in ("degrees", "fov", "exit", "lat", "lon", "confidence_25km"):
            value = getattr(self, field)
            if field in required and value is None:
                raise ValueError(f"{field} is required for {self.type}")
            if field not in allowed and value is not None:
                raise ValueError(f"{field} is not allowed for {self.type}")
        if self.type == "look" and abs(self.degrees) > 60:
            raise ValueError("look pitch must be between -60 and 60 degrees")
        return self
