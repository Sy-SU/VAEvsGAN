"""Lazy registry for model workflows exposed through ``tools/``."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from .workflow import ModelWorkflow


@dataclass(frozen=True)
class WorkflowSpec:
    module: str
    attribute: str
    description: str


_WORKFLOWS: dict[str, WorkflowSpec] = {}


def register_model(
    name: str,
    module: str,
    *,
    attribute: str = "workflow",
    description: str,
) -> None:
    """Register a lazily imported model workflow."""
    normalized = name.strip().lower()
    if not normalized or normalized in _WORKFLOWS:
        raise ValueError(f"model workflow is already registered or invalid: {name!r}")
    _WORKFLOWS[normalized] = WorkflowSpec(module, attribute, description)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_WORKFLOWS))


def model_descriptions() -> dict[str, str]:
    return {name: spec.description for name, spec in sorted(_WORKFLOWS.items())}


def get_workflow(name: str) -> ModelWorkflow:
    normalized = name.lower()
    try:
        spec = _WORKFLOWS[normalized]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(f"unknown model {name!r}; available models: {choices}") from error

    module = import_module(spec.module)
    workflow = getattr(module, spec.attribute)
    if workflow.name != normalized:
        raise RuntimeError(
            f"workflow {spec.module}.{spec.attribute} declares name "
            f"{workflow.name!r}, expected {normalized!r}"
        )
    return workflow


register_model(
    "vae",
    "vae.workflow",
    description="conditional convolutional VAE for MNIST",
)
register_model(
    "gan",
    "gan.workflow",
    description="conditional DCGAN for MNIST",
)
