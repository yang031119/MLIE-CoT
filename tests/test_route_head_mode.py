from __future__ import annotations
from pathlib import Path

from types import SimpleNamespace

import pytest


def test_head_router_preserves_explicit_checkpoint() -> None:
	from inference.self_route_version.modules.pipeline import configure_semantic_router

	args = SimpleNamespace(semantic_router="head", qwen_route_head="/models/route_head.pt")
	assert configure_semantic_router(args) == "head"
	assert args.qwen_route_head == "/models/route_head.pt"


@pytest.mark.parametrize("mode", ["qwen", "bagel"])
def test_generative_router_disables_route_head(mode: str) -> None:
	from inference.self_route_version.modules.pipeline import configure_semantic_router

	args = SimpleNamespace(semantic_router=mode, qwen_route_head="/models/route_head.pt")
	assert configure_semantic_router(args) == mode
	assert args.qwen_route_head == ""


def test_head_router_requires_checkpoint() -> None:
	from inference.self_route_version.modules.pipeline import configure_semantic_router

	with pytest.raises(ValueError, match="requires --qwen-route-head"):
		configure_semantic_router(SimpleNamespace(semantic_router="head", qwen_route_head=""))


def test_cli_accepts_head_router() -> None:
	from inference.self_route_version.modules.cli import build_parser

	args = build_parser().parse_args(["--semantic-router", "head", "--qwen-route-head", "/tmp/head.pt"])
	assert args.semantic_router == "head"
	assert args.qwen_route_head == Path("/tmp/head.pt")


def test_cli_defaults_to_bundled_route_head() -> None:
	from inference.self_route_version.modules.cli import build_parser
	from inference.self_route_version.modules.config import DEFAULT_QWEN_ROUTE_HEAD

	args = build_parser().parse_args(["--semantic-router", "head"])
	assert args.qwen_route_head == DEFAULT_QWEN_ROUTE_HEAD
	assert args.qwen_route_head == Path(__file__).resolve().parents[1] / "data" / "router_heads" / "route_head.pt"


def test_cli_cross_executor_transfer_is_explicit_opt_in() -> None:
	from inference.self_route_version.modules.cli import build_parser

	default_args = build_parser().parse_args(["--semantic-router", "head"])
	transfer_args = build_parser().parse_args(
		["--semantic-router", "head", "--allow-cross-executor-route-head-transfer"]
	)
	assert default_args.allow_cross_executor_route_head_transfer is False
	assert transfer_args.allow_cross_executor_route_head_transfer is True
