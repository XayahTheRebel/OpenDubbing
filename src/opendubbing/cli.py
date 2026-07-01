"""Command-line interface for OpenDubbing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opendubbing.application import list_providers, process_video, run_api_server


def main(argv: list[str] | None = None) -> int:
    """Entry point for the opendubbing CLI."""
    parser = argparse.ArgumentParser(
        prog="opendubbing",
        description="OpenDubbing: AI video dubbing platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process a video")
    process_parser.add_argument("--input", required=True, help="Input video path")
    process_parser.add_argument("--config", required=True, help="Configuration file")
    process_parser.add_argument(
        "--resume", action="store_true", help="Resume from cached state"
    )
    process_parser.add_argument(
        "--workspace", default=None, help="Override workspace root"
    )

    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--config", required=True, help="Configuration file")
    api_parser.add_argument("--host", default=None, help="API server host")
    api_parser.add_argument("--port", type=int, default=None, help="API server port")

    subparsers.add_parser("providers", help="List registered providers")

    clean_parser = subparsers.add_parser("clean", help="Clean workspace cache")
    clean_parser.add_argument("--workspace", required=True, help="Workspace root")

    args = parser.parse_args(argv)

    if args.command == "process":
        process_video(
            input_path=Path(args.input),
            config_path=Path(args.config),
            resume=args.resume,
            workspace_root=args.workspace,
        )
    elif args.command == "api":
        run_api_server(
            config_path=Path(args.config),
            host=args.host,
            port=args.port,
        )
    elif args.command == "providers":
        list_providers()
    elif args.command == "clean":
        from opendubbing.core.workspace import Workspace
        from opendubbing.pipeline.cache import PipelineCache

        workspace = Workspace(args.workspace)
        PipelineCache(workspace).reset()
        print(f"Cleaned workspace cache: {workspace.root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
