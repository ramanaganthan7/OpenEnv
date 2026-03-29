"""
ClimateWatch — OpenEnv Server Entry Point
==========================================
Required by openenv validate for multi-mode deployment.
Starts the FastAPI server via uvicorn.

Usage:
  python -m server.app
  # or via project.scripts:
  server
"""

import uvicorn


def main():
    """Entry point for `server` CLI command (project.scripts)."""
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()
