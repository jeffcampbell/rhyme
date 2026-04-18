"""CLI entry point for the web labeling interface."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Sifter Web — labeling interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--data-dir", default="data/web", help="Data directory (for SQLite fallback)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    from .app import create_app

    app = create_app(data_dir=args.data_dir, verbose=args.verbose or args.debug)
    print(f"Sifter Web running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
