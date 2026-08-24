#!/usr/bin/env python3
"""Start the official NVIDIA NeMo Switchyard *Rust* server on :4000.

This is switchyard_rust.server.Server — the same native proxy as
`switchyard-server`, shipped in the nemo-switchyard 0.2.0 macOS wheel.
It is not the educational Python classifier in src/router.py.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

CONFIG = ROOT / "configs" / "routes.nim.toml"
HOST_PORT = int(os.getenv("SWITCHYARD_PORT", "4000"))


def main() -> int:
    if not os.getenv("NVIDIA_API_KEY", "").strip():
        print("ERROR: NVIDIA_API_KEY is not set (see .env)", file=sys.stderr)
        return 1
    if not CONFIG.exists():
        print(f"ERROR: missing {CONFIG}", file=sys.stderr)
        return 1

    from switchyard_rust.server import Server

    print(f"Starting native Switchyard  {CONFIG.name}  port={HOST_PORT}", flush=True)
    server = Server(str(CONFIG), port=HOST_PORT)
    print(f"  base_url : {server.base_url}", flush=True)
    print(f"  port     : {server.port}", flush=True)
    print("  health   : GET /health", flush=True)
    print("  models   : GET /v1/models", flush=True)
    print("  chat     : POST /v1/chat/completions  model=switchyard", flush=True)
    print("Ctrl-C to stop.", flush=True)

    def _stop(*_a: object) -> None:
        print("\nStopping Switchyard…")
        try:
            server.close()
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while True:
        time.sleep(3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
