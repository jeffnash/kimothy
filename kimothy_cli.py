#!/usr/bin/env python3
"""CLI entrypoint for Kimothy proxy server."""

import os
import sys
import argparse
import uvicorn
from typing import Optional

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Kimothy - OpenAI-compatible proxy for Kimi-like providers"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=int(os.getenv("PORT", "8928")),
        help="Port to listen on (default: 8928)"
    )
    
    parser.add_argument(
        "-H", "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "-b", "--upstream-base-url",
        default=os.getenv("UPSTREAM_BASE_URL"),
        help="Upstream base URL (e.g., https://api.openai.com/v1)"
    )
    
    parser.add_argument(
        "-k", "--upstream-api-key",
        default=os.getenv("UPSTREAM_API_KEY"),
        help="Upstream API key/Bearer token"
    )
    
    parser.add_argument(
        "--preset",
        choices=["chutes", "nahcrof"],
        help="Use preset configuration"
    )
    
    parser.add_argument(
        "--default-model",
        default=os.getenv("DEFAULT_MODEL"),
        help="Default model when not specified in request"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Set environment variables from CLI arguments
    if args.upstream_base_url:
        os.environ["UPSTREAM_BASE_URL"] = args.upstream_base_url
    
    if args.upstream_api_key:
        os.environ["UPSTREAM_API_KEY"] = args.upstream_api_key
    
    if args.default_model:
        os.environ["DEFAULT_MODEL"] = args.default_model
    
    if args.preset:
        os.environ["PRESET"] = args.preset
    
    # Import after setting environment
    from kimothy.config import get_settings
    from kimothy.main import app
    
    # Get settings to trigger validation
    try:
        settings = get_settings()
        print(f"Starting Kimothy proxy...")
        print(f"  Upstream: {settings.upstream_base_url}")
        print(f"  Model: {settings.default_model or '(not set)'}")
        print(f"  Preset: {args.preset or '(none)'}")
        print(f"  Listening on: http://{args.host}:{args.port}")
        print()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run the server
    uvicorn.run(
        "kimothy.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
        access_log=True
    )


if __name__ == "__main__":
    main()
