#!/usr/bin/env python3
"""
Kenya CBC Pathway Recommendation System
AI-Powered Pathway Guidance using Deep Q-Learning
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kenya CBC Pathway Recommendation System"
    )
    parser.add_argument('--port', '-p', type=int, default=8050, help='Port (default: 8050)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host (default: 127.0.0.1)')
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_args()
    debug_mode = args.debug
    
    # Detect pytest or other test runners — disable reloader to avoid
    # FileNotFoundError from Dash's file watcher conflicting with pytest's
    # __pycache__ management.
    running_under_test = 'pytest' in sys.modules or 'unittest' in sys.argv[0:1]
    
    print(f"\n  🎓 Kenya CBC Pathway System")
    print(f"  ─────────────────────────────")
    print(f"  http://{args.host}:{args.port}\n")
    
    try:
        from pages.dashboard import create_app
        app = create_app()
        app.run(
            debug=debug_mode,
            port=args.port,
            host=args.host,
            use_reloader=not running_under_test,
        )
    except KeyboardInterrupt:
        print("\n  Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
