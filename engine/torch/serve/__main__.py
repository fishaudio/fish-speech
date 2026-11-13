"""Allow ``python -m engine.torch.serve`` to start the HTTP server."""

from .server import run

if __name__ == "__main__":
    run()
