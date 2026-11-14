"""Start the HTTP server with ``python -m engine.torch.serve``."""

import uvicorn

from engine.torch.serve.server import app

if __name__ == "__main__":
    uvicorn.run(app, host=app.state.config.host, port=app.state.config.port)
