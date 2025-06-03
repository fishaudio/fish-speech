import re
from threading import Lock

import pyrootutils
import uvicorn
from kui.asgi import (
    Depends,
    FactoryClass,
    HTTPException,
    HttpRoute,
    Kui,
    OpenAPI,
    Routes,
)
from kui.cors import CORSConfig
from kui.openapi.specification import Info
from kui.security import bearer_auth
from loguru import logger
from typing_extensions import Annotated

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.server.api_utils import MsgPackRequest, parse_args
from tools.server.exception_handler import ExceptionHandler
from tools.server.model_manager import ModelManager
from tools.server.views import routes


class API(ExceptionHandler):
    def __init__(self):
        self.args = parse_args()

        def api_auth(endpoint):
            async def verify(token: Annotated[str, Depends(bearer_auth)]):
                if token != self.args.api_key:
                    raise HTTPException(401, None, "Invalid token")
                return await endpoint()

            async def passthrough():
                return await endpoint()

            if self.args.api_key is not None:
                return verify
            else:
                return passthrough

        self.routes = Routes(
            routes,  # keep existing routes
            http_middlewares=[api_auth],  # apply api_auth middleware
        )

        # OpenAPIの設定
        self.openapi = OpenAPI(
            Info(
                {
                    "title": "Fish Speech API",
                    "version": "1.5.0",
                }
            ),
        ).routes

        # Initialize the app
        self.app = Kui(
            routes=self.routes + self.openapi[1:],  # Remove the default route
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config=CORSConfig(),
        )

        # Add the state variables
        self.app.state.lock = Lock()
        self.app.state.device = self.args.device
        self.app.state.max_text_length = self.args.max_text_length

        # Associate the app with the model manager
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        # Make the ModelManager available to the views
        app.state.model_manager = ModelManager(
            mode=self.args.mode,
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            asr_enabled=self.args.load_asr_model,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Startup done, listening server at http://{self.args.listen}")


# Each worker process created by Uvicorn has its own memory space,
# meaning that models and variables are not shared between processes.
# Therefore, any variables (like `llama_queue` or `decoder_model`)
# will not be shared across workers.

# Multi-threading for deep learning can cause issues, such as inconsistent
# outputs if multiple threads access the same buffers simultaneously.
# Instead, it's better to use multiprocessing or independent models per thread.

if __name__ == "__main__":
    api = API()

    # IPv6 address format is [xxxx:xxxx::xxxx]:port
    match = re.search(r"\[([^\]]+)\]:(\d+)$", api.args.listen)
    if match:
        host, port = match.groups()  # IPv6
    else:
        host, port = api.args.listen.split(":")  # IPv4

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=api.args.workers,
        log_level="info",
    )
