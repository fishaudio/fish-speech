from threading import Lock

import pyrootutils
import uvicorn
from kui.asgi import FactoryClass, HTTPException, HttpRoute, Kui, OpenAPI, Routes
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.server.api_utils import MsgPackRequest, parse_args
from tools.server.exception_handler import ExceptionHandler
from tools.server.model_manager import ModelManager
from tools.server.views import (
    ASRView,
    ChatView,
    HealthView,
    TTSView,
    VQGANDecodeView,
    VQGANEncodeView,
)


class API(ExceptionHandler):
    def __init__(self):
        self.args = parse_args()
        self.routes = [
            ("/v1/health", HealthView),
            ("/v1/vqgan/encode", VQGANEncodeView),
            ("/v1/vqgan/decode", VQGANDecodeView),
            ("/v1/asr", ASRView),
            ("/v1/tts", TTSView),
            ("/v1/chat", ChatView),
        ]
        self.routes = Routes([HttpRoute(path, view) for path, view in self.routes])

        self.openapi = OpenAPI(
            {
                "title": "Fish Speech API",
                "version": "1.5.0",
            },
        ).routes

        # Initialize the app
        self.app = Kui(
            routes=self.routes + self.openapi[1:],  # Remove the default route
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config={},
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
    host, port = api.args.listen.split(":")

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=api.args.workers,
        log_level="info",
    )
