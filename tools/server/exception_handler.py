import traceback
from http import HTTPStatus

from kui.asgi import HTTPException, JSONResponse


class ExceptionHandler:

    async def http_exception_handler(self, exc: HTTPException):
        return JSONResponse(
            dict(
                statusCode=exc.status_code,
                message=exc.content,
                error=HTTPStatus(exc.status_code).phrase,
            ),
            exc.status_code,
            exc.headers,
        )

    async def other_exception_handler(self, exc: Exception):
        traceback.print_exc()

        status = HTTPStatus.INTERNAL_SERVER_ERROR
        return JSONResponse(
            dict(statusCode=status, message=str(exc), error=status.phrase),
            status,
        )
