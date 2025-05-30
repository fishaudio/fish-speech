class SimpleFlaskApp:
    """Very small Flask-like application object."""

    def __init__(self):
        self.routes = {}

    def route(self, path, methods=None):
        methods = tuple(sorted(methods or ["GET"]))

        def decorator(func):
            self.routes[(path, methods)] = func
            return func

        return decorator


def create_app():
    """Return a new :class:`SimpleFlaskApp`."""
    return SimpleFlaskApp()


__all__ = ["create_app", "SimpleFlaskApp"]
