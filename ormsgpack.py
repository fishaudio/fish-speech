import json

OPT_SERIALIZE_PYDANTIC = 0


def packb(obj, option=None):
    """Serialize an object to bytes using JSON."""
    try:
        # Support pydantic models
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
    except Exception:
        pass
    return json.dumps(obj).encode()


def unpackb(data, option=None):
    """Deserialize bytes produced by :func:`packb`."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode()
    return json.loads(data)
