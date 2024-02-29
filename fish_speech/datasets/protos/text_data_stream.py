import struct

from .text_data_pb2 import TextData


def read_pb_stream(f):
    while True:
        buf = f.read(4)
        if len(buf) == 0:
            break
        size = struct.unpack("I", buf)[0]
        buf = f.read(size)
        text_data = TextData()
        text_data.ParseFromString(buf)
        yield text_data


def write_pb_stream(f, text_data):
    buf = text_data.SerializeToString()
    f.write(struct.pack("I", len(buf)))
    f.write(buf)


def pack_pb_stream(text_data):
    buf = text_data.SerializeToString()
    return struct.pack("I", len(buf)) + buf


def split_pb_stream(f):
    while True:
        head = f.read(4)
        if len(head) == 0:
            break
        size = struct.unpack("I", head)[0]
        buf = f.read(size)
        yield head + buf
