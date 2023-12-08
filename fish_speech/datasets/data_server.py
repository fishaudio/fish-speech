import asyncio
import random

import grpc
from loguru import logger

from fish_speech.datasets.protos.text_data_pb2 import SampleDataRequest, SampledData
from fish_speech.datasets.protos.text_data_pb2_grpc import (
    DataServiceServicer,
    DataServiceStub,
    add_DataServiceServicer_to_server,
)
from fish_speech.datasets.protos.text_data_stream import read_pb_stream


class DataService(DataServiceServicer):
    def __init__(
        self,
        files: list[str],
    ):
        super().__init__()

        self.files = files

        # Read all lines, and group by speaker
        self.groups = []
        self.weights = []

        count = 0
        for filename in self.files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    self.groups.append(text_data)
                    self.weights.append(len(text_data.sentences))
                    count += 1

                    if count % 10000 == 0:
                        logger.info(f"Read {count} groups of text data")
                        break

    def SampleData(self, request: SampleDataRequest, context):
        group = random.choices(self.groups, weights=self.weights, k=1)[0]
        k = min(request.num_samples, len(group.sentences))
        samples = random.choices(group.sentences, k=k)

        return SampledData(
            samples=samples,
        )


async def run():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = DataServiceStub(channel)
        import time

        from tqdm import tqdm

        start = time.time()
        for _ in tqdm(range(10000)):
            await stub.SampleData(SampleDataRequest(num_samples=50))

        print(
            f"Time taken: {time.time() - start}, {10000 / (time.time() - start)} samples/s"
        )


async def serve():
    server = grpc.aio.server()
    add_DataServiceServicer_to_server(
        DataService(files=["data/quantized-dataset-1205.protos"]), server
    )
    listen_addr = "127.0.0.1:50051"
    server.add_insecure_port(listen_addr)
    print(f"Starting server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    # asyncio.run(serve())
    # Launch 14 workers

    asyncio.run(run())
