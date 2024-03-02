import socket
import time

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to connect to")
@click.option("--port", default=50051, help="Port to connect to")
def wait_data_server(host, port):
    """Wait for the data server to be ready"""

    while True:
        try:
            with socket.create_connection((host, port)):
                break
        except ConnectionRefusedError:
            click.echo("Server is not ready yet! Waiting...")
            time.sleep(1)

    click.echo("Server is ready! Starting training...")


if __name__ == "__main__":
    wait_data_server()
