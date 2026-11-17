"""Run the non-actuating Quest transport host on a configured LAN interface."""

import argparse
import asyncio

from abstract_stack.teleop.transport import ControlServer


async def run(host, port):
    server = await ControlServer(host, port).start()
    print(f"Quest control: ws://{host}:{server.bound_port}/quest/control (actuation disabled)")
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.host, args.port))
    except KeyboardInterrupt:
        pass
