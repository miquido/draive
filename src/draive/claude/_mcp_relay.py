import os
import socket
import sys
import threading

_CHUNK = 64 * 1024
_ARGUMENTS = 2  # the script and the socket path


def _forward_input(connection: socket.socket) -> None:
    try:
        while data := os.read(0, _CHUNK):
            connection.sendall(data)

    except OSError:
        pass  # the other side is gone, the main thread exits

    try:
        connection.shutdown(socket.SHUT_WR)

    except OSError:
        pass


def main() -> int:
    if len(sys.argv) != _ARGUMENTS:
        sys.stderr.write("usage: _mcp_relay.py <socket path>\n")
        return 2

    for descriptor in (0, 1):
        try:
            os.set_blocking(descriptor, True)  # EAGAIN would end the copying or fail a write

        except OSError:
            pass

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(sys.argv[1])

    except OSError as exc:
        sys.stderr.write(f"draive tools server is not reachable: {exc}\n")
        return 1

    threading.Thread(target=_forward_input, args=(connection,), daemon=True).start()
    try:
        while data := connection.recv(_CHUNK):
            view = memoryview(data)
            while view:
                view = view[os.write(1, view) :]

    except OSError:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
