"""Free TCP ports for tests that start real node processes.

Fixed ports broke whenever Windows reserved their range for Hyper-V/WSL
(51095-51294 after Docker Desktop started on 2026-11-04: "access forbidden").
Port 0 makes the OS pick one that is free and not reserved.
"""
import socket


def free_ports(n: int) -> list:
    socks, ports = [], []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            socks.append(s)
            ports.append(s.getsockname()[1])
    finally:
        for s in socks:
            s.close()
    return ports
