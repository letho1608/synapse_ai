"""
A tensor port that cannot be reached is not retried on every decode step.
Found 2026-11-04 on the first real 2-machine LAN run: the tensor port was
ephemeral, so a firewall (ufw on the Ubuntu node) could not be opened for it,
and each send to a blocked port waits out the 5 s connect timeout before
falling back to the bridge.
"""
import asyncio
import socket
import time

from synapse.transport.tcp_transport import TCPClientPool


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_failed_connect_is_remembered_then_retried():
    async def run():
        pool = TCPClientPool(connect_timeout=1.0)
        addr = f"127.0.0.1:{_closed_port()}"
        assert await pool.send(addr, b"x") is False
        assert addr in pool._failed

        calls = []
        real = asyncio.open_connection

        async def counting(*a, **k):
            calls.append(a)
            return await real(*a, **k)

        asyncio.open_connection = counting
        try:
            assert await pool.send(addr, b"x") is False
            assert calls == []  # skipped while inside retry_after

            pool._failed[addr] = time.time() - pool.retry_after - 1
            assert await pool.send(addr, b"x") is False
            assert len(calls) == 1  # retried after the window
        finally:
            asyncio.open_connection = real

    asyncio.run(run())
