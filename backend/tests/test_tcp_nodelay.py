"""
Test: TCP_NODELAY is set on TCP transport sockets.

Verifies the fix for H4: the README claims "TCP with Nagle disabled
for minimal latency" but TCP_NODELAY was not set on the TCP transport.
"""
import socket
import asyncio
import pytest


@pytest.mark.asyncio
async def test_tcp_server_sets_nodelay():
    """Verify TCPServer sets TCP_NODELAY on its listening socket."""
    from synapse.transport.tcp_transport import TCPServer

    server = TCPServer(host="127.0.0.1", port=0)
    port = await server.start()

    try:
        for sock in server._server.sockets:
            # TCP_NODELAY should be enabled (value == 1)
            nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            assert nodelay == 1, (
                f"TCP_NODELAY not set on server socket; got {nodelay}"
            )
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tcp_connection_handler_sets_nodelay():
    """Verify TCP_NODELAY is set on accepted connection sockets."""
    from synapse.transport.tcp_transport import TCPServer

    received_sock_nodelay = None

    async def capture_nodelay(reader, writer):
        nonlocal received_sock_nodelay
        sock = writer.get_extra_info("socket")
        if sock is not None:
            received_sock_nodelay = sock.getsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY
            )
        writer.close()

    server = TCPServer(host="127.0.0.1", port=0)
    # Monkey-patch the handler for this test
    original_handle = server._handle_connection
    server._handle_connection = capture_nodelay
    port = await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await asyncio.sleep(0.2)
        assert received_sock_nodelay == 1, (
            f"TCP_NODELAY not set on connection socket; got {received_sock_nodelay}"
        )
    finally:
        server._handle_connection = original_handle
        await server.stop()


@pytest.mark.asyncio
async def test_tcp_client_pool_sets_nodelay():
    """Verify TCPClientPool sets TCP_NODELAY on outbound connections."""
    from synapse.transport.tcp_transport import TCPClientPool, TCPServer

    received_nodelay = None

    async def check_conn(reader, writer):
        nonlocal received_nodelay
        sock = writer.get_extra_info("socket")
        if sock is not None:
            received_nodelay = sock.getsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY
            )
        # Keep connection open long enough for client to connect
        try:
            await asyncio.wait_for(reader.readexactly(4), timeout=2.0)
        except asyncio.IncompleteReadError:
            pass
        writer.close()

    server = TCPServer(host="127.0.0.1", port=0)
    server._handle_connection = check_conn
    port = await server.start()

    try:
        pool = TCPClientPool()
        result = await pool.send(f"127.0.0.1:{port}", b"test")
        assert result is True
        await asyncio.sleep(0.2)
        assert received_nodelay == 1, (
            f"TCP_NODELAY not set on client pool socket; got {received_nodelay}"
        )
    finally:
        await pool.close_all()
        await server.stop()
