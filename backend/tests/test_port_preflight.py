"""
Behavioral tests: port preflight (bind-conflict fail-fast).

- Free ports report no conflicts.
- An occupied port reports {name, port, holder} and the fix-it message
  names netstat/taskkill plus the --node-port/--chatgpt-api-port escape hatch.
- Invalid entries never raise.
"""
import socket

from synapse.helpers import check_ports_free, format_port_conflicts


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_free_ports_no_conflicts():
    assert check_ports_free({"p2p": _free_port(), "api": _free_port()}) == []
    assert check_ports_free({}) == []
    assert check_ports_free(None) == []
    assert check_ports_free({"x": "not-a-port"}) == []


def test_occupied_port_reports_holder_and_fix():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("0.0.0.0", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        conflicts = check_ports_free({"p2p": port})
        assert len(conflicts) == 1
        assert conflicts[0]["name"] == "p2p"
        assert conflicts[0]["port"] == port
        msg = format_port_conflicts(conflicts)
        assert str(port) in msg
        assert "taskkill" in msg
        assert "--node-port" in msg
    finally:
        holder.close()
    assert check_ports_free({"p2p": port}) == []
