"""
Unit tests: NetworkTopology, PeerConfig, DeviceCapabilities.
Chay: pytest tests/test_network_config.py -v
"""
import sys
import os
import tempfile
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_network_topology_from_path():
    """NetworkTopology.from_path voi config hop le."""
    from synapse.networking.manual.network_topology_config import NetworkTopology
    config_path = os.path.join(ROOT, "synapse", "networking", "manual", "test_data", "test_config_single_node.json")
    topo = NetworkTopology.from_path(config_path)
    assert topo.peers is not None
    assert "node1" in topo.peers
    peer = topo.peers["node1"]
    assert peer.address == "localhost"
    assert peer.port == 50051
    assert peer.device_capabilities is not None
    assert peer.device_capabilities.model == "Unknown Model"
    print("  [OK] test_network_topology_from_path")


def test_network_topology_file_not_found():
    """NetworkTopology.from_path file khong ton tai -> FileNotFoundError."""
    import pytest
    from synapse.networking.manual.network_topology_config import NetworkTopology
    with pytest.raises(FileNotFoundError):
        NetworkTopology.from_path("/nonexistent/path/config.json")
    print("  [OK] test_network_topology_file_not_found")


def test_network_topology_invalid_json():
    """NetworkTopology.from_path JSON thieu truong bat buoc -> ValueError."""
    import pytest
    from synapse.networking.manual.network_topology_config import NetworkTopology
    invalid_path = os.path.join(ROOT, "synapse", "networking", "manual", "test_data", "invalid_config.json")
    if os.path.isfile(invalid_path):
        with pytest.raises((ValueError, Exception)):
            NetworkTopology.from_path(invalid_path)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"peers": "not a dict"}')
            path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                NetworkTopology.from_path(path)
        finally:
            os.unlink(path)
    print("  [OK] test_network_topology_invalid_json")


def test_device_capabilities():
    """DeviceCapabilities, DeviceFlops, UNKNOWN_DEVICE_CAPABILITIES."""
    from synapse.topology.device_capabilities import (
        DeviceCapabilities,
        DeviceFlops,
        UNKNOWN_DEVICE_CAPABILITIES,
    )
    flops = DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)
    assert flops.fp32 == 1.0
    cap = DeviceCapabilities(model="Test", chip="CPU", memory=8192, flops=flops)
    assert cap.memory == 8192
    d = cap.to_dict()
    assert "model" in d and d["model"] == "Test"
    assert UNKNOWN_DEVICE_CAPABILITIES.model == "Unknown Model"
    print("  [OK] test_device_capabilities")
