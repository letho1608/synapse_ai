"""
Behavioral tests: Tailscale discovery without a Tailscale CLI.

- start() probes once: posture task is SKIPPED (not erroring every
  interval), with one clear line pointing at udp/manual for LAN.
- stop() is safe with update_task None.
- No log spam: the old per-interval error line never appears.
"""
import pytest

from synapse.networking.tailscale.tailscale_discovery import TailscaleDiscovery


def _disc():
    return TailscaleDiscovery("t1", 5678, lambda *a: None,
                              tailscale_api_key="", tailnet="")


@pytest.mark.asyncio
async def test_no_cli_skips_posture_task_once(capsys):
    d = _disc()
    await d.start()
    try:
        assert d.update_task is None
        assert d.discovery_task is not None
        out = capsys.readouterr().out
        assert "posture updates OFF" in out
        assert out.count("posture updates OFF") == 1
        assert "udp" in out and "manual" in out
        assert "Could not get Tailscale device ID" not in out
    finally:
        await d.stop()
    assert d.discovery_task is None and d.update_task is None


@pytest.mark.asyncio
async def test_stop_without_start_is_safe():
    d = _disc()
    await d.stop()
