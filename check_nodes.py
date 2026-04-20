import asyncio
import time
import aiohttp

API_KEY = "tskey-api-kZkPWrVyM311CNTRL-91psFey7AHYgSzXpLr7GJYKZm43RZkXVD"
TAILNET = "testdoki925@gmail.com"
GRPC_PORT = 50051
TIMEOUT = 5

async def check_grpc_port(host: str, port: int) -> bool:
    """Ket noi thu TCP den cong gRPC de xac nhan Synapse dang chay."""
    try:
        conn = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(conn, timeout=TIMEOUT)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False

async def get_tailscale_devices():
    url = f"https://api.tailscale.com/api/v2/tailnet/{TAILNET}/devices"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("devices", [])

async def main():
    print("=" * 60)
    print("   SYNAPSE AI - KIEM TRA NODE (Dua tren gRPC thuc te)")
    print("   Tieu chi: Cong 50051 co mo khong? (bo qua Tailscale cached)")
    print("=" * 60)

    print("\n[*] Dang lay danh sach thiet bi tu Tailscale API...")
    try:
        devices = await get_tailscale_devices()
    except Exception as e:
        print(f"[ERROR] Khong the ket noi Tailscale API: {e}")
        return

    if not devices:
        print("[WARN] Khong tim thay thiet bi nao trong Tailnet.")
        return

    current_time = time.time()
    print(f"[OK] Tim thay {len(devices)} thiet bi. Dang kiem tra tat ca...\n")

    synapse_nodes = []
    non_synapse = []

    # Kiem tra tat ca thiet bi song song
    tasks = []
    device_info = []
    for dev in devices:
        name = dev.get("hostname") or dev.get("name", "Unknown")
        addresses = dev.get("addresses", [])
        ip = next((a for a in addresses if a.startswith("100.")), None)
        last_seen_str = dev.get("lastSeen", "")

        try:
            from datetime import datetime, timezone
            last_seen = datetime.strptime(last_seen_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            seconds_ago = int(current_time - last_seen.timestamp())
            if seconds_ago < 60:
                ts_label = f"Tailscale: ONLINE ({seconds_ago}s truoc)"
            elif seconds_ago < 3600:
                ts_label = f"Tailscale: offline ({seconds_ago // 60}m truoc)"
            else:
                ts_label = f"Tailscale: offline ({seconds_ago // 3600}h truoc)"
        except Exception:
            ts_label = "Tailscale: Khong ro"

        device_info.append({"name": name, "ip": ip, "ts_label": ts_label})
        if ip:
            tasks.append(check_grpc_port(ip, GRPC_PORT))
        else:
            tasks.append(asyncio.sleep(0))  # placeholder

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, dev in enumerate(device_info):
        ip = dev["ip"]
        name = dev["name"]
        ts_label = dev["ts_label"]

        if not ip:
            print(f"  [SKIP] {name} - Khong co IP Tailscale")
            continue

        port_open = results[i] is True

        if port_open:
            synapse_nodes.append({"name": name, "ip": ip, "ts_label": ts_label})
            print(f"  [SYNAPSE NODE] {name}")
            print(f"     IP        : {ip}:{GRPC_PORT}")
            print(f"     {ts_label}")
            print(f"     gRPC      : PHAN HOI - Day la Synapse Node that su!")
        else:
            non_synapse.append(name)
            print(f"  [KHONG PHAI NODE] {name}")
            print(f"     IP        : {ip}:{GRPC_PORT}")
            print(f"     {ts_label}")
            print(f"     gRPC      : KHONG PHAN HOI (Synapse chua chay)")
        print()

    print("=" * 60)
    print(f"KET QUA TONG HOP:")
    print(f"  - Synapse Node that su (gRPC OK) : {len(synapse_nodes)}")
    print(f"  - May khong chay Synapse          : {len(non_synapse)}")

    if synapse_nodes:
        print(f"\nCac Synapse Node dang hoat dong:")
        for n in synapse_nodes:
            print(f"  + {n['name']} ({n['ip']})")
    else:
        print("\n[CANH BAO] Khong co Synapse Node nao dang phan hoi!")
        print("  --> Nhung may hien tren Dashboard co the la du lieu cu (cached).")
        print("  --> Hay chay 'python main.py' de khoi dong Node.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
