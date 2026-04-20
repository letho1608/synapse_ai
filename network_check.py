import socket
import psutil
import os
import sys

def get_ips():
    ips = []
    for interface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == socket.AF_INET:
                ips.append((interface, snic.address))
    return ips

def check_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return conn.laddr.ip, conn.pid
    return None

def main():
    print("=== Synapse AI Network Diagnostics ===")
    
    # 1. Check local IPs
    print("\n[1] Local IP Interfaces:")
    ips = get_ips()
    tailscale_ip = None
    for iface, ip in ips:
        print(f"  - {iface}: {ip}")
        if "tailscale" in iface.lower() or ip.startswith("100."):
            tailscale_ip = ip
            
    # 2. Check Port 50051
    print("\n[2] Port 50051 Status:")
    port_info = check_port(50051)
    if port_info:
        ip, pid = port_info
        print(f"  ✅ Port 50051 is LISTENING on {ip} (PID: {pid})")
        if ip == "127.0.0.1":
            print("  ⚠️  WARNING: Port is only listening on localhost. Remote nodes cannot connect!")
        elif ip == "0.0.0.0":
            print("  ✅ Port is listening on all interfaces.")
    else:
        print("  ❌ Port 50051 is NOT listening. Is main.py running?")

    # 3. Check Port 52415 (API)
    print("\n[3] Port 52415 (API) Status:")
    api_info = check_port(52415)
    if api_info:
        print(f"  ✅ Port 52415 is LISTENING on {api_info[0]}")
    else:
        print("  ❌ Port 52415 is NOT listening.")

    # 4. Tailscale Check
    print("\n[4] Tailscale Info:")
    if tailscale_ip:
        print(f"  ✅ Tailscale IP detected: {tailscale_ip}")
    else:
        print("  ❌ No Tailscale IP found in environment markers (100.x.x.x).")

    print("\n=== End of Diagnostics ===")

if __name__ == "__main__":
    main()
