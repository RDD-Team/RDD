import argparse
import json
import socket
from datetime import datetime, timezone
from ipaddress import IPv4Address


WARNING_TYPE = "road_debris_warning"


def send_warning(address):
    warning = {
        "type": WARNING_TYPE,
        "in_ego_lane": True,
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source": "demo",
    }
    message = json.dumps(warning)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(message.encode("utf-8"), address)
    print(f"[송신] {address[0]}:{address[1]}  {message}")


def receive_warnings(address):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(address)
        print(f"[수신 대기] {address[0]}:{address[1]}  종료: Ctrl+C", flush=True)
        while True:
            data, sender_address = receiver.recvfrom(65535)
            try:
                warning = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print("[건너뜀] JSON 형식이 아닌 메시지", flush=True)
                continue
            if not isinstance(warning, dict) or warning.get("type") != WARNING_TYPE:
                print("[건너뜀] 낙하물 경고 형식과 다른 메시지", flush=True)
                continue
            print(f"[수신] {sender_address[0]}:{sender_address[1]}  {json.dumps(warning)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="RDD 낙하물 경고 UDP 송수신 시험")
    parser.add_argument("mode", choices=("receive", "send"))
    parser.add_argument("ipv4", nargs="?", type=IPv4Address, default=IPv4Address("127.0.0.1"))
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port는 1~65535 범위에서 지정해 주세요.")

    address = (str(args.ipv4), args.port)
    try:
        if args.mode == "send":
            send_warning(address)
        else:
            receive_warnings(address)
    except KeyboardInterrupt:
        print("\n수신을 종료했어요.")
    except OSError as error:
        parser.exit(1, f"UDP 시험 중단: {error}\n")


if __name__ == "__main__":
    main()
