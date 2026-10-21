"""``python -m dainamo [--host 127.0.0.1] [--port 7865] [--no-browser] [--password PW | --no-auth] [--new-key]``"""

import argparse
import os
import socket
import threading
import webbrowser

import uvicorn


def lan_address() -> str:
    """This machine's address on the local network (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET: only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def main():
    ap = argparse.ArgumentParser(prog="dainamo", description="Dainamo Trainer web UI")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to reach it from other machines (sign-in is then required)")
    ap.add_argument("--port", type=int, default=7865)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--password", default=os.environ.get("DAINAMO_PASSWORD"),
                    help="require this password (default with --host 0.0.0.0: a generated access key)")
    ap.add_argument("--no-auth", action="store_true",
                    help="don't ask for sign-in even when reachable from other machines")
    ap.add_argument("--new-key", action="store_true", help="replace the generated access key (signs every device out)")
    args = ap.parse_args()

    from .server import auth

    if args.new_key:
        auth.generated_key(new=True)
        print("Made a new access key; devices signed in with the old one must sign in again.")
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if "DAINAMO_ALLOWED_HOSTS" not in os.environ:
        os.environ["DAINAMO_ALLOWED_HOSTS"] = "127.0.0.1,localhost" if loopback else "*"
    local = f"http://127.0.0.1:{args.port}" if args.host in ("0.0.0.0", "::") else f"http://{args.host}:{args.port}"
    open_url = local
    if args.password is not None and args.password.strip():
        auth.enable(args.password.strip())
        print("Sign-in is on: use the password you set.")
    elif not loopback and not args.no_auth:
        key = auth.generated_key()
        auth.enable(key)
        open_url = f"{local}/?key={key}"
        remote = f"http://{lan_address() if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
        print(f"Sign-in is on. Access key: {key}\n"
              f"On other devices, open {remote}/?key={key} (it signs that browser in).", flush=True)
    elif not loopback:
        print("Warning: listening on the network without sign-in. Anyone who can reach this port can browse files "
              "and start jobs on this machine.")
    print(f"Dainamo Trainer → {local}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(open_url)).start()
    uvicorn.run("dainamo.server.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
