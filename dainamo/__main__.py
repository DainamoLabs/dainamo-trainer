"""``python -m dainamo [--host 127.0.0.1] [--port 7865] [--browser] [--password PW | --no-auth] [--new-key]``

Opens the port first, so the link answers (the browser waits) while the app loads, then starts the server and
prints the link to open. ``--browser`` opens it in the default browser too; ``--no-browser``
(the old default's switch) is still accepted and does nothing."""

import argparse
import os
import socket
import sys
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
    ap.add_argument("--browser", action="store_true", help="also open the link in your browser")
    ap.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)  # the old default; kept for scripts
    ap.add_argument("--password", default=os.environ.get("DAINAMO_PASSWORD"),
                    help="require this password (default with --host 0.0.0.0: a generated access key)")
    ap.add_argument("--no-auth", action="store_true",
                    help="don't ask for sign-in even when reachable from other machines")
    ap.add_argument("--new-key", action="store_true", help="replace the generated access key (signs every device out)")
    args = ap.parse_args()

    # listening before the app is imported: a browser that opens the link meanwhile waits for the page instead of
    # finding nothing there ("unreachable"). The app loads when the server starts (``Server.run``).
    config = uvicorn.Config("dainamo.server.app:app", host=args.host, port=args.port, log_level="warning")
    sock = config.bind_socket()  # exits with uvicorn's message when the port is taken
    sock.listen(config.backlog)

    from .server import auth

    if args.new_key:
        auth.generated_key(new=True)
        print("Made a new access key; devices signed in with the old one must sign in again.")
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    os.environ["DAINAMO_LISTEN"] = args.host  # media_download.on_network
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
    if not loopback:
        print("Warning: Add from URL (experimental) works from other machines too: anyone who can use Dainamo can "
              "make this machine download from any web address, including ones on your local network.", flush=True)
    print(f"Dainamo Trainer is starting. Open {open_url} in your browser (Ctrl+C stops it).", flush=True)
    if args.browser:
        webbrowser.open(open_url)
    server = uvicorn.Server(config)
    stopped = threading.Event()

    def announce():
        while not server.started and not stopped.wait(0.1):
            pass
        if server.started:
            print("Ready.", flush=True)

    threading.Thread(target=announce, daemon=True).start()
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:  # Ctrl+C, as ``uvicorn.run`` takes it
        pass
    finally:
        stopped.set()
    if not server.started:
        sys.exit(3)  # uvicorn's STARTUP_FAILURE


if __name__ == "__main__":
    main()
