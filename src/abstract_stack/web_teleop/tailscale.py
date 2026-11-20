"""Reach the teleop host over a Tailscale tailnet, with Tailscale logins as identity.

The host binds to loopback only and ``tailscale serve`` publishes it at
``https://<machine>.<tailnet>.ts.net`` with a real certificate (so browsers get
``wss://``). For tailnet requests, Serve adds ``Tailscale-User-Login`` and
``Tailscale-User-Name`` headers naming the signed-in user. The host trusts those
headers only on connections from loopback (where Serve connects from) and admits
only allowlisted logins; the node owner is allowed by default.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOGIN_HEADER = "tailscale-user-login"
NAME_HEADER = "tailscale-user-name"
LOOPBACK = frozenset({"127.0.0.1", "::1"})

_APP_PATHS = (
    Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale"),
    Path(r"C:\Program Files\Tailscale\tailscale.exe"),
)


class TailscaleError(RuntimeError):
    """Tailscale is missing, signed out, or not set up for HTTPS."""


@dataclass(frozen=True)
class Identity:
    login: str
    name: str

    def public(self) -> dict:
        return {"login": self.login, "name": self.name}


@dataclass(frozen=True)
class NodeInfo:
    """What the host needs from ``tailscale status --json``."""

    dns_name: str  # e.g. "studio-mac.tail1234.ts.net" (no trailing dot)
    owner_login: str
    https_enabled: bool

    @property
    def url(self) -> str:
        return f"https://{self.dns_name}/"

    @property
    def origin(self) -> str:
        return f"https://{self.dns_name}"


def tailscale_binary() -> str:
    found = shutil.which("tailscale")
    if found:
        return found
    for path in _APP_PATHS:
        if path.exists():
            return str(path)
    raise TailscaleError(
        "Tailscale is not installed: get it from https://tailscale.com/download, sign in, "
        "then run this again."
    )


def parse_status(status: dict) -> NodeInfo:
    """Node name, owner login and HTTPS readiness from ``tailscale status --json``."""
    state = status.get("BackendState")
    if state != "Running":
        raise TailscaleError(
            f"Tailscale is {state or 'not running'}: sign in (menu bar app, or `tailscale up`) "
            "and run this again."
        )
    self_node = status.get("Self") or {}
    dns_name = str(self_node.get("DNSName") or "").rstrip(".")
    users = status.get("User") or {}
    owner = users.get(str(self_node.get("UserID"))) or {}
    login = str(owner.get("LoginName") or "")
    if not dns_name or not login:
        raise TailscaleError("Tailscale status is missing this machine's name or owner.")
    https = bool(status.get("CertDomains"))
    return NodeInfo(dns_name=dns_name, owner_login=login.lower(), https_enabled=https)


def node_info(binary: str | None = None) -> NodeInfo:
    binary = binary or tailscale_binary()
    try:
        result = subprocess.run(
            [binary, "status", "--json"], capture_output=True, text=True, timeout=10, check=False
        )
        status = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise TailscaleError(f"could not read Tailscale status: {error}") from error
    info = parse_status(status)
    if not info.https_enabled:
        raise TailscaleError(
            "HTTPS certificates are off for this tailnet: enable MagicDNS and HTTPS in the "
            "Tailscale admin console (DNS page), then run this again."
        )
    return info


def serve_command(port: int, binary: str = "tailscale") -> list[str]:
    """Publish http://127.0.0.1:<port> at https://<machine>.<tailnet>.ts.net (tailnet only)."""
    return [binary, "serve", "--bg", f"http://127.0.0.1:{port}"]


def start_serve(port: int, binary: str | None = None) -> None:
    binary = binary or tailscale_binary()
    result = subprocess.run(
        serve_command(port, binary), capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode != 0:
        raise TailscaleError(f"tailscale serve failed: {(result.stderr or result.stdout).strip()}")


class TailscaleAccess:
    """Admit only allowlisted Tailscale logins arriving through ``tailscale serve``.

    Identity headers are trusted only from ``trusted_peers`` (loopback by default):
    the host listens on loopback, so remote clients reach it only through Serve,
    which sets these headers from the tailnet's authenticated identity. Logins are
    compared case-insensitively.
    """

    def __init__(self, allowed_logins, *, trusted_peers=LOOPBACK):
        self.allowed = frozenset(login.strip().lower() for login in allowed_logins if login)
        if not self.allowed:
            raise ValueError("at least one Tailscale login must be allowed")
        self.trusted_peers = frozenset(trusted_peers)

    def check(self, peer: str | None, headers) -> tuple[Identity | None, str | None]:
        """(identity, None) if admitted, else (None, reason for the person refused)."""
        if peer not in self.trusted_peers:
            return None, "Connect through the Tailscale address, not directly."
        login = (headers.get(LOGIN_HEADER) or "").strip()
        if not login:
            return None, "Not signed in through Tailscale. Open the https://….ts.net address."
        name = (headers.get(NAME_HEADER) or "").strip() or login
        if login.lower() not in self.allowed:
            return None, (
                f"{login} is not allowed to operate this robot. "
                f"Ask the host to add --allow-user {login}."
            )
        return Identity(login=login, name=name), None


def print_setup(info: NodeInfo, port: int, *, serving: bool) -> None:
    print(f"Web teleop (Tailscale): {info.url}", flush=True)
    print(f"Allowed: {info.owner_login} (owner) and any --allow-user logins.", flush=True)
    if not serving:
        print(
            "Publish it on your tailnet (once; it persists):\n"
            f"  {' '.join(serve_command(port))}\n"
            "or rerun with --tailscale-serve. Stop with: tailscale serve reset",
            file=sys.stdout,
            flush=True,
        )
