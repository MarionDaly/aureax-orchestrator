# Security policy

Aureax is a pre-1.0 prototype. Security fixes target the current main branch; older commits
and private fleet deployments do not receive automatic updates.

## Reporting privately

Report through the repository's Security tab → Report a vulnerability, which opens a private
GitHub advisory. If that route is unavailable to you, open an issue asking for a private channel,
without exploit details, credentials, fleet addresses, or logs.
Do not post live secrets in public issues or pull requests. Include the affected revision, the
prerequisites, and a synthetic reproduction once a private channel is open.

## Supported trust boundary

The dashboard system and clients permitted to reach its LAN listener are fully trusted.
The dashboard intentionally has no login, dashboard credential or client certificate requirement.
Local HTTP on loopback requires no certificate. HTTPS with the fleet CA protects the LAN transport.
Host and Origin validation blocks cross-site browser requests and DNS rebinding, but it is not
an authorization mechanism for local programs or LAN clients. Do not expose the listener to
untrusted clients. API responses, including key reveals, are marked `no-store`.

Node llama-server and MCP keep their separate TLS and bearer checks. RPC is plaintext and
unauthenticated. Its init container must install and verify the per-head host firewall rules for
every IP family the host has, before the worker starts. Preserve those rules while RPC runs; a
host firewall reset requires stopping workers and recreating their pods. The runtime containers run as root
with host networking, PID visibility and persistent mounts, so they do not provide strong
isolation from a compromised node.

SSH first contact is trust on first use, on a trusted provisioning network. A changed host key is
logged at error level with both SHA256 fingerprints, replaced in the operator's trust store, and
connected to once more, because nodes are wiped and re-imaged and provisioning has to stay
automated. Treat that error line as a signal: on a node you did not re-image, stop and
investigate. `aureax trust-host-key <node> --fingerprint SHA256:...` is available for verifying a
fingerprint out of band before any connection is made; it is optional.

Node provisioning fetches the K3s installer from `https://get.k3s.io` over TLS and pipes it to a
root shell. That script is trusted on TLS and on the vendor's release infrastructure alone: there
is no published checksum or signature for the script itself to pin against. The k3s binary the
installer then downloads is covered, because the installer verifies it against the sha256sum
published with that release. The agent version is pinned to the cluster server's.

Credentials and certificates live in private operator state, never in published artifacts. Keep
backups of old histories outside the public repository. Rotate any operational credential whose
old value was shared or exposed; removing it from Git cannot revoke it on a node.
