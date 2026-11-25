#!/usr/bin/env bash
# ==============================================================================
# DuckDNS Dynamic DNS Automation Setup Script
# Keeps your AWS EC2 Public IPv4 synchronized with DuckDNS domain name
# ==============================================================================

set -euo pipefail

echo "========================================================"
echo " DuckDNS Dynamic DNS Synchronization Setup"
echo "========================================================"

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: sudo bash duckdns_setup.sh <duckdns_subdomain> <duckdns_token>"
    echo "Example: sudo bash duckdns_setup.sh my-skin-project a1b2c3d4-e5f6-7890-abcd-1234567890ab"
    echo ""
    read -rp "Enter DuckDNS Subdomain (e.g. 'my-skin-project' for my-skin-project.duckdns.org): " DOMAIN
    read -rp "Enter DuckDNS Token (from duckdns.org dashboard): " TOKEN
else
    DOMAIN="$1"
    TOKEN="$2"
fi

# Clean inputs
DOMAIN=$(echo "$DOMAIN" | tr -d ' ' | sed 's/\.duckdns\.org//g')
TOKEN=$(echo "$TOKEN" | tr -d ' ')

if [ -z "$DOMAIN" ] || [ -z "$TOKEN" ]; then
    echo "Error: Subdomain and Token cannot be empty."
    exit 1
fi

DUCK_DIR="/opt/duckdns"
echo "[1/4] Creating DuckDNS directory at ${DUCK_DIR}..."
mkdir -p "${DUCK_DIR}"

DUCK_SCRIPT="${DUCK_DIR}/duck.sh"
echo "[2/4] Generating updater script at ${DUCK_SCRIPT}..."
cat <<EOF > "${DUCK_SCRIPT}"
#!/bin/bash
RESPONSE=\$(curl -s -k "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=")
echo "\$(date -u) - DuckDNS update for ${DOMAIN}.duckdns.org: \${RESPONSE}" >> ${DUCK_DIR}/duck.log
EOF

chmod 700 "${DUCK_SCRIPT}"

echo "[3/4] Testing DuckDNS update immediately..."
bash "${DUCK_SCRIPT}"
cat "${DUCK_DIR}/duck.log" | tail -n 1

echo "[4/4] Registering cron job (runs every 5 minutes)..."
CRON_JOB="*/5 * * * * ${DUCK_SCRIPT} >/dev/null 2>&1"
( crontab -l 2>/dev/null | grep -Fv "${DUCK_SCRIPT}" ; echo "${CRON_JOB}" ) | crontab -

echo "========================================================"
echo "✔ DuckDNS configuration completed successfully!"
echo "Domain: https://${DOMAIN}.duckdns.org"
echo "Log file: ${DUCK_DIR}/duck.log"
echo "Cron: Synchronizes every 5 minutes."
echo "========================================================"
