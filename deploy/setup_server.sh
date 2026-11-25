#!/usr/bin/env bash
# ==============================================================================
# Automated Server Provisioning Script: CP-1 Skin Disease Detection
# Target OS: Ubuntu 22.04 LTS on AWS EC2 (t2.medium / t3.medium recommended)
# ==============================================================================

set -euo pipefail

# Require root/sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (use: sudo bash setup_server.sh)" >&2
    exit 1
fi

echo "========================================================================"
echo " Starting Ubuntu Server Deployment for Skin Disease Detection App"
echo "========================================================================"

APP_DIR="/var/www/skindisease"
APP_USER="ubuntu"
APP_GROUP="www-data"

# Prompt for DuckDNS Domain
echo ""
read -rp "Enter your DuckDNS domain (e.g. 'my-skin-ai.duckdns.org'): " DOMAIN_NAME
DOMAIN_NAME=$(echo "$DOMAIN_NAME" | tr -d ' ')

if [ -z "$DOMAIN_NAME" ]; then
    echo "Error: Domain name cannot be empty." >&2
    exit 1
fi

echo ""
echo "[Step 1/8] Updating package index and installing system prerequisites..."
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    certbot \
    python3-certbot-nginx \
    apache2-utils \
    curl \
    unzip \
    ufw

echo "[Step 2/8] Setting up application directories at ${APP_DIR}..."
mkdir -p "${APP_DIR}"
mkdir -p "${APP_DIR}/uploads"
mkdir -p "${APP_DIR}/reports"
mkdir -p "${APP_DIR}/data"
mkdir -p "/var/www/certbot"

# If script is run from an extracted deployment folder, copy files over
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "${PARENT_DIR}/main.py" ] && [ "$PARENT_DIR" != "$APP_DIR" ]; then
    echo "Copying application files from ${PARENT_DIR} to ${APP_DIR}..."
    cp -ru "${PARENT_DIR}/"* "${APP_DIR}/"
fi

echo "[Step 3/8] Creating Python virtual environment and installing dependencies..."
if [ ! -d "${APP_DIR}/venv" ]; then
    python3 -m venv "${APP_DIR}/venv"
fi

# Upgrade pip and install wheels
"${APP_DIR}/venv/bin/pip" install --upgrade pip setuptools wheel
"${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements-prod.txt"

echo "[Step 4/8] Initializing database and file permissions..."
# Ensure database and folders exist and are writable by both ubuntu and www-data
touch "${APP_DIR}/data/skin_disease.db"
chown -R "${APP_USER}:${APP_GROUP}" "${APP_DIR}"
chmod -R 775 "${APP_DIR}/uploads"
chmod -R 775 "${APP_DIR}/reports"
chmod -R 775 "${APP_DIR}/data"
chmod 664 "${APP_DIR}/data/skin_disease.db"

echo "[Step 5/8] Configuring Academic Access Control (.htpasswd)..."
HTPASSWD_FILE="/etc/nginx/.htpasswd"
if [ ! -f "$HTPASSWD_FILE" ]; then
    echo "Creating evaluator login credentials for protected endpoints (/api/predict, /reports, etc.)."
    read -rp "Enter username for academic evaluator [default: evaluator]: " AUTH_USER
    AUTH_USER=${AUTH_USER:-evaluator}
    htpasswd -c "$HTPASSWD_FILE" "$AUTH_USER"
    chmod 640 "$HTPASSWD_FILE"
    chown root:www-data "$HTPASSWD_FILE"
    echo "Evaluator credentials saved to ${HTPASSWD_FILE}."
else
    echo "Existing .htpasswd found at ${HTPASSWD_FILE}. Skipping generation."
fi

echo "[Step 6/8] Configuring systemd service (skindisease.service)..."
cp "${APP_DIR}/deploy/skindisease.service" /etc/systemd/system/skindisease.service
systemctl daemon-reload
systemctl enable skindisease.service
systemctl restart skindisease.service

echo "[Step 7/8] Configuring Nginx reverse proxy..."
NGINX_CONF="/etc/nginx/sites-available/skindisease"
cp "${APP_DIR}/deploy/nginx_skindisease.conf" "$NGINX_CONF"
# Replace domain placeholder with actual domain
sed -i "s/YOUR_SUBDOMAIN\.duckdns\.org/${DOMAIN_NAME}/g" "$NGINX_CONF"

# Enable site
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/skindisease
# Remove default site if present
rm -f /etc/nginx/sites-enabled/default

# Test Nginx syntax
nginx -t

echo "[Step 8/8] Configuring UFW Firewall..."
ufw --force enable
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw reload

echo "========================================================================"
echo "✔ Core server configuration completed!"
echo "Service status: $(systemctl is-active skindisease.service)"
echo ""
echo "NEXT STEP: Obtain free Let's Encrypt SSL certificate via Certbot:"
echo "Run the following command:"
echo "    sudo certbot --nginx -d ${DOMAIN_NAME}"
echo ""
echo "After Certbot finishes, restart Nginx:"
echo "    sudo systemctl restart nginx"
echo ""
echo "Your secure application will be accessible at: https://${DOMAIN_NAME}"
echo "========================================================================"
