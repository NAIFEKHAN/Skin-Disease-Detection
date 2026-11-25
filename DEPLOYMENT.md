# Production Deployment Runbook: AWS EC2 with DuckDNS & HTTPS

## AI-Based Skin Disease Detection with Treatment Cost Estimation

This guide details the complete, step-by-step procedure for deploying the **CP-1 Skin Disease Detection & Cost Estimation** monolithic web application to an **AWS EC2** virtual server running **Ubuntu 22.04 LTS** with a free **DuckDNS** domain, **Nginx** reverse proxy, **Let's Encrypt SSL (HTTPS)**, and **HTTP Basic Authentication** for clinical evaluation.

---

## 1. Production Architecture Overview

```
[ User / Evaluator Web Browser ]
               │
               ▼  HTTPS (Port 443) / HTTP (Port 80 -> 443 Redirect)
    [ DuckDNS Dynamic DNS: https://<your-subdomain>.duckdns.org ]
               │
               ▼  AWS Internet Gateway
  [ AWS EC2 Security Group: Inbound Ports 22 (SSH), 80 (HTTP), 443 (HTTPS) ]
               │
               ▼
   [ Nginx Web Server / Reverse Proxy (Port 80/443) ]
       ├── Static File Serving & Caching (/static/ -> /frontend/)
       ├── SSL Termination (Certbot / Let's Encrypt TLSv1.2 & TLSv1.3)
       ├── HTTP Basic Authentication (/etc/nginx/.htpasswd for /api/predict, /reports, etc.)
       └── Reverse Proxy Pass (http://127.0.0.1:8000)
               │
               ▼
    [ Uvicorn ASGI Server (127.0.0.1:8000, 2 CPU Workers) ]
       └── Supervised by systemd (`skindisease.service`)
               │
               ▼
     [ FastAPI Monolithic Application (`main.py`) ]
       ├── PyTorch ResNet18-UNet (Lesion Segmentation on CPU)
       ├── PyTorch Vision Transformer ViT-B/16 (7-Class Dermoscopic Classifier on CPU)
       ├── XAI Attention Rollout Generator
       ├── Severity & Cost Estimation Engines
       ├── ReportLab PDF Generator
       └── SQLite Embedded Database (`data/skin_disease.db`)
```

---

## 2. Recommended AWS EC2 Instance Sizing

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Instance Type** | `t2.small` (1 vCPU, 2 GB RAM) + 2 GB Swap | **`t2.medium` or `t3.medium`** (2 vCPU, 4 GB RAM) |
| **Operating System**| Ubuntu Server 22.04 LTS (HVM), SSD Volume Type | **Ubuntu Server 22.04 LTS (x86_64)** |
| **Storage (EBS)** | 20 GB gp3 | **25 – 30 GB gp3** |
| **Pricing Tier** | AWS Free Tier Eligible (t2.micro requires swap) | Cost-effective academic standard (~$0.04/hr) |

> [!NOTE]
> The application uses **CPU-optimized PyTorch** (`requirements-prod.txt`). No expensive GPU instance is required. A `t2.medium` or `t3.medium` instance executes full deep learning inference (U-Net lesion segmentation + ViT-B/16 classification + XAI attention rollout) in **under 1.5 seconds per image**.

---

## 3. Step-by-Step Deployment Walkthrough

### Step 1: Launch EC2 Instance & Configure Security Group

1. Log into the [AWS Management Console](https://aws.amazon.com/console/).
2. Navigate to **EC2** → **Launch Instance**.
3. Configure the instance:
   - **Name**: `Skin-Disease-Detection-Server`
   - **AMI**: `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type` (64-bit x86)
   - **Instance type**: `t3.medium` (or `t2.medium`)
   - **Key pair**: Create or select an existing `.pem` key pair (e.g., `skindisease-key.pem`)
4. In **Network Settings**, configure **Inbound Security Group Rules**:
   - **Rule 1 (SSH)**: Type `SSH`, Port `22`, Source `My IP` (or `0.0.0.0/0` for remote testing).
   - **Rule 2 (HTTP)**: Type `HTTP`, Port `80`, Source `0.0.0.0/0` (Anywhere IPv4).
   - **Rule 3 (HTTPS)**: Type `HTTPS`, Port `443`, Source `0.0.0.0/0` (Anywhere IPv4).
5. In **Configure Storage**:
   - Change root volume size to **`25 GiB`** (General Purpose SSD `gp3`).
6. Click **Launch Instance**. Note the **Public IPv4 Address** of your instance (e.g., `54.210.88.12`).

---

### Step 2: Register a Free DuckDNS Domain

1. Go to [DuckDNS.org](https://www.duckdns.org/) and sign in with Google or GitHub.
2. Under **domains**, choose a subdomain name for your project (e.g., `skin-ai-evaluator`).
3. In the **current ip** box, enter your EC2 instance's **Public IPv4 Address**.
4. Click **update ip**.
5. Copy your **DuckDNS Token** displayed at the top of the DuckDNS dashboard.
   - Example domain: `skin-ai-evaluator.duckdns.org`
   - Example token: `a1b2c3d4-e5f6-7890-abcd-1234567890ab`

---

### Step 3: Transfer the Deployment Package to EC2

Open a terminal (PowerShell, Command Prompt, or bash) on your local computer where `ec2_deployment_package.zip` is stored:

```bash
# Set secure permissions on your private key (Linux/macOS)
chmod 400 skindisease-key.pem

# Securely copy (SCP) the deployment archive to the Ubuntu home directory
scp -i path/to/skindisease-key.pem ec2_deployment_package.zip ubuntu@<YOUR_EC2_PUBLIC_IP>:~/
```

---

### Step 4: SSH into EC2 & Run the Automated Server Setup

Connect to your EC2 instance:

```bash
ssh -i path/to/skindisease-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Once connected, run the following commands to extract the package and launch the automated provisioning script:

```bash
# 1. Install unzip if not present
sudo apt-get update -y && sudo apt-get install -y unzip

# 2. Create the destination directory and extract the archive
sudo mkdir -p /var/www/skindisease
sudo unzip -q ~/ec2_deployment_package.zip -d /var/www/skindisease
sudo chown -R ubuntu:www-data /var/www/skindisease

# 3. Execute the automated setup script
cd /var/www/skindisease/deploy
sudo bash setup_server.sh
```

During execution, `setup_server.sh` will automatically:
1. Update system packages and install Python 3.11/venv, Nginx, Certbot, and Apache2-utils.
2. Create `/var/www/skindisease/venv` and install `requirements-prod.txt` using the fast CPU-only PyTorch index.
3. Initialize writable directories (`uploads/`, `reports/`, `data/`) with proper `ubuntu:www-data` permissions.
4. Prompt you to enter your **DuckDNS domain** (e.g., `skin-ai-evaluator.duckdns.org`).
5. Prompt you to set a password for the **Academic Evaluator** account (`evaluator`).
6. Register and start the `skindisease.service` systemd daemon.
7. Configure the Nginx reverse proxy and enable the UFW firewall (Ports 22, 80, 443).

---

### Step 5: Configure Automatic DuckDNS IP Synchronization

To ensure that your domain always points to the correct EC2 IP even after reboot or instance stop/start:

```bash
sudo bash /var/www/skindisease/deploy/duckdns_setup.sh <YOUR_SUBDOMAIN> <YOUR_DUCKDNS_TOKEN>
```

This installs a system cron job running every 5 minutes at `/opt/duckdns/duck.sh`.

---

### Step 6: Acquire Let's Encrypt SSL Certificate (HTTPS)

Execute Certbot to automatically configure TLS/SSL on Nginx:

```bash
sudo certbot --nginx -d <YOUR_SUBDOMAIN>.duckdns.org
```

- When prompted, enter your email for renewal notices.
- Agree to the Terms of Service.
- Certbot will automatically verify domain ownership, retrieve the certificate, and update `/etc/nginx/sites-available/skindisease`.

Finally, reload Nginx:

```bash
sudo systemctl restart nginx
```

---

## 4. Verification & Health Checks

### Check System Services

```bash
# 1. Verify Uvicorn FastAPI daemon status
sudo systemctl status skindisease.service

# 2. Verify Nginx web server status
sudo systemctl status nginx

# 3. Test local FastAPI health endpoint directly
curl -s http://127.0.0.1:8000/api/health | jq .
```

### Accessing the Web Application

Open your browser and navigate to:
**`https://<YOUR_SUBDOMAIN>.duckdns.org`**

- **Public Views**:
  - `https://<YOUR_SUBDOMAIN>.duckdns.org/` — Skin Analysis Home, Live Webcam capture, Upload, XAI Attention Rollout, animated charts.
  - `https://<YOUR_SUBDOMAIN>.duckdns.org/library` — Disease Library with 16 conditions, displaying **AI Analysis Active** (green) and **Educational Reference Only** (amber) badges.
  - `https://<YOUR_SUBDOMAIN>.duckdns.org/about` — Project documentation, model performance metrics, architecture, and team info.

- **Protected Evaluator Endpoints**:
  - `https://<YOUR_SUBDOMAIN>.duckdns.org/reports` — Analysis Reports Management, interactive search, and PDF download.
  - Triggering an AI prediction or downloading a medical report will prompt for HTTP Basic Auth:
    - **Username**: `evaluator`
    - **Password**: *(The password configured during Step 4)*

---

## 5. Maintenance & Troubleshooting Commands

| Task | Command |
| :--- | :--- |
| **View Real-Time Application Logs** | `sudo journalctl -u skindisease.service -f` |
| **Restart Application** | `sudo systemctl restart skindisease.service` |
| **View Nginx Error Logs** | `sudo tail -f /var/log/nginx/error.log` |
| **Test Nginx Configuration Syntax** | `sudo nginx -t` |
| **Restart Nginx** | `sudo systemctl restart nginx` |
| **Add / Update Evaluator Credentials** | `sudo htpasswd /etc/nginx/.htpasswd <username>` |
| **Check DuckDNS Update Log** | `cat /opt/duckdns/duck.log | tail -n 10` |
| **Simulate SSL Certificate Renewal** | `sudo certbot renew --dry-run` |

---

## 6. Security & Cost Optimization Best Practices

1. **EBS Cost Control**: AWS Free Tier includes 30 GB of EBS gp3 storage. The 25 GB allocation fits completely within the free allocation.
2. **Instance Stop When Not In Use**: When not presenting or evaluating, you can stop the instance from the AWS EC2 console to incur zero compute charges. When restarted, DuckDNS will automatically update your public IP within 5 minutes.
3. **Data Protection**: Uploaded dermatological patient images and clinical reports are stored locally in `/var/www/skindisease/uploads/` and `/var/www/skindisease/reports/`, protected by Nginx Basic Auth and isolated from public web crawlers.
