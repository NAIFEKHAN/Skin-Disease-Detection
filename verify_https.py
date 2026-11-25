import sys
import os
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Read rotated credentials securely from protected file
cred_path = Path("/home/ubuntu/skindisease/.evaluator_credentials")
assert cred_path.exists(), "Credentials file missing"

user = None
password = None
with open(cred_path, "r") as f:
    for line in f:
        if line.startswith("username:"):
            user = line.split(":", 1)[1].strip()
        elif line.startswith("password:"):
            password = line.split(":", 1)[1].strip()

assert user and password, "Could not parse credentials"

BASE_URL = "https://skin-detection.duckdns.org"

# Setup basic auth handler
auth_handler = urllib.request.HTTPBasicAuthHandler()
auth_handler.add_password(
    realm="Skin Disease Detection - Authorized Evaluator Access",
    uri=BASE_URL,
    user=user,
    passwd=password
)
opener = urllib.request.build_opener(auth_handler)

def test(name):
    print(f"\n--- {name} ---")

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with opener.open(req) as resp:
        return resp.status, resp.read()

def get_json(path):
    status, data = get(path)
    return status, json.loads(data.decode("utf-8"))

test("1. HTTPS Health Check (GET /health)")
status, health = get_json("/health")
assert status == 200
assert health["status"] == "healthy"
assert health["models"]["unet_segmentation"]["ready"] is True
assert health["models"]["vit_classification"]["ready"] is True
assert health["models"]["attention_rollout"]["ready"] is True
print("✔ HTTPS Health Check PASSED: U-Net & ViT active over trusted SSL.")

test("2. HTTPS Web Views (Home, Library, Reports, About)")
for route, title in [("/", "Skin Analysis"), ("/library", "Disease Library"), ("/reports", "Analysis Reports"), ("/about", "About the Project")]:
    status, html = get(route)
    html_text = html.decode("utf-8")
    assert status == 200
    assert title.lower() in html_text.lower()
    print(f"✔ Route {route} returned HTTP 200 over HTTPS ({title}).")

test("3. Disease Library 16 Categories over HTTPS (GET /api/disease-library)")
status, lib = get_json("/api/disease-library")
assert status == 200
diseases = lib["diseases"]
assert len(diseases) == 16
ai_supported = [d for d in diseases if d.get("ai_supported") is True]
edu_only = [d for d in diseases if d.get("ai_supported") is False]
assert len(ai_supported) == 7
assert len(edu_only) == 9
print(f"✔ Disease Library PASSED: {len(ai_supported)} AI Active, {len(edu_only)} Educational Only.")

test("4. HTTPS Image Upload & Inference Pipeline (POST /api/predict)")
test_img_path = Path("/home/ubuntu/skindisease/frontend/img/library/ISIC_0026586.jpg")
assert test_img_path.exists()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
with open(test_img_path, "rb") as f:
    img_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="{test_img_path.name}"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    f"{BASE_URL}/api/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
with opener.open(req) as resp:
    assert resp.status == 200
    pred_data = json.loads(resp.read().decode("utf-8"))

record_id = pred_data.get("record_id") or pred_data.get("id")
ref_id = pred_data.get("reference_id", f"REC_{record_id}")
disease_name = pred_data["prediction"]["disease_name"]
confidence = pred_data["prediction"]["confidence_percentage"]
coverage = pred_data["segmentation"]["relative_lesion_area_percentage"]
formatted_cost = pred_data["cost_estimation"]["formatted_total_range"]
print(f"✔ HTTPS Inference PASSED! Record ID: {record_id} (Ref: {ref_id})")
print(f"  Prediction: {disease_name} - Confidence: {confidence}%")
print(f"  Segmentation Lesion Coverage: {coverage:.2f}%")
print(f"  Treatment Cost: {formatted_cost}")

test("5. HTTPS PDF Report Download (GET /api/report/{id})")
status, pdf_bytes = get(f"/api/report/{record_id}")
assert status == 200
assert pdf_bytes.startswith(b"%PDF")
print(f"✔ HTTPS PDF Download PASSED ({len(pdf_bytes):,} bytes).")

test("6. HTTP to HTTPS Redirection")
http_req = urllib.request.Request("http://skin-detection.duckdns.org/")
try:
    # Disable auto-follow to check 301 status directly
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    no_redir_opener = urllib.request.build_opener(NoRedirect)
    no_redir_opener.open(http_req)
except urllib.error.HTTPError as e:
    assert e.code == 301, f"Expected HTTP 301 redirect, got {e.code}"
    loc = e.headers.get("Location")
    assert loc.startswith("https://skin-detection.duckdns.org"), f"Unexpected redirect target: {loc}"
    print(f"✔ HTTP redirect PASSED: {e.code} -> {loc}")

print("\n======================================================================")
print("ALL LIVE HTTPS ENDPOINTS & PIPELINES VERIFIED (100% OPERATIONAL)!")
print("======================================================================")
