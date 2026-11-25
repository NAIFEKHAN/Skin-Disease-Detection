import sys
import os
import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8015'

def test(name):
    print(f'\n--- {name} ---')

def get(path):
    req = urllib.request.Request(f'{BASE_URL}{path}')
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read()

def get_json(path):
    status, data = get(path)
    return status, json.loads(data.decode('utf-8'))

test('1. System Health Check (GET /health)')
status, health = get_json('/health')
assert status == 200, f'Expected 200, got {status}'
assert health['status'] == 'healthy'
assert health['models']['unet_segmentation']['ready'] is True
assert health['models']['vit_classification']['ready'] is True
assert health['models']['attention_rollout']['ready'] is True
assert health['models']['common_skin_classification']['ready'] is False
print('✔ Health check PASSED: U-Net and ViT models active and healthy on CPU.')

test('2. Web Views (Home, Library, Reports, About)')
for route, title in [('/', 'Skin Analysis'), ('/library', 'Disease Library'), ('/reports', 'Analysis Reports'), ('/about', 'About the Project')]:
    status, html = get(route)
    html_text = html.decode('utf-8')
    assert status == 200, f'Route {route} returned {status}'
    assert title.lower() in html_text.lower(), f'Title {title} not found in {route}'
    print(f'✔ Route {route} returned HTTP 200 ({title}).')

test('3. Disease Library 16 Categories (GET /api/disease-library)')
status, lib = get_json('/api/disease-library')
assert status == 200
diseases = lib['diseases']
assert len(diseases) == 16, f'Expected 16 diseases, got {len(diseases)}'
ai_supported = [d for d in diseases if d.get('ai_supported') is True]
edu_only = [d for d in diseases if d.get('ai_supported') is False]
assert len(ai_supported) == 7, f'Expected 7 AI-supported classes, got {len(ai_supported)}'
assert len(edu_only) == 9, f'Expected 9 educational categories, got {len(edu_only)}'
print(f'✔ Disease Library PASSED: {len(ai_supported)} AI Active, {len(edu_only)} Educational Only.')

test('4. Full Inference Pipeline (POST /api/predict)')
test_img_path = Path('/home/ubuntu/skindisease/frontend/img/library/ISIC_0026586.jpg')
assert test_img_path.exists(), 'Test image missing'

# Build multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
with open(test_img_path, 'rb') as f:
    img_bytes = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="image"; filename="{test_img_path.name}"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    f'{BASE_URL}/api/predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
with urllib.request.urlopen(req) as resp:
    assert resp.status == 200, f'Prediction returned {resp.status}'
    pred_data = json.loads(resp.read().decode('utf-8'))

record_id = pred_data.get('record_id') or pred_data.get('id')
ref_id = pred_data.get('reference_id', f'REC_{record_id}')
print(f'✔ Inference PASSED! Record ID: {record_id} (Ref: {ref_id})')

disease_name = pred_data['prediction']['disease_name']
disease_code = pred_data['prediction']['disease_code']
conf_pct = pred_data['prediction']['confidence_percentage']
top_3 = pred_data['prediction']['top_3_predictions']
coverage = pred_data['segmentation']['relative_lesion_area_percentage']
attn_avail = pred_data['attention_map']['available']
sev_level = pred_data['severity']['level']
formatted_cost = pred_data['cost_estimation']['formatted_total_range']

print(f'  Prediction: {disease_name} ({disease_code}) - Confidence: {conf_pct}%')
print(f'  Segmentation Lesion Coverage: {coverage:.2f}%')
print(f'  Attention Heatmap Available: {attn_avail}')
print(f'  Severity Level: {sev_level}')
print(f'  Estimated Treatment Cost: {formatted_cost}')
assert len(top_3) == 3, 'Expected top 3 predictions for animated charts'
print('✔ Top 3 predictions structured for Chart.js animated visualizations.')

test('5. Database Persistence Verification')
status, rec = get_json(f'/api/results/{record_id}')
assert status == 200
assert rec['id'] == record_id
print(f'✔ SQLite persistence PASSED for record ID {record_id}.')

test('6. Clinical PDF Report Generation & Download (GET /api/report/{id})')
status, pdf_bytes = get(f'/api/report/{record_id}')
assert status == 200
assert pdf_bytes.startswith(b'%PDF'), 'Not a valid PDF file'
assert len(pdf_bytes) > 50000, 'PDF size too small'
print(f'✔ PDF download PASSED ({len(pdf_bytes):,} bytes, valid PDF header).')

test('7. Safe Record Deletion (DELETE /api/records/{id})')
del_req = urllib.request.Request(f'{BASE_URL}/api/records/{record_id}', method='DELETE')
with urllib.request.urlopen(del_req) as resp:
    assert resp.status == 200
    del_data = json.loads(resp.read().decode('utf-8'))
    print(f'✔ Record deletion PASSED: {del_data["message"]}')

print('\n======================================================================')
print('ALL DEPLOYMENT VERIFICATION TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)')
print('======================================================================')
