import urllib.request
import json
import re

def test_pdf_endpoint():
    print("Testing /api/records...")
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/records")
    assert req.status == 200
    data = json.loads(req.read().decode())
    records = data["records"] if isinstance(data, dict) and "records" in data else data
    print(f"Found {len(records)} existing records in database.")
    assert len(records) > 0, "Expected at least one record in the database."
    
    latest = records[0]
    rec_id = latest["id"]
    ref_id = latest.get("reference_id", f"REC{rec_id}")
    print(f"Testing PDF download for record ID {rec_id} (Ref: {ref_id})...")
    
    pdf_req = urllib.request.urlopen(f"http://127.0.0.1:8000/api/report/{rec_id}")
    assert pdf_req.status == 200
    content_type = pdf_req.headers.get("Content-Type")
    content_disp = pdf_req.headers.get("Content-Disposition")
    print(f"Content-Type: {content_type}")
    print(f"Content-Disposition: {content_disp}")
    
    assert "application/pdf" in content_type, f"Expected application/pdf, got {content_type}"
    assert f"report_{ref_id}.pdf".lower() in content_disp.lower(), f"Unexpected Content-Disposition: {content_disp}"
    
    pdf_bytes = pdf_req.read()
    print(f"PDF size: {len(pdf_bytes)} bytes")
    assert len(pdf_bytes) > 20000, "PDF size is unexpectedly small"
    
    # 1. Verify standard PDF signature
    assert pdf_bytes.startswith(b"%PDF-"), "File is not a valid PDF document"
    
    # 2. Count pages via /Type /Page and /Count
    page_matches = re.findall(rb'/Type\s*/Page\b', pdf_bytes)
    pages_catalog = re.findall(rb'/Count\s+(\d+)', pdf_bytes)
    num_pages = len(page_matches)
    print(f"Detected page objects in PDF: {num_pages}, catalog count: {pages_catalog}")
    assert num_pages == 2, f"Expected exactly 2 pages, got {num_pages}"
    assert b'2' in pages_catalog, f"Expected catalog count 2, got {pages_catalog}"
    
    # 3. Check for template version
    assert b"Template v2.4" in pdf_bytes, "Expected Template v2.4 in PDF metadata/content"
    
    # 4. Verify SegoeUI font registration in PDF
    assert b"SegoeUI" in pdf_bytes, "Expected SegoeUI TrueType font in PDF stream"
    
    print("\n[SUCCESS] PDF endpoint, 2-page layout, and TrueType font verified successfully!")

if __name__ == "__main__":
    test_pdf_endpoint()
