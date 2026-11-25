"""
Automated Browser Chart Verification Test.
Verifies Chart.js library loading, 3 animated charts rendering, tooltips/labels,
doughnut center text, and clean reset on new analysis.
"""

import sys
import json
import time
import asyncio
import tempfile
import subprocess
import urllib.request
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Error: websockets package is required.")
    sys.exit(1)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://127.0.0.1:8000"
DEBUG_PORT = 9228


class ChromeCDP:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=25 * 1024 * 1024)

    async def send(self, method, params=None):
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("id") == self.msg_id:
                return resp.get("result", {})

    async def eval(self, expr):
        res = await self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True
        })
        val = res.get("result", {}).get("value")
        exc = res.get("exceptionDetails")
        if exc:
            raise RuntimeError(f"JS Eval Error: {exc}")
        return val

    async def close(self):
        if self.ws:
            await self.ws.close()


async def run_tests():
    print("=" * 60)
    print("STARTING BROWSER CHART.JS VERIFICATION")
    print("=" * 60)

    user_data_dir = tempfile.mkdtemp(prefix="chrome_test_charts_")
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={user_data_dir}",
        "--disable-extensions",
        "--no-first-run",
        f"{BASE_URL}/"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await asyncio.sleep(2.0)

    try:
        targets_req = urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json")
        targets = json.loads(targets_req.read().decode())
        page_target = next(t for t in targets if t.get("type") == "page")
        ws_url = page_target["webSocketDebuggerUrl"]

        cdp = ChromeCDP(ws_url)
        await cdp.connect()

        # Wait for page readiness and Chart.js initialization
        for _ in range(25):
            ready = await cdp.eval("document.readyState === 'complete' && typeof window.Chart === 'function'")
            if ready:
                break
            await asyncio.sleep(0.3)

        # 1. Verify Chart.js is loaded
        chart_version = await cdp.eval("typeof Chart !== 'undefined' ? Chart.version : 'MISSING'")
        print(f"✔ Chart.js version loaded: {chart_version}")
        assert chart_version != "MISSING", "Chart.js failed to load on the page!"

        # 2. Check mock report rendering with real sample payload
        mock_payload = {
            "record_id": 16,
            "reference_id": "REF-TEST-CHART-01",
            "created_at": "2026-11-24 21:50:00",
            "prediction": {
                "disease_code": "DF",
                "predicted_disease_name": "Dermatofibroma",
                "confidence_percentage": "94.1%",
                "top_3_predictions": [
                    {"rank": 1, "class_code": "DF", "display_name": "Dermatofibroma", "percentage": 94.1},
                    {"rank": 2, "class_code": "AKIEC", "display_name": "Actinic Keratoses", "percentage": 3.7},
                    {"rank": 3, "class_code": "BKL", "display_name": "Benign Keratosis", "percentage": 0.9}
                ]
            },
            "segmentation": {
                "relative_lesion_area_percentage": 24.62
            },
            "severity": {
                "level": "Low Concern",
                "summary": "Low Priority - Characteristically Benign Pattern"
            },
            "cost_estimation": {
                "total_min": 4700,
                "total_max": 11200,
                "formatted_total_range": "₹4,700 – ₹11,200",
                "breakdown": [
                    {"category": "Doctor Consultation", "min_cost": 600, "max_cost": 1200, "formatted_range": "INR 600 – 1,200"},
                    {"category": "Diagnostic Tests", "min_cost": 800, "max_cost": 1800, "formatted_range": "INR 800 – 1,800"},
                    {"category": "Medication & Dressings", "min_cost": 300, "max_cost": 700, "formatted_range": "INR 300 – 700"},
                    {"category": "Procedure / Intervention", "min_cost": 2500, "max_cost": 6500, "formatted_range": "INR 2,500 – 6,500"},
                    {"category": "Follow-Up Consultation", "min_cost": 500, "max_cost": 1000, "formatted_range": "INR 500 – 1,000"}
                ]
            },
            "image_paths": {
                "original": "/static/img/library/ISIC_0029177.jpg",
                "mask": "/static/img/library/ISIC_0029177.jpg",
                "overlay": "/static/img/library/ISIC_0029177.jpg",
                "attention": "/static/img/library/ISIC_0029177.jpg"
            },
            "disease_info": {
                "name": "Dermatofibroma",
                "code": "DF",
                "short_description": "A common, harmless cutaneous nodule."
            }
        }

        eval_render = f"renderOnPageReport({json.dumps(mock_payload)});"
        await cdp.eval(eval_render)
        await asyncio.sleep(0.5)

        # 3. Verify Prediction Bar Chart Canvas
        pred_chart_info = await cdp.eval("""
            (() => {
                const canvas = document.getElementById('chart-predictions');
                const chart = typeof Chart !== 'undefined' ? Chart.getChart(canvas) : null;
                return {
                    found: !!canvas,
                    width: canvas ? canvas.clientWidth : 0,
                    height: canvas ? canvas.clientHeight : 0,
                    chartInstance: !!chart,
                    chartType: chart ? chart.config.type : null,
                    labels: chart ? chart.data.labels : []
                };
            })()
        """)
        print(f"✔ Prediction Bar Chart: found={pred_chart_info['found']}, type={pred_chart_info['chartType']}, size={pred_chart_info['width']}x{pred_chart_info['height']}")
        assert pred_chart_info["chartInstance"] is True, "Prediction bar chart instance not found!"
        assert pred_chart_info["chartType"] == "bar", f"Expected bar chart, got {pred_chart_info['chartType']}"
        assert len(pred_chart_info["labels"]) == 3, f"Expected 3 labels, got {pred_chart_info['labels']}"

        # 4. Verify Lesion Coverage Doughnut Chart Canvas & Center Text
        coverage_chart_info = await cdp.eval("""
            (() => {
                const canvas = document.getElementById('chart-lesion-coverage');
                const chart = typeof Chart !== 'undefined' ? Chart.getChart(canvas) : null;
                const centerVal = document.getElementById('doughnut-pct-val');
                const covLesion = document.getElementById('coverage-lesion-pct');
                const covSkin = document.getElementById('coverage-skin-pct');
                return {
                    found: !!canvas,
                    width: canvas ? canvas.clientWidth : 0,
                    height: canvas ? canvas.clientHeight : 0,
                    chartInstance: !!chart,
                    chartType: chart ? chart.config.type : null,
                    centerText: centerVal ? centerVal.textContent : null,
                    lesionText: covLesion ? covLesion.textContent : null,
                    skinText: covSkin ? covSkin.textContent : null
                };
            })()
        """)
        print(f"✔ Lesion Coverage Doughnut: found={coverage_chart_info['found']}, type={coverage_chart_info['chartType']}, centerText='{coverage_chart_info['centerText']}'")
        assert coverage_chart_info["chartInstance"] is True, "Lesion coverage chart instance not found!"
        assert coverage_chart_info["chartType"] == "doughnut", f"Expected doughnut, got {coverage_chart_info['chartType']}"
        assert coverage_chart_info["centerText"] == "24.62%", f"Expected 24.62%, got {coverage_chart_info['centerText']}"
        assert coverage_chart_info["skinText"] == "75.38%", f"Expected 75.38%, got {coverage_chart_info['skinText']}"

        # 5. Verify Treatment Cost Comparison Bar Chart Canvas & Total Summary Strip
        cost_chart_info = await cdp.eval("""
            (() => {
                const canvas = document.getElementById('chart-cost-comparison');
                const chart = typeof Chart !== 'undefined' ? Chart.getChart(canvas) : null;
                const totalMin = document.getElementById('cost-total-min');
                const totalMax = document.getElementById('cost-total-max');
                return {
                    found: !!canvas,
                    width: canvas ? canvas.clientWidth : 0,
                    height: canvas ? canvas.clientHeight : 0,
                    chartInstance: !!chart,
                    chartType: chart ? chart.config.type : null,
                    datasetsCount: chart ? chart.data.datasets.length : 0,
                    totalMin: totalMin ? totalMin.textContent : null,
                    totalMax: totalMax ? totalMax.textContent : null
                };
            })()
        """)
        print(f"✔ Cost Comparison Bar Chart: found={cost_chart_info['found']}, type={cost_chart_info['chartType']}, min='{cost_chart_info['totalMin']}', max='{cost_chart_info['totalMax']}'")
        assert cost_chart_info["chartInstance"] is True, "Cost comparison chart instance not found!"
        assert cost_chart_info["chartType"] == "bar", f"Expected bar chart, got {cost_chart_info['chartType']}"
        assert cost_chart_info["datasetsCount"] == 2, f"Expected 2 datasets (min/max), got {cost_chart_info['datasetsCount']}"
        assert "4,700" in cost_chart_info["totalMin"], f"Expected 4,700 in totalMin, got {cost_chart_info['totalMin']}"
        assert "11,200" in cost_chart_info["totalMax"], f"Expected 11,200 in totalMax, got {cost_chart_info['totalMax']}"

        # 6. Verify Bottom Action Buttons
        bottom_actions = await cdp.eval("""
            (() => {
                const dlBtn = document.getElementById('btn-download-pdf-bottom');
                const prBtn = document.getElementById('btn-print-report-bottom');
                const newBtn = document.getElementById('btn-new-analysis-bottom');
                return {
                    dlFound: !!dlBtn,
                    dlTag: dlBtn ? dlBtn.tagName.toLowerCase() : null,
                    prFound: !!prBtn,
                    newFound: !!newBtn
                };
            })()
        """)
        print(f"✔ Bottom Action Buttons: PDF download tag='{bottom_actions['dlTag']}', Print button={bottom_actions['prFound']}, New Analysis={bottom_actions['newFound']}")
        assert bottom_actions["dlFound"] is True, "Bottom download PDF button missing!"
        assert bottom_actions["dlTag"] == "button", f"Expected <button> tag, got {bottom_actions['dlTag']}"
        assert bottom_actions["prFound"] is True, "Bottom print report button missing!"
        assert bottom_actions["newFound"] is True, "Bottom new analysis button missing!"

        # 7. Verify Reset upon clicking Start New Analysis
        reset_test = await cdp.eval("""
            (() => {
                const newBtn = document.getElementById('btn-new-analysis-bottom');
                if (newBtn) newBtn.click();
                const reportSection = document.getElementById('report-section');
                const pChart = typeof Chart !== 'undefined' ? Chart.getChart(document.getElementById('chart-predictions')) : null;
                const cChart = typeof Chart !== 'undefined' ? Chart.getChart(document.getElementById('chart-lesion-coverage')) : null;
                const costChart = typeof Chart !== 'undefined' ? Chart.getChart(document.getElementById('chart-cost-comparison')) : null;
                return {
                    reportDisplay: reportSection ? reportSection.style.display : null,
                    predChartCleaned: !pChart,
                    covChartCleaned: !cChart,
                    costChartCleaned: !costChart
                };
            })()
        """)
        print(f"✔ Reset behavior: reportDisplay='{reset_test['reportDisplay']}', charts cleaned={reset_test['predChartCleaned'] and reset_test['covChartCleaned'] and reset_test['costChartCleaned']}")
        assert reset_test["reportDisplay"] == "none", "Report section was not hidden on new analysis!"
        assert reset_test["predChartCleaned"] is True, "Prediction chart was not destroyed!"
        assert reset_test["covChartCleaned"] is True, "Coverage chart was not destroyed!"
        assert reset_test["costChartCleaned"] is True, "Cost chart was not destroyed!"

        await cdp.close()

        print("\n" + "=" * 60)
        print("ALL BROWSER CHART.JS VERIFICATIONS PASSED (100% SUCCESS)")
        print("=" * 60)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(run_tests())
