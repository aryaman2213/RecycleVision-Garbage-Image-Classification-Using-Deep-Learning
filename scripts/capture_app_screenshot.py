"""Screenshot the running Streamlit app classifying an example image -> reports/figures/app_screenshot.png

Start the app first:  streamlit run app/streamlit_app.py --server.port 8531
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8531"
EXAMPLE = sys.argv[2] if len(sys.argv) > 2 else "plastic"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
    page.goto(URL)
    page.get_by_text("try an example from the held-out test set").wait_for(timeout=60000)
    page.locator("[data-testid='stSelectbox']").click()
    page.get_by_role("option", name=EXAMPLE, exact=True).click()
    page.get_by_text("Top-3 predictions").wait_for(timeout=60000)
    page.get_by_text("Grad-CAM: red").wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(ROOT / "reports" / "figures" / "app_screenshot.png"), full_page=True)
    print(page.locator("[data-testid='stAlert']").first.inner_text())
    browser.close()
