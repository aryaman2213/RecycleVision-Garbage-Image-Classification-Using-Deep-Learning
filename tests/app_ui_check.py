"""End-to-end UI check of the running Streamlit app (every control, every example, odd uploads).

Not collected by pytest (needs a running server):
    streamlit run app/streamlit_app.py --server.port 8531
    python tests/app_ui_check.py [http://localhost:8531]
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8531"
# A remote server (Streamlit Cloud: pass https://<app>.streamlit.app/~/+/) needs longer before a rerun starts.
SETTLE_MS = 400 if "localhost" in URL else 2500
EX = ROOT / "app" / "examples"
CLASSES = sorted(p.stem for p in EX.glob("*.jpg"))
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def make_uploads(d: Path) -> dict:
    """Files exercising every input path; value = expected outcome."""
    metal, organic, glass = (Image.open(EX / f"{c}.jpg") for c in ("metal", "organic", "glass"))
    files = {}
    p = d / "photo.jpg"; metal.save(p); files[p] = "metal"
    p = d / "transparent.png"; im = metal.convert("RGBA"); a = np.array(im); a[:20, :, 3] = 0
    Image.fromarray(a).save(p); files[p] = "metal"
    p = d / "image.webp"; organic.save(p); files[p] = "organic"
    p = d / "image.bmp"; glass.save(p); files[p] = "glass"
    # phone photo stored sideways with an EXIF "rotate 90" tag: must be shown and classified upright
    p = d / "sideways_phone.jpg"; ex = Image.Exif(); ex[0x0112] = 6
    metal.resize((300, 200)).rotate(90, expand=True).save(p, exif=ex); files[p] = "metal"  # stored 200x300
    p = d / "cmyk.jpg"; organic.convert("CMYK").save(p); files[p] = "organic"
    p = d / "gray.jpg"; organic.convert("L").save(p); files[p] = "organic"
    p = d / "corrupt.jpg"; p.write_bytes(b"\xff\xd8 this is not really a jpeg"); files[p] = "error"
    p = d / "bomb.png"; Image.new("L", (14000, 14000)).save(p, optimize=True); files[p] = "error"  # >178 MP
    p = d / "noise.png"; Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(p)
    files[p] = "any"
    return files


def no_exception(page, where):
    check(page.locator("[data-testid='stException']").count() == 0, f"no Python exception shown ({where})")


def wait_idle(page):
    """Wait until Streamlit has finished the rerun and no stale (previous-run) elements remain."""
    page.wait_for_timeout(SETTLE_MS)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]') &&"
        " !document.querySelector('[data-stale=\"true\"]')", timeout=90000)
    page.wait_for_timeout(300)


def wait_result(page):
    wait_idle(page)
    page.get_by_text("Top-3 predictions").first.wait_for(timeout=60000)


with sync_playwright() as pw, tempfile.TemporaryDirectory() as tmp:
    browser = pw.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.goto(URL, timeout=120000)
    page.get_by_text("try an example from the held-out test set").wait_for(timeout=120000)

    # --- sidebar facts come from best_model.json
    side = page.locator("[data-testid='stSidebar']").inner_text()
    check("96.5%" in side and "0.963" in side and "2,423" in side and "16,153" in side, "sidebar shows model facts")
    check("EfficientNetB0" in side, "sidebar names the deployed model")

    # --- every example in the dropdown
    for c in CLASSES:
        page.locator("[data-testid='stSelectbox']").click()
        page.get_by_role("option", name=c, exact=True).click()
        wait_result(page)
        page.wait_for_timeout(300)
        alert = page.locator("[data-testid='stAlert']").first.inner_text()
        bars = page.locator("[data-testid='stProgress']").count()
        cams = page.get_by_text("Grad-CAM: red").count()
        check(alert.lower().startswith(c) and bars == 3 and cams == 1, f"example '{c}': '{alert.splitlines()[0]}', {bars} bars, gradcam={cams}")
        if c == "battery":
            check(page.get_by_text("Hazardous waste").count() == 1, "battery shows hazardous-waste warning")
    no_exception(page, "examples")

    # --- Grad-CAM toggle off / on
    toggle = page.locator("[data-testid='stSidebar'] [data-testid='stCheckbox'] input, [data-testid='stSidebar'] input[role='switch']").first
    page.locator("[data-testid='stSidebar']").get_by_text("Show Grad-CAM explanation").click()
    wait_idle(page)
    check(page.get_by_text("Grad-CAM: red").count() == 0, "toggle off hides Grad-CAM")
    page.locator("[data-testid='stSidebar']").get_by_text("Show Grad-CAM explanation").click()
    wait_idle(page)
    check(page.get_by_text("Grad-CAM: red").count() == 1, "toggle on shows Grad-CAM again")

    # --- uploads
    for path, expected in make_uploads(Path(tmp)).items():
        page.locator("input[type='file']").set_input_files(str(path))
        page.wait_for_timeout(800)
        if expected == "error":
            wait_idle(page)
            page.get_by_text("could not be read as an image").wait_for(timeout=30000)
            check(page.get_by_text("Top-3 predictions").count() == 0, f"upload {path.name}: friendly error, no result")
        else:
            wait_result(page)
            page.wait_for_timeout(300)
            alert = page.locator("[data-testid='stAlert']").first.inner_text().splitlines()[0]
            ok = expected == "any" or expected in alert.lower()
            check(ok, f"upload {path.name}: '{alert}'")
            if path.name == "noise.png":
                check("Not sure" in alert, "random-noise image is flagged as unsure (low confidence)")
            if path.name == "sideways_phone.jpg":
                img = page.locator("[data-testid='stImage'] img").first
                w, h = img.evaluate("e => [e.naturalWidth, e.naturalHeight]")
                check(w > h, f"EXIF-rotated photo displayed upright, landscape as shot ({w}x{h})")
        check(page.get_by_text("try an example from the held-out test set").count() == 0, f"example picker hidden while a file is uploaded ({path.name})")
        no_exception(page, path.name)

    # --- removing the upload brings the example picker back
    page.locator("[data-testid='stFileUploader'] button[aria-label^='Remove']").first.click()
    page.get_by_text("try an example from the held-out test set").wait_for(timeout=30000)
    wait_idle(page)
    check(page.get_by_text("Top-3 predictions").count() == 0, "removing the upload clears the result and restores the picker")

    # --- model comparison tab
    page.get_by_role("tab", name="Model comparison").click()
    wait_idle(page)
    grid = page.locator("[data-testid='stDataFrame']")
    check(grid.count() == 1, "comparison table rendered")
    text = page.locator("[data-testid='stTabs']").inner_text()
    check("held-out test images" in text, "comparison tab explains the test set")
    imgs = page.locator("[data-testid='stTabs'] [data-testid='stImage']")
    check(page.get_by_text("Confusion matrix of the deployed model").count() == 1 and
          page.get_by_text("Test-set metrics per model").count() == 1, "comparison + confusion figures rendered")
    no_exception(page, "comparison tab")
    page.screenshot(path=str(Path(tmp) / "cmp.png"), full_page=True)

    # --- phone-width layout: no horizontal scroll
    phone = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
    phone.goto(URL, timeout=120000)
    phone.get_by_text("try an example from the held-out test set").wait_for(timeout=60000)
    phone.locator("[data-testid='stSelectbox']").click()
    phone.get_by_role("option", name="plastic", exact=True).click()
    wait_result(phone)
    overflow = phone.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check(overflow <= 1, f"phone layout has no horizontal overflow ({overflow}px)")

    real_errors = [e for e in console_errors if "favicon" not in e.lower()]
    check(not real_errors, f"no browser console errors ({real_errors[:3]})")
    browser.close()

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)
