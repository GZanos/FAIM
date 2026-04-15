# Building a FAIM .dmg for Mac (local install)

This lets you package the app so someone can install it on a Mac and run it like an app (double‑click to start; it opens in the browser).

---

## What you’re creating

- A **.dmg** disk image that contains a **FAIM** folder.
- Inside the folder:
  - All app code (main folder only, no `Archived/`, no `__pycache__`, no `.venv`).
  - **Install FAIM.command** – run once to set up Python and dependencies.
  - **Run FAIM.command** – double‑click to start the app (browser opens).
- User flow: copy FAIM from the .dmg → run **Install FAIM.command** once → then use **Run FAIM.command** to open the app.

---

## Files to include in the .dmg (main folder only)

Include **only** these from the main V1.5 folder (no subfolders like `Archived/` or `__pycache__`):

| File | Purpose |
|------|--------|
| `wildfire_forecast_app_V1_5_5.py` | Main app |
| `fuzzy_bayesian_regression_V3.py` | FBLiR module |
| `fblir_integration.py` | FBLiR integration |
| `requirements.txt` | Dependencies |
| `Install FAIM.command` | One‑time setup |
| `Run FAIM.command` | Start the app |
| `README_FBLIR.md` | Optional docs |
| `QUICKSTART_FBLIR.py` | Optional |
| `setup_script.py` | Optional |
| `deployment_guide.md` | Optional |
| `DEMO_DEPLOYMENT.md` | Optional |
| `SUMMARY_FBLIR.md` | Optional |

Do **not** include: `Archived/`, `__pycache__/`, `.venv/`, `.git/`, `.DS_Store`, `.gitignore`.

---

## Step 1: Prepare the FAIM folder

1. Create a new folder named **FAIM** (e.g. on your Desktop).
2. Copy into it **only** the files listed above from your V1.5 **main** folder (no subfolders).
3. Ensure **Install FAIM.command** and **Run FAIM.command** are **executable**:
   ```bash
   chmod +x "/path/to/FAIM/Install FAIM.command"
   chmod +x "/path/to/FAIM/Run FAIM.command"
   ```

---

## Step 2: Create the .dmg (Disk Utility)

1. Open **Disk Utility** (Applications → Utilities).
2. **File** → **New Image** → **Image from Folder**.
3. Select your **FAIM** folder → **Open**.
4. Set:
   - **Save as:** e.g. `FAIM-Wildfire-App`
   - **Where:** Desktop (or wherever you want).
   - **Image format:** compressed (smaller file).
   - **Encryption:** none (unless you want password protection).
5. Click **Save**. You get `FAIM-Wildfire-App.dmg`.

---

## Step 2 (alternative): Create the .dmg from Terminal

If you use [create-dmg](https://github.com/create-dmg/create-dmg) (install with `brew install create-dmg`):

```bash
create-dmg \
  --volname "FAIM" \
  --window-pos 200 120 \
  --window-size 500 320 \
  --icon-size 100 \
  --app-drop-link 380 220 \
  FAIM-Wildfire-App.dmg \
  /path/to/FAIM
```

Replace `/path/to/FAIM` with the path to your FAIM folder.

---

## What the end user does

1. Double‑click **FAIM-Wildfire-App.dmg** to mount it.
2. Drag the **FAIM** folder to **Applications** (or Desktop).
3. Open the **FAIM** folder.
4. Double‑click **Install FAIM.command** once (Terminal opens and installs dependencies; may take a few minutes).
5. From then on, double‑click **Run FAIM.command** to start the app; the browser opens to the app.

They need **Python 3** installed (e.g. from python.org or `brew install python3`). The install script will say so if it’s missing.

---

## Optional: “FAIM.app” in Applications

To have an icon in Applications that starts the app:

1. Open **Automator** (Applications).
2. **File** → **New** → **Application**.
3. Add action **Run Shell Script**.
4. In the script box, paste (adjust path if FAIM is elsewhere):
   ```bash
   /bin/bash "/Applications/FAIM/Run FAIM.command"
   ```
5. **File** → **Save** → name it **FAIM** → save to **Applications**.
6. Optionally: **Get Info** on FAIM.app → drag an icon into the app icon to customize.

Then the user can start FAIM from the Applications folder or Spotlight.

---

## Summary

| You do | User does |
|--------|-----------|
| Put main‑folder files + `Install FAIM.command` + `Run FAIM.command` in a **FAIM** folder | Mount .dmg, copy FAIM to Applications |
| Create .dmg from that folder (Disk Utility or create-dmg) | Run **Install FAIM.command** once |
| Share **FAIM-Wildfire-App.dmg** | Use **Run FAIM.command** (or FAIM.app) to open the app in the browser |

The app runs **locally**; no code or data is sent to a server except any APIs the app already uses (e.g. NASA POWER).
