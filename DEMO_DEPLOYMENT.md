
## Streamlit Community Cloud – secret link + password

**Best if:** You want a public platform but control access by “link + password”.

1. Go to **https://share.streamlit.io** and sign in with **GitHub**.
2. Click **New app**.
3. **Repository:** select your GitHub username and the repo where you uploaded the app (e.g. `faim-wildfire-forecast`).
4. **Branch:** `main` (or the branch you use).
5. **Main file path:** `wildfire_forecast_app_V1_5_5.py` (must match the filename in the repo).
6. **App URL:** you can leave the default (e.g. `something-random.streamlit.app`) – **do not post this URL publicly**.
7. Click **Deploy**. Wait until the app builds and runs.
8. **Set the demo password:**
   - In the app dashboard, open your app → **Settings** (or **⋮** → **Settings**).
   - Go to **Secrets** and add:
     ```toml
     demo_password = "YourSecretPassword123"
     ```
   - Save. The app will restart; from now on, visitors must enter this password to use the app.
9. **Share access:** Send the app URL **only** to people you trust (e.g. by email). They open the link, enter the password you gave them, and can use the demo.

**Result:** The app is not listed anywhere public. Only people who have both the link and the password can use it. Your GitHub repo stays private, so nobody sees the code from Streamlit.

---

## Safety summary

| What you want              | What to do |
|----------------------------|------------|
| Only specific people       | Use **Hugging Face Private Space** and add them as collaborators. |
| Link + password, no listing| Use **Streamlit Community Cloud**, set `demo_password` in Secrets, share the URL only by email. |
| No one uses without password | The app now has a **password gate** when `demo_password` or `DEMO_PASSWORD` is set. |

**Local run:** If you don’t set `DEMO_PASSWORD` or Streamlit Secrets, the password gate is off and the app runs normally on your machine.

**Changing the password:** Update it in Streamlit Secrets or in the Hugging Face Space (e.g. in a secret or config), then redeploy/restart.
