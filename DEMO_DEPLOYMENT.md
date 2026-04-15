# FAIM demo – safe, restricted access

Your app now has an **optional password gate**: only people who enter the correct password can use it. Use this when you deploy a demo and want to control who can access it.

---

## Option 1: Hugging Face Spaces (Private) – only people you invite

**Best if:** You want real access control: only specific people can open the app.

1. Go to **https://huggingface.co** and sign in.
2. Click your profile (top right) → **New Space**.
3. **Name:** e.g. `faim-wildfire-demo`. **License:** None or your choice. **Select SDK:** **Streamlit**.
4. **Space hardware:** keep **CPU basic** (free).
5. **Visibility:** choose **Private**.
6. Click **Create Space**.
7. **Upload your files** in the Space:
   - Create a file `app.py` and paste the **entire** contents of `wildfire_forecast_app_V1_5_5.py`.
   - Create `requirements.txt` with the contents of your `requirements_file.txt` (and add: `scikit-learn` if not already there).
   - Upload the three files: `fuzzy_bayesian_regression.py`, `fuzzy_bayesian_regression_V2.py`, `fuzzy_bayesian_regression_V3.py` (e.g. drag into the Space file list).
8. In the Space **Settings** → **Collaborators**, add the email or username of each person who may access the demo. Only they (and you) can open the Space.

**Result:** Only you and invited collaborators can see and use the demo. Your code is in the Space (private); nobody else can see the repo unless you add them.

---

## Option 2: Streamlit Community Cloud – secret link + password

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
