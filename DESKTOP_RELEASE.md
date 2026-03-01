# Create Your First Desktop Release

The Windows download button will 404 until you create a release. Do this **once**:

## Option 1: One-click (recommended)

1. Go to **GitHub** → your repo → **Actions**
2. Click **"Build Desktop App (Windows, Mac, Linux)"**
3. Click **"Run workflow"** (dropdown on the right)
4. Leave version as `1.0.0` or change it
5. Click the green **"Run workflow"** button
6. Wait ~10–15 minutes for the build to finish
7. Go to **Releases** – you should see `desktop-v1.0.0` with the installer
8. The Streamlit download button will now work

## Option 2: Via git tag

```bash
git tag desktop-v1.0.0
git push origin desktop-v1.0.0
```

This triggers the same workflow and creates the release.

---

After the first release, the **Download for Windows** button will start the installer download directly.
