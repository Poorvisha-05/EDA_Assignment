# Uploading to GitHub (step by step)

1. Create an account at https://github.com and verify your email.
2. Install Git: https://git-scm.com/downloads. Check with `git --version`.
3. One-time setup:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
4. On GitHub click **New repository** → name `breast-cancer-eda-mlp` → **Public** → do NOT tick "Add README" → Create.
5. In a terminal, go to the project folder and push:
   ```bash
   cd breast-cancer-eda-mlp
   git init
   git add .
   git commit -m "Initial commit: EDA, MLP model and report"
   git branch -M main
   git remote add origin https://github.com/<your-username>/breast-cancer-eda-mlp.git
   git push -u origin main
   ```
   (GitHub asks you to sign in; use a Personal Access Token as the password if prompted.)
6. Open the repository URL in a private/incognito window to confirm it is public.
7. Submit that URL in the Google Form: https://forms.gle/kT9DU5aERo2peTNh8
