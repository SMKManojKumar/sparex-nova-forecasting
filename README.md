# SPREX NOVA — Vercel Deployment

## Deploy to Vercel

### Option 1 — Vercel CLI
```bash
npm i -g vercel
cd sprex_nova
vercel
```

### Option 2 — Vercel Dashboard
1. Push this folder to a GitHub / GitLab repo.
2. Go to https://vercel.com/new → Import your repo.
3. Framework: **Other** (auto-detected as Python).
4. Add environment variable:
   - `SECRET_KEY` → any long random string (e.g. `openssl rand -hex 32`)
5. Click **Deploy**.

## Local development
```bash
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000

## Notes
- Data is stored **in-memory**. Because Vercel is serverless, data resets between cold starts.
  For persistence add a database (Vercel Postgres, PlanetScale, Supabase, etc.) and replace
  the `_DB` dict in `app.py` with real DB calls.
- `SECRET_KEY` must be set as a Vercel environment variable for sessions to survive redeploys.
