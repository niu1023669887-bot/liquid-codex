# Liquid Codex

Cocktail recipe platform with 416 ingredients database.

## What is this

A system that makes AI-generated cocktail recipes actually work in real life. Not another toy that lets LLMs make up random recipes.

Core features:
- 416 ingredients database (pH, Brix, density, and other objective data)
- 20 physics-based validation rules (dilution, acidity balance, emulsion safety, etc.)
- AI recipe generation + physical verification

## Local Setup

```bash
git clone https://github.com/niu1023669887-bot/liquid-codex.git
cd liquid-codex

pip install -r requirements.txt

cd backend
uvicorn main:app --reload --port 8000

# Open another terminal
cd frontend
python -m http.server 3000
```

Open http://localhost:3000

## Environment Variables

Configure in `backend/.env`:

```
JWT_SECRET=your-secret-key-here
PASSWORD=your-admin-password
DEEPSEEK_API_KEY=your-deepseek-key
PERPLEXITY_API_KEY=your-perplexity-key
```

## Deployment

Backend: Railway  
Frontend: Vercel

```bash
railway up
vercel deploy
```

## Acknowledgments

- FastAPI - Backend framework
- DeepSeek - AI reasoning
- Perplexity - Ingredient data verification
- Railway & Vercel - Deployment platforms
- Dave Arnold "Liquid Intelligence" - Dilution thermodynamics theory
- Jim Meehan "Meehan's Bartender Manual" - Spatial proportion theory

## License

MIT License
