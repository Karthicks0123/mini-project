# 📚 Smart Study Notes Generator

An AI-powered academic and technical text summarizer. It condenses lengthy articles, research papers, and lecture notes into concise executive summaries, extracts key takeaway bullet points, and provides quantitative compression analytics.

---

## 🚀 Features

- **Abstractive Summarization:** Uses Hugging Face Transformer models (`sshleifer/distilbart-cnn-12-6` / BART) to synthesize key insights rather than just cutting sentences.
- **Key Takeaways & Flashcards:** Automatically breaks down the summary into structured study points.
- **Quantitative Analytics:** Real-time metrics on original word count, summary word count, words saved, and percentage compression.
- **Vercel Serverless Ready:** Lightweight API endpoint designed to run within Vercel's execution limits with ultra-fast cold starts.
- **Modern Responsive Web UI:** Styled with Tailwind CSS, supporting one-click sample loaders, detail-level presets, and instant clipboard copying.

---

## 📁 Project Structure

```text
OPEN AI PR/
├── api/
│   └── index.py            # Vercel Serverless Python handler (Hugging Face Inference API)
├── public/
│   └── index.html          # Interactive Web Frontend (Tailwind CSS)
├── evaluate_summaries.py   # Multi-domain evaluation suite (Bio, ML, History)
├── study_notes_generator.py # Local PyTorch/Transformers CLI module
├── main.py                 # Interactive terminal CLI entrypoint
├── requirements.txt        # Serverless deployment dependencies
├── vercel.json             # Vercel routing configuration
├── .env.example            # Environment variable template
└── .gitignore              # Files excluded from git
```

---

## 🌐 Deploying to Vercel (Step-by-Step)

### Step 1: Get a Free Hugging Face Token
1. Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. Log in or create a free account.
3. Click **Create new token**, choose **Read** permissions, and copy your token (e.g. `hf_...`).

---

### Step 2: Deploy via GitHub (Recommended)

1. Initialize Git and push your project to a new GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Vercel deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. Go to [vercel.com](https://vercel.com) and log in.
3. Click **Add New...** > **Project** and import your GitHub repository.
4. Expand **Environment Variables** and add:
   - **Key:** `HF_TOKEN`
   - **Value:** `your_hugging_face_token`
5. Click **Deploy**.

---

### Step 3: Deploy via Vercel CLI (Alternative)

If you have the Vercel CLI installed:

```bash
# Install Vercel CLI (if not already installed)
npm install -g vercel

# Deploy project
vercel
```

Follow the interactive prompts, then set your environment variable:
```bash
vercel env add HF_TOKEN
```
And deploy to production:
```bash
vercel --prod
```

---

## 💻 Running Locally

### Option A: Run the Local Web Server (No PyTorch download required)
You can run the web application locally on port 3000:
```bash
python api/index.py
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

*(Optional)* Set your Hugging Face token in PowerShell before running:
```powershell
$env:HF_TOKEN="hf_your_token_here"
python api/index.py
```

---

### Option B: Run Local Offline CLI (Using local PyTorch)
To run the interactive terminal prompt directly:
```bash
python main.py
```

### Option C: Run Multi-Domain Benchmark Suite
To test factual retention and coherence scoring across domains:
```bash
python evaluate_summaries.py
```
