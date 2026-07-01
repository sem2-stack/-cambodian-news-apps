# Cambodian News Apps

A machine learning application for Cambodian news processing using transformer-based models.

## Project Structure

```
├── html/                 # Frontend (web UI)
│   ├── index.html       # Main HTML page
│   ├── app.js           # JavaScript frontend logic
│   └── styles.css       # Styling
├── inference/           # Model inference pipeline
│   └── predictor.py     # Prediction logic
├── models/              # Pre-trained transformer models (Git LFS)
│   └── undersampling_no_environment/
│       ├── bert_best.pt
│       ├── distilbert_best.pt
│       ├── electra_best.pt
│       └── roberta_best.pt
├── server.py            # Backend server
├── streamlit_app.py     # Streamlit web application
└── __init__.py          # Package initialization

```

## Models

This project uses Git LFS (Large File Storage) to manage large model files. The models are pre-trained transformer models:

- **bert_best.pt** - BERT model
- **distilbert_best.pt** - DistilBERT model (lightweight)
- **electra_best.pt** - ELECTRA model
- **roberta_best.pt** - RoBERTa model

### Setup for Model Files

**First time cloning?** Install Git LFS:

```bash
# Install Git LFS
git lfs install

# Clone the repository
git clone https://github.com/sem2-stack/-cambodian-news-apps.git

# Pull model files
git lfs pull
```

**Already cloned without Git LFS?**

```bash
git lfs install
git lfs pull
```

## Deployment

### Option 1: Hugging Face Spaces ⭐ (Recommended)

Best for ML/PyTorch apps with native model support.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
   - Name: `cambodian-news-classifier`
   - Space type: **Streamlit**
   - Visibility: Public/Private

3. Link your repository:
   - Go to Space Settings → Repository → "Link repository"
   - Select: `sem2-stack/-cambodian-news-apps`
   - Branch: `main`
   - Run command: `streamlit run streamlit_app.py`

4. Done! Your app will be live at `huggingface.co/spaces/<username>/<space-name>`

**Why HF Spaces?**
- ✅ 50GB+ storage (models included)
- ✅ PyTorch & transformers pre-optimized
- ✅ Git LFS support built-in
- ✅ Free tier with no time limits
- ✅ Persistent storage for model caching

### Option 2: Streamlit Cloud

⚠️ Not recommended for large PyTorch models. Free tier has 1GB storage and memory limits.

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Deploy from repository → Select this repo
3. May encounter memory/storage issues with large models

### Option 3: Docker (Self-hosted)

For production deployments:

```bash
docker build -t cambodian-news-app .
docker run -p 8501:8501 cambodian-news-app
```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open browser to `http://localhost:8501`

## Troubleshooting

**Models not loading?**
- Ensure Git LFS is installed: `git lfs install`
- Pull models: `git lfs pull`
- Check model files exist in `models/undersampling_no_environment/`

**ImportError on deployment?**
- Verify all imports are from standard libraries or `requirements.txt`
- Check `inference/predictor.py` doesn't import training modules

**Out of memory?**
- Use Hugging Face Spaces instead (more resources)
- Or optimize model loading with caching

## Notes

- Model files are stored using **Git LFS** to avoid repository size bloat
- Ensure Git LFS is installed before pulling large files
- Each model file is approximately 300-600 MB

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]
