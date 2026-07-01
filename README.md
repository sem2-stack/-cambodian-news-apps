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

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sem2-stack/-cambodian-news-apps.git
   cd -cambodian-news-apps
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   **Option A: Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

   **Option B: Flask/Custom Server**
   ```bash
   python server.py
   ```

## Usage

### Via Streamlit
Open your browser to `http://localhost:8501` after running streamlit.

### Via Web Server
Open your browser to `http://localhost:5000` (or configured port).

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
