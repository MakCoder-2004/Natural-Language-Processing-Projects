News Article Details & Summary Streamlit App

Usage

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r "requirements.txt"
```

2. Run the app:

```powershell
streamlit run "streamlit application\app.py"
```

Notes
- The app uses `newspaper3k` to download and parse articles. `newspaper3k` relies on NLTK's `punkt` tokenizer which the app attempts to download automatically on first run.
- Some websites block scrapers or require JS; for those, extraction may fail. Consider copying the article text into another tool if needed.

