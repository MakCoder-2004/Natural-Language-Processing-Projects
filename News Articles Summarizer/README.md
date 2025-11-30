# News Articles Summarizer

A Streamlit web application that fetches and summarizes news articles from any given URL. The application extracts key information such as the article title, authors, and publication date, and provides a concise summary using various summarization techniques.

## Features

- **Multiple Summarization Methods**:
  - **Newspaper**: Uses the built-in summarization from the `newspaper3k` library
  - **TextRank**: Implements the TextRank algorithm for extractive summarization
  - **First N Sentences**: Simple extraction of the first N sentences
  - **Abstractive Summarization** (optional): Uses Hugging Face's Transformers for AI-generated summaries

- **User-Friendly Interface**:
  - Clean, intuitive UI with Streamlit
  - Adjustable summary length
  - Real-time processing feedback

- **Article Details**:
  - Extracts and displays article title
  - Lists all authors
  - Shows publication date
  - Presents the full article text

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Internet connection (for fetching articles)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-articles-summarizer.git
   cd news-articles-summarizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   For optional abstractive summarization (requires more resources):
   ```bash
   pip install transformers torch
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run streamlit\ application\app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Enter a news article URL in the input field

4. Select your preferred summarization method:
   - **Newspaper**: Good for general articles
   - **TextRank**: Better for technical or complex articles
   - **First N Sentences**: Quick overview
   - **Abstractive**: AI-generated summary (requires additional dependencies)

5. Adjust the number of sentences for the summary using the slider

6. Click "Fetch & Summarize" to process the article

## How It Works

1. **Article Fetching**:
   - The application uses the `newspaper3k` library to download and parse the article content from the provided URL
   - It extracts the main text, title, authors, and publication date

2. **Summarization Methods**:
   - **TextRank**: Implements a graph-based ranking algorithm that treats sentences as nodes and their relationships as edges
   - **Newspaper**: Uses the built-in summarization from `newspaper3k`
   - **First N Sentences**: Simple extraction of the first N sentences of the article
   - **Abstractive**: Uses Hugging Face's DistilBART model to generate new sentences that summarize the content

3. **User Interface**:
   - Built with Streamlit for a simple, interactive experience
   - Responsive design that works on different screen sizes

## Project Structure

```
News Articles Summarizer/
├── streamlit application/
│   └── app.py              # Main application code
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## Dependencies

- `streamlit` - Web application framework
- `newspaper3k` - Article scraping and basic NLP
- `transformers` - For abstractive summarization (optional)
- `torch` - Required for transformers

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web framework
- [Newspaper3k](https://newspaper.readthedocs.io/) for article extraction
- [Hugging Face](https://huggingface.co/) for the Transformers library and pre-trained models

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
