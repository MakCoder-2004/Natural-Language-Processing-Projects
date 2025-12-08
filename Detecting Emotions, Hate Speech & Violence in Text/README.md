# Text Analysis: Emotion, Hate Speech, and Violence Detection

## Project Overview
This project focuses on analyzing text data to detect and classify content into three main categories:
1. **Emotion Detection**: Classifies text into different emotional states
2. **Hate Speech Detection**: Identifies hateful or offensive language
3. **Violence Detection**: Detects violent content in text

The system uses deep learning techniques to perform multi-label classification on text data, providing insights into the emotional tone and potential harmful content of the input text.

## Project Structure
```
.
├── Emotion.csv           # Dataset for emotion detection
├── Hate.csv              # Dataset for hate speech detection
├── Violence.csv          # Dataset for violence detection
├── main.ipynb            # Main Jupyter notebook containing the complete implementation
└── README.md             # This file
```

## Datasets Used

### 1. Emotion Detection Dataset
- **Source**: Custom dataset containing text samples labeled with emotions
- **Classes**: 6 different emotional states
- **Size**: Approximately 400,000+ samples
- **Features**: Text content and corresponding emotion labels

### 2. Hate Speech Detection Dataset
- **Source**: Publicly available hate speech dataset
- **Classes**: 
  - Hate speech
  - Offensive language
  - Neither
- **Size**: Several thousand samples
- **Features**: Text content and multi-label classification

### 3. Violence Detection Dataset
- **Source**: Curated dataset of tweets and online content
- **Classes**: Binary classification (violent/non-violent)
- **Features**: Text content and violence indicators

## Technical Implementation

### Libraries Used
- **Core Libraries**: 
  - Pandas & NumPy for data manipulation
  - Scikit-learn for preprocessing and evaluation
  - TensorFlow & Keras for deep learning models
  - Matplotlib & Seaborn for visualization

### Model Architecture
The project implements a multi-output neural network with the following components:
1. **Embedding Layer**: Converts text to dense vector representations
2. **LSTM Layers**: For capturing sequential patterns in text
3. **Global Pooling**: To aggregate sequence information
4. **Dense Layers**: For final classification
5. **Dropout Layers**: For regularization

### Key Features
- Multi-label classification for comprehensive text analysis
- Interactive text input for real-time prediction
- Confusion matrix visualization for model evaluation
- Preprocessing pipeline including tokenization and padding

## How to Use

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (install via `pip install -r requirements.txt`)

### Running the Project
1. Clone the repository
2. Install required dependencies
3. Open `main.ipynb` in Jupyter Notebook
4. Run all cells to train the model or load pre-trained weights
5. Use the interactive widget to test the model with custom text

## Results

The model provides:
- Emotion classification with confidence scores
- Hate speech detection with severity levels
- Violence probability score
- Combined analysis for comprehensive text understanding

## Future Improvements
- Expand training datasets for better generalization
- Implement transfer learning with BERT or other transformer models
- Add more detailed emotion categories
- Improve handling of sarcasm and context in text
- Create a web interface for easier access
