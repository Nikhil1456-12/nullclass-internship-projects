# 📊 Sentiment Analysis Tool

A comprehensive sentiment analysis application that processes text to determine emotional tone, sentiment polarity, and intent using advanced natural language processing techniques.

## 🌟 Overview

This sentiment analysis tool provides real-time analysis of text content to identify emotions, sentiment, and intentions. It's designed for social media monitoring, customer feedback analysis, and content moderation applications.

## ✨ Key Features

### 😊 Sentiment Detection
- **Multi-dimensional Analysis** - Positive, negative, neutral classification
- **Emotion Recognition** - Joy, anger, sadness, fear, surprise, disgust
- **Intent Detection** - Query, complaint, appreciation, suggestion identification
- **Confidence Scoring** - Reliability metrics for each analysis

### 📈 Advanced Analytics
- **Real-time Processing** - Live text analysis capabilities
- **Batch Processing** - Handle multiple texts simultaneously
- **Visualization Dashboard** - Interactive charts and sentiment trends
- **Export Capabilities** - CSV, JSON, and PDF report generation

### 🎯 Accuracy Optimization
- **Contextual Understanding** - Considers context and domain-specific language
- **Multi-language Support** - Analysis in multiple languages
- **Customizable Thresholds** - Adjustable sensitivity settings
- **Continuous Learning** - Model improvement over time

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- NLP libraries (transformers, torch)
- Visualization libraries (plotly, matplotlib)

### Installation & Setup

1. **Navigate to project directory:**
   ```bash
   cd internship_projects/sentiment_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

3. **Download required models** (first run):
   ```python
   # Models will be automatically downloaded on first use
   from sentiment_analyzer import SentimentAnalyzer
   analyzer = SentimentAnalyzer()
   ```

4. **Run the application:**
   ```bash
   streamlit run sentiment_analyzer.py
   ```

5. **Access at:** `http://localhost:8501`

## 💬 Usage Examples

### Single Text Analysis
```python
# Example analysis:
# Input: "I love this new product! It's amazing and works perfectly."
# Output:
#   - Sentiment: Positive (0.95 confidence)
#   - Emotions: Joy (0.89), Surprise (0.23)
#   - Intent: Appreciation
#   - Overall Score: 8.5/10
```

### Batch Analysis
```python
# Analyze multiple texts at once
texts = [
    "This service is terrible and slow!",
    "Thank you for the quick response.",
    "How do I reset my password?"
]

analyzer = SentimentAnalyzer()
results = analyzer.analyze_batch(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Emotions: {result['emotions']}")
```

### Programmatic Usage
```python
from internship_projects.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
result = analyzer.analyze_text("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
print(f"Emotions: {result['emotions']}")

# Analyze with custom thresholds
result = analyzer.analyze_text(
    "This is okay, not great but not bad either.",
    sentiment_threshold=0.1,
    emotion_threshold=0.3
)
```

## ⚙️ Configuration

### Sentiment Analysis Settings
```yaml
sentiment_analysis:
  threshold: 0.1
  emotion_threshold: 0.3
  lexicon_path: "data/sentiment_lexicon"
```

### Model Configuration
```yaml
models:
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  emotion_model: "j-hartmann/emotion-english-distilroberta-base"
  intent_model: "facebook/bart-large-mnli"
```

### Processing Settings
```yaml
processing:
  max_text_length: 512
  batch_size: 32
  cache_results: true
  parallel_processing: true
```

## 🏗️ Architecture

### Core Components

#### 1. SentimentAnalyzer
- **Purpose:** Main analysis engine for sentiment processing
- **Features:** Multi-model integration, result aggregation, confidence calculation
- **Dependencies:** Transformer models, preprocessing pipeline

#### 2. TextPreprocessor
- **Purpose:** Text cleaning and normalization
- **Features:** Tokenization, stopword removal, lemmatization
- **Dependencies:** NLTK, spaCy, custom preprocessing rules

#### 3. ModelManager
- **Purpose:** Handles multiple ML models for different analysis types
- **Features:** Model loading, caching, inference optimization
- **Dependencies:** transformers, torch, model configuration

#### 4. ResultAggregator
- **Purpose:** Combines results from multiple models
- **Features:** Weighted scoring, conflict resolution, confidence aggregation
- **Dependencies:** Scoring algorithms, result normalization

### Processing Pipeline
```
Raw Text → Preprocessing → Model Analysis → Result Aggregation → Visualization → Export
```

## 🔧 API Reference

### SentimentAnalyzer Class

#### Key Methods

**`__init__(config_path=None)`**
- Initializes the sentiment analyzer
- Loads models and configurations

**`analyze_text(text, **kwargs)`**
- Analyzes a single text input
- **Parameters:**
  - `text` (str): Text to analyze
  - `sentiment_threshold` (float): Minimum confidence threshold
  - `emotion_threshold` (float): Emotion detection threshold
- **Returns:** Dictionary with analysis results

**`analyze_batch(texts, batch_size=32)`**
- Analyzes multiple texts efficiently
- **Parameters:**
  - `texts` (list): List of texts to analyze
  - `batch_size` (int): Batch processing size
- **Returns:** List of analysis result dictionaries

**`get_sentiment_score(text)`**
- Returns numerical sentiment score (-1 to 1)
- **Returns:** Float sentiment polarity score

**`extract_emotions(text)`**
- Extracts emotion probabilities from text
- **Returns:** Dictionary of emotion types and scores

## 📊 Analysis Types

### Sentiment Classification
- **Positive** - Enthusiastic, satisfied, happy language
- **Negative** - Dissatisfied, angry, disappointed language
- **Neutral** - Factual, informative, balanced language

### Emotion Detection
- **Joy** - Happiness, delight, amusement
- **Sadness** - Disappointment, grief, melancholy
- **Anger** - Frustration, irritation, rage
- **Fear** - Anxiety, worry, terror
- **Surprise** - Amazement, shock, astonishment
- **Disgust** - Revulsion, contempt, aversion

### Intent Recognition
- **Query** - Seeking information or assistance
- **Complaint** - Expressing dissatisfaction or problems
- **Appreciation** - Positive feedback or thanks
- **Suggestion** - Offering ideas or recommendations

## 📈 Visualization Features

### Interactive Dashboards
- **Sentiment Distribution** - Pie charts and bar graphs
- **Emotion Trends** - Time-series emotion tracking
- **Word Clouds** - Frequent terms by sentiment
- **Heat Maps** - Sentiment intensity visualization

### Export Options
- **CSV Reports** - Structured data for analysis
- **JSON API** - Machine-readable results
- **PDF Summaries** - Formatted reports
- **Image Charts** - Visual export options

## 🧪 Testing & Validation

### Test Coverage
- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow validation
- **Performance Tests** - Speed and resource usage testing
- **Accuracy Tests** - Model performance validation

### Running Tests
```bash
# Run all sentiment analysis tests
python -m pytest tests/ -v -k sentiment

# Run performance benchmarks
python benchmarks/sentiment_performance.py

# Validate model accuracy
python validation/accuracy_test.py
```

## 🚀 Deployment

### Streamlit Cloud Deployment
1. **Connect to GitHub** - Link repository to Streamlit Cloud
2. **App Configuration:**
   - **Main file:** `internship_projects/sentiment_analysis/sentiment_analyzer.py`
   - **Requirements:** Auto-detected from `requirements.txt`
3. **Resource Allocation:**
   - **Memory:** 2GB+ recommended for model loading
   - **CPU:** Multi-core for batch processing
4. **Deploy** - Automatic deployment with model caching

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "sentiment_analyzer.py", "--server.headless", "true"]
```

### Production Considerations
- **Model Caching** - Pre-load models for faster inference
- **Auto-scaling** - Handle variable traffic loads
- **Monitoring** - Track model performance and drift
- **Updates** - Automated model retraining pipeline

## 📊 Performance Metrics

- **Analysis Speed** - < 2 seconds per text
- **Batch Throughput** - 100+ texts per minute
- **Accuracy** - >85% for sentiment classification
- **Emotion Detection** - >80% accuracy across emotion categories
- **Intent Recognition** - >90% accuracy for clear intents

## 🔧 Customization

### Adding Custom Models
```python
# Register new sentiment model
from sentiment_analyzer import ModelManager

model_manager = ModelManager()
model_manager.register_model(
    name="custom_sentiment",
    model_path="path/to/custom/model",
    model_type="sentiment"
)
```

### Custom Emotion Categories
```python
# Define custom emotion taxonomy
custom_emotions = {
    'brand_loyalty': ['loyal', 'dedicated', 'committed'],
    'purchase_intent': ['buy', 'purchase', 'interested'],
    'churn_risk': ['cancel', 'leave', 'switch']
}
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
- Check internet connection for model downloads
- Verify available disk space
- Check model compatibility with hardware

**Slow Performance**
- Reduce batch size for memory-constrained systems
- Enable model caching
- Consider using smaller models for deployment

**Low Accuracy**
- Adjust confidence thresholds
- Fine-tune model parameters
- Consider domain-specific training data

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
DEBUG=true streamlit run sentiment_analyzer.py
```

## 📚 Dependencies

Key libraries and frameworks:
- **streamlit** - Web interface
- **transformers** - Pre-trained language models
- **torch** - Deep learning framework
- **plotly** - Interactive visualizations
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **nltk** - Natural language processing
- **scikit-learn** - Machine learning utilities

## 🤝 Contributing

### Development Guidelines
1. **Model Validation** - Test new models thoroughly
2. **Performance Testing** - Ensure efficient processing
3. **Documentation** - Update API and usage docs
4. **Backward Compatibility** - Maintain existing interfaces

### Contribution Process
1. Fork the repository
2. Create a sentiment analysis feature branch
3. Implement and test changes
4. Submit for review
5. Merge after approval

## 📋 Use Cases

### Social Media Monitoring
- Brand sentiment tracking
- Crisis detection
- Influencer analysis
- Campaign effectiveness

### Customer Service
- Feedback analysis
- Issue prioritization
- Agent performance
- Satisfaction tracking

### Content Moderation
- Toxic content detection
- Spam identification
- Harassment prevention
- Community guidelines

## 📞 Support

For technical issues:
- Check existing GitHub issues
- Review troubleshooting guide
- Contact development team

---

**Built with ❤️ for understanding human emotions through text**