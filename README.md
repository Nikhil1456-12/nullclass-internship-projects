# NullClass Internship Projects

A comprehensive collection of AI-powered chatbot applications and supporting infrastructure built during internship projects. This repository showcases various natural language processing implementations including multilingual support, domain expertise, medical Q&A, sentiment analysis, and multimodal capabilities.

## 🚀 Live Applications

All projects are deployed and accessible online:

- **🌐 Multilingual Chatbot** - [View on Streamlit Cloud](https://nullclass-multilingual-chatbot.streamlit.app/)
- **🏥 Medical Q&A Chatbot** - [View on Streamlit Cloud](https://nullclass-medical-qa.streamlit.app/)
- **📊 Sentiment Analysis** - [View on Streamlit Cloud](https://nullclass-sentiment-analysis.streamlit.app/)
- **🎯 Domain Expert Chatbot** - [View on Streamlit Cloud](https://nullclass-domain-expert.streamlit.app/)
- **🎨 Multimodal Chatbot** - [View on Streamlit Cloud](https://nullclass-multimodal.streamlit.app/)

## 📁 Project Structure

```
nullclass-internship-projects/
├── internship_projects/           # Main project implementations
│   ├── multilingual_support/      # Multi-language chatbot with translation
│   ├── medical_qa_chatbot/       # Medical question-answering system
│   ├── sentiment_analysis/        # Sentiment and emotion analysis
│   ├── domain_expert_chatbot/    # Domain-specific expert chatbot
│   └── multimodal_chatbot/        # Multi-modal input processing
├── knowledge_updater/            # Knowledge base management system
│   ├── core/                     # Core configuration and utilities
│   ├── data_sources/             # RSS and API data ingestion
│   ├── vector_db/                # Vector database management
│   ├── embeddings/               # Text embedding generation
│   └── query/                    # Query processing and enhancement
├── data/                         # Datasets and vector stores
├── tests/                        # Unit and integration tests
├── logs/                         # Application logs
└── requirements.txt              # Python dependencies
```

## 🌟 Featured Projects

### 🌐 Multilingual Support Chatbot
**Location:** `internship_projects/multilingual_support/`

An advanced chatbot supporting 12+ languages with:
- **Automatic Language Detection** - Identifies user language with high confidence
- **Real-time Translation** - Seamless translation between supported languages
- **Cultural Adaptation** - Contextually appropriate responses
- **Multi-language Conversation History** - Maintains context across languages

**Supported Languages:** English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi

### 🏥 Medical Q&A Chatbot
**Location:** `internship_projects/medical_qa_chatbot/`

A specialized medical question-answering system featuring:
- **Medical Knowledge Base** - Curated medical dataset integration
- **Evidence-based Responses** - Citations and confidence scoring
- **Safety Features** - Medical disclaimers and ethical guidelines
- **Multi-format Responses** - Text, structured data, and explanations

### 📊 Sentiment Analysis Tool
**Location:** `internship_projects/sentiment_analysis/`

Comprehensive sentiment analysis with:
- **Multi-dimensional Analysis** - Sentiment, emotion, and intent detection
- **Real-time Processing** - Live text analysis capabilities
- **Visualization Dashboard** - Interactive charts and metrics
- **Batch Processing** - Handle multiple texts simultaneously

### 🎯 Domain Expert Chatbot
**Location:** `internship_projects/domain_expert_chatbot/`

Domain-specific expertise system with:
- **ArXiv Integration** - Latest research papers and findings
- **Contextual Understanding** - Domain-specific terminology handling
- **Confidence Scoring** - Response reliability indicators
- **Source Attribution** - Citation of expert sources

### 🎨 Multimodal Chatbot
**Location:** `internship_projects/multimodal_chatbot/`

Multi-modal input processing featuring:
- **Text Analysis** - Advanced NLP capabilities
- **Image Processing** - Visual content understanding (planned)
- **Audio Processing** - Speech and sound analysis (planned)
- **Unified Interface** - Single chatbot for multiple input types

## 🔧 Technical Infrastructure

### Knowledge Updater System
**Location:** `knowledge_updater/`

A robust knowledge management system providing:
- **Automated Data Ingestion** - RSS feeds and API integration
- **Vector Database Management** - FAISS and ChromaDB support
- **Embedding Generation** - Multiple embedding model support
- **Query Enhancement** - Advanced search and retrieval
- **Rate Limiting** - API usage management and optimization

### Core Components
- **Configuration Management** - YAML-based configuration with validation
- **Logging System** - Structured logging with multiple levels
- **Scheduler** - Automated update scheduling and management
- **Data Quality** - Content filtering and validation

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Nikhil1456-12/nullclass-internship-projects.git
cd nullclass-internship-projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run any project (example: multilingual chatbot)
cd internship_projects/multilingual_support
streamlit run multilingual_chatbot.py
```

### Configuration
Each project includes its own `config.yaml` file. Key configuration areas:
- **Data Sources** - RSS feeds and API endpoints
- **Vector Database** - Storage and retrieval settings
- **Embedding Models** - Text encoding configurations
- **Rate Limiting** - API usage controls
- **Logging** - Debug and monitoring settings

## 📊 Datasets

The project includes several curated datasets:
- **ArXiv CS Dataset** - Computer science research papers
- **MedQuad Dataset** - Medical question-answer pairs
- **Custom Knowledge Base** - Domain-specific content

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_config.py
python -m pytest tests/test_data_processor.py

# Integration tests
python test_integration.py
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)
All projects are configured for easy deployment on Streamlit Cloud:
1. Connect your GitHub repository to Streamlit Cloud
2. Select the desired project directory as the app root
3. Configure environment variables (API keys, etc.)
4. Deploy with automatic updates on git push

### Local Development
Each project can be run independently:
```bash
# Terminal 1 - Multilingual Chatbot
cd internship_projects/multilingual_support
streamlit run multilingual_chatbot.py --server.port=8501

# Terminal 2 - Medical Q&A
cd internship_projects/medical_qa_chatbot
streamlit run medical_qa_chatbot.py --server.port=8502

# Additional projects on ports 8503-8505
```

## 🔑 Environment Variables

Configure these environment variables for full functionality:
```bash
# OpenAI API (for embeddings)
OPENAI_API_KEY=your_openai_key

# News API (for news data)
NEWSAPI_KEY=your_newsapi_key

# Optional: Custom embedding models
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Hugging Face** - For transformer models and datasets
- **FAISS** - For efficient vector similarity search
- **LangDetect** - For language detection capabilities
- **Google Translator** - For translation services

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check existing documentation

---

**Built with ❤️ during internship at NullClass**