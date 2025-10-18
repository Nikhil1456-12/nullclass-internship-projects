# Knowledge Base Update System

A robust, scalable system for dynamically expanding chatbot knowledge bases through automated, periodic updates to a vector database. This system ingests data from multiple sources, processes it, and makes it available for intelligent query enhancement.

## 🚀 Features

- **Automated Data Ingestion**: Fetches content from RSS feeds, APIs, and other sources
- **Intelligent Processing**: Cleans, validates, and deduplicates content
- **Vector Embeddings**: Supports multiple embedding backends (OpenAI, Sentence-BERT)
- **Vector Database**: ChromaDB integration with similarity search
- **Query Enhancement**: Real-time context retrieval for chatbot responses
- **Scheduled Updates**: Automated periodic knowledge base updates
- **Rate Limiting**: Built-in rate limiting for API calls
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Error Handling**: Robust error handling and recovery mechanisms
- **Scalable Architecture**: Batch processing and concurrent operations

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledge-base-updater
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your settings
   ```

4. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export NEWSAPI_KEY="your-newsapi-key"
   # Add other API keys as needed
   ```

## ⚙️ Configuration

The system uses a comprehensive YAML configuration file (`config.yaml`) with the following sections:

### Scheduler Configuration
```yaml
scheduler:
  update_interval_hours: 24  # How often to run updates
  timezone: "UTC"
  max_concurrent_jobs: 3
```

### Data Sources
```yaml
data_sources:
  rss_feeds:
    - name: "TechCrunch"
      url: "https://techcrunch.com/feed/"
      enabled: true
      max_articles: 50

  apis:
    - name: "NewsAPI"
      base_url: "https://newsapi.org/v2"
      api_key_env: "NEWSAPI_KEY"
      endpoints:
        - path: "/top-headlines"
          params:
            category: "technology"
            language: "en"
```

### Vector Database
```yaml
vector_db:
  provider: "chromadb"
  collection_name: "knowledge_base"
  persist_directory: "./data/chroma"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

## 🎯 Usage

### Command Line Interface

#### Update Knowledge Base
```bash
python main.py update
```

#### Query Knowledge Base
```bash
python main.py query "What is artificial intelligence?"
```

#### Test System
```bash
python main.py test
```

#### Start Server (for scheduled updates)
```bash
python main.py serve --host localhost --port 8000
```

### Programmatic Usage

```python
from knowledge_updater.data_sources.manager import DataSourceManager
from knowledge_updater.vector_db.manager import VectorDBManager
from knowledge_updater.query.enhancer import QueryEnhancer

# Fetch and process new data
data_manager = DataSourceManager()
articles = data_manager.fetch_and_process()

# Update vector database
vector_manager = VectorDBManager()
update_stats = vector_manager.update_knowledge_base(articles)

# Enhance queries
enhancer = QueryEnhancer()
enhanced = enhancer.enhance_query("What is machine learning?")
```

## 🏗️ Architecture

### Core Components

1. **Scheduler** (`knowledge_updater/core/scheduler.py`)
   - Manages periodic updates using APScheduler
   - Supports cron and interval triggers
   - Thread-safe job management

2. **Data Sources** (`knowledge_updater/data_sources/`)
   - RSS feed handler with robust parsing
   - API client with rate limiting
   - Content validation and cleaning

3. **Embedding Generator** (`knowledge_updater/embeddings/`)
   - OpenAI embeddings backend
   - Sentence-BERT backend
   - Batch processing support

4. **Vector Database** (`knowledge_updater/vector_db/`)
   - ChromaDB client with persistence
   - Similarity search capabilities
   - Metadata management

5. **Query Enhancer** (`knowledge_updater/query/`)
   - Real-time context retrieval
   - Query feature extraction
   - Response enhancement

### Data Flow

```
Sources (RSS/API) → Data Manager → Processing → Embedding → Vector DB
                                                           ↓
User Query → Query Enhancer → Context Retrieval → Enhanced Response
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_data_processor.py -v
```

## 📊 Monitoring

### Logging
The system provides structured logging with configurable levels:
- `DEBUG`: Detailed operation information
- `INFO`: General operation status
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

### Performance Metrics
- Update frequency and success rates
- Query response times
- Embedding generation performance
- Database operation statistics

## 🔧 Advanced Configuration

### Custom Embedding Models
```yaml
vector_db:
  openai_embeddings:
    model: "text-embedding-ada-002"
    batch_size: 100
    api_key_env: "OPENAI_API_KEY"
```

### Rate Limiting
```yaml
rate_limiting:
  default_requests_per_minute: 60
  max_retries: 3
  retry_backoff_factor: 0.3
```

### Data Quality Filters
```yaml
data_quality:
  min_content_length: 100
  max_content_length: 10000
  content_filters:
    - "spam"
    - "advertisement"
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- **Network failures**: Automatic retry with exponential backoff
- **API rate limits**: Intelligent rate limiting and queuing
- **Malformed data**: Robust parsing with fallback mechanisms
- **Database errors**: Transaction rollback and recovery

## 🔒 Privacy and Compliance

- **GDPR Compliance**: Configurable data retention policies
- **Data Anonymization**: Personal data filtering options
- **Source Exclusions**: Configurable source filtering
- **Audit Logging**: Complete operation logs for compliance

## 📈 Scalability

### Batch Processing
- Configurable batch sizes for large datasets
- Concurrent processing support
- Memory-efficient streaming

### Performance Optimization
- Connection pooling for external APIs
- Caching for frequently accessed data
- Async operations where applicable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the configuration documentation

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
  - RSS and API data ingestion
  - ChromaDB vector storage
  - Query enhancement
  - Scheduled updates
  - Comprehensive testing

## 🗺️ Roadmap

- [ ] Web dashboard for monitoring
- [ ] Additional vector database backends (Pinecone, Weaviate)
- [ ] Advanced NLP preprocessing
- [ ] Multi-language support
- [ ] Real-time streaming updates
- [ ] Advanced analytics and insights