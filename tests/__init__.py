"""
Unit tests for the Knowledge Base Update System
"""

# Test configuration
import os
import tempfile
import shutil

# Set up test environment
TEST_CONFIG = {
    'test_data_dir': None,
    'test_logs_dir': None,
    'original_env': {}
}

def setup_test_environment():
    """Set up test environment"""
    # Create temporary directories
    test_data_dir = tempfile.mkdtemp(prefix='knowledge_updater_test_')
    test_logs_dir = os.path.join(test_data_dir, 'logs')

    os.makedirs(test_logs_dir, exist_ok=True)

    TEST_CONFIG['test_data_dir'] = test_data_dir
    TEST_CONFIG['test_logs_dir'] = test_logs_dir

    # Store original environment
    TEST_CONFIG['original_env'] = os.environ.copy()

    # Set test environment variables
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['NEWSAPI_KEY'] = 'test_newsapi_key'

def teardown_test_environment():
    """Clean up test environment"""
    # Restore original environment
    os.environ.clear()
    os.environ.update(TEST_CONFIG['original_env'])

    # Clean up test directories
    if TEST_CONFIG['test_data_dir'] and os.path.exists(TEST_CONFIG['test_data_dir']):
        shutil.rmtree(TEST_CONFIG['test_data_dir'])

    TEST_CONFIG['test_data_dir'] = None
    TEST_CONFIG['test_logs_dir'] = None

# Set up and tear down automatically
setup_test_environment()