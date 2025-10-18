"""
Unit tests for configuration management
"""

import os
import tempfile
import pytest
import yaml
from unittest.mock import patch, mock_open

from knowledge_updater.core.config import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_config = {
            'scheduler': {
                'update_interval_hours': 24,
                'timezone': 'UTC'
            },
            'data_sources': {
                'rss_feeds': [
                    {
                        'name': 'Test Feed',
                        'url': 'https://example.com/feed.xml',
                        'enabled': True
                    }
                ]
            },
            'vector_db': {
                'provider': 'chromadb',
                'collection_name': 'test_collection'
            }
        }

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        manager = ConfigManager()
        assert manager.config_path == "config.yaml"
        assert manager._config is None

        # Test with custom path
        custom_manager = ConfigManager("/custom/path/config.yaml")
        assert custom_manager.config_path == "/custom/path/config.yaml"

    def test_load_config_from_file(self):
        """Test loading configuration from file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            config = manager.load_config()

            assert config == self.test_config
            assert manager._config == self.test_config
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test loading configuration with non-existent file"""
        manager = ConfigManager("/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            manager.load_config()

    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            with pytest.raises(yaml.YAMLError):
                manager.load_config()
        finally:
            os.unlink(config_path)

    def test_load_config_empty_file(self):
        """Test loading configuration with empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            with pytest.raises(ValueError, match="Configuration file is empty"):
                manager.load_config()
        finally:
            os.unlink(config_path)

    def test_get_config_value(self):
        """Test getting configuration values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            manager.load_config()

            # Test nested value retrieval
            assert manager.get('scheduler.update_interval_hours') == 24
            assert manager.get('data_sources.rss_feeds.0.name') == 'Test Feed'
            assert manager.get('nonexistent.key', 'default') == 'default'
        finally:
            os.unlink(config_path)

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config"""
        config_with_env = {
            'api_key': '${TEST_API_KEY}',
            'nested': {
                'value': '${TEST_NESTED_VALUE}'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_env, f)
            config_path = f.name

        try:
            # Set environment variables
            os.environ['TEST_API_KEY'] = 'test_key_value'
            os.environ['TEST_NESTED_VALUE'] = 'nested_value'

            manager = ConfigManager(config_path)
            config = manager.load_config()

            assert config['api_key'] == 'test_key_value'
            assert config['nested']['value'] == 'nested_value'
        finally:
            # Clean up environment
            if 'TEST_API_KEY' in os.environ:
                del os.environ['TEST_API_KEY']
            if 'TEST_NESTED_VALUE' in os.environ:
                del os.environ['TEST_NESTED_VALUE']
            os.unlink(config_path)

    def test_config_validation(self):
        """Test configuration validation"""
        # Test missing required section
        invalid_config = {
            'scheduler': {'update_interval_hours': 24}
            # Missing data_sources and vector_db
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            with pytest.raises(ValueError, match="Missing required configuration section"):
                manager.load_config()
        finally:
            os.unlink(config_path)

    def test_config_validation_scheduler_settings(self):
        """Test scheduler configuration validation"""
        # Test invalid scheduler settings
        invalid_config = {
            'scheduler': {
                'update_interval_hours': 'invalid'  # Should be int
            },
            'data_sources': {'rss_feeds': []},
            'vector_db': {'provider': 'chromadb'}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            with pytest.raises(ValueError, match="scheduler.update_interval_hours must be an integer"):
                manager.load_config()
        finally:
            os.unlink(config_path)

    def test_config_validation_vector_db_provider(self):
        """Test vector database provider validation"""
        # Test invalid provider
        invalid_config = {
            'scheduler': {'update_interval_hours': 24},
            'data_sources': {'rss_feeds': []},
            'vector_db': {'provider': 'invalid_provider'}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)

            with pytest.raises(ValueError, match="vector_db.provider must be one of"):
                manager.load_config()
        finally:
            os.unlink(config_path)

    def test_reload_config(self):
        """Test configuration reload"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            config = manager.load_config()

            # Modify config and reload
            manager._config['scheduler']['update_interval_hours'] = 48
            reloaded_config = manager.reload_config()

            assert reloaded_config['scheduler']['update_interval_hours'] == 24  # Back to original
        finally:
            os.unlink(config_path)

    def test_save_config(self):
        """Test saving configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            manager.load_config()

            # Modify and save
            manager._config['scheduler']['update_interval_hours'] = 48

            # Save to new file
            save_path = config_path + '.saved'
            manager.save_config(save_path)

            # Verify saved file
            with open(save_path, 'r') as f:
                saved_config = yaml.safe_load(f)

            assert saved_config['scheduler']['update_interval_hours'] == 48
        finally:
            # Clean up
            for path in [config_path, config_path + '.saved']:
                if os.path.exists(path):
                    os.unlink(path)

    @patch.dict(os.environ, {'TEST_API_KEY': 'test_value'})
    def test_missing_environment_variable_warning(self):
        """Test warning for missing environment variables"""
        config_with_missing_env = {
            'api_key': '${MISSING_ENV_VAR}',
            'scheduler': {'update_interval_hours': 24},
            'data_sources': {'rss_feeds': []},
            'vector_db': {'provider': 'chromadb'}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_missing_env, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_path)
            config = manager.load_config()

            # Should still load but with None for missing env var
            assert config['api_key'] is None
        finally:
            os.unlink(config_path)