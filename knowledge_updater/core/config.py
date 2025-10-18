import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[Dict[str, Any]] = None
        self._env_vars_loaded = False

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)

            if self._config is None:
                raise ValueError("Configuration file is empty or invalid")

            self._load_environment_variables()
            self._validate_config()

            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self._config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def get_config(self) -> Dict[str, Any]:
        if self._config is None:
            self.load_config()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        config = self.get_config()
        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def _load_environment_variables(self):
        if self._env_vars_loaded or not self._config:
            return

        def _replace_env_vars(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        env_value = os.getenv(env_var)
                        if env_value is not None:
                            obj[key] = env_value
                        else:
                            logger.warning(f"Environment variable {env_var} not found")
                    elif isinstance(value, (dict, list)):
                        _replace_env_vars(value)
            elif isinstance(obj, list):
                for item in obj:
                    _replace_env_vars(item)

        _replace_env_vars(self._config)
        self._env_vars_loaded = True

    def _validate_config(self):
        if not self._config:
            return

        required_sections = [
            'scheduler', 'data_sources', 'vector_db',
            'query_enhancement', 'rate_limiting', 'logging'
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        scheduler = self._config['scheduler']
        if not isinstance(scheduler.get('update_interval_hours'), int):
            raise ValueError("scheduler.update_interval_hours must be an integer")

        data_sources = self._config['data_sources']
        if 'rss_feeds' not in data_sources and 'apis' not in data_sources:
            raise ValueError("At least one data source (RSS or API) must be configured")

        vector_db = self._config['vector_db']
        if vector_db.get('provider') not in ['chromadb', 'pinecone', 'faiss']:
            raise ValueError("vector_db.provider must be one of: chromadb, pinecone, faiss")

        logger.info("Configuration validation completed successfully")

    def reload_config(self) -> Dict[str, Any]:
        self._config = None
        self._env_vars_loaded = False
        return self.load_config()

    def save_config(self, config_path: Optional[str] = None) -> None:
        save_path = config_path or self.config_path

        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self._config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Dict[str, Any]:
    return get_config_manager().get_config()