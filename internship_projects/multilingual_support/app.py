import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from multilingual_chatbot import create_multilingual_ui

if __name__ == "__main__":
    create_multilingual_ui()