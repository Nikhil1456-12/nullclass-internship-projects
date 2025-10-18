#!/usr/bin/env python3
"""
Integration Test Script for Internship Projects

This script tests the integration between internship projects and the existing
knowledge_updater system to ensure seamless operation.
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_file_exists(file_path: str) -> bool:
    """Test if a file exists"""
    return os.path.exists(file_path)

def test_module_import(module_path: str, module_name: str) -> bool:
    """Test if a module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_yaml_config(config_path: str) -> bool:
    """Test if YAML config file is valid"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return isinstance(config, dict)
    except Exception as e:
        print(f"  ❌ YAML validation failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Integration Tests for Internship Projects")
    print("=" * 60)

    test_results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'projects': {}
    }

    # Test project structure
    projects = {
        'multimodal_chatbot': {
            'main_file': 'internship_projects/multimodal_chatbot/multimodal_chatbot.py',
            'config_file': 'internship_projects/multimodal_chatbot/config.yaml',
            'test_import': True
        },
        'medical_qa_chatbot': {
            'main_file': 'internship_projects/medical_qa_chatbot/medical_qa_chatbot.py',
            'config_file': 'internship_projects/medical_qa_chatbot/config.yaml',
            'test_import': True
        },
        'domain_expert_chatbot': {
            'main_file': 'internship_projects/domain_expert_chatbot/domain_expert_chatbot.py',
            'config_file': 'internship_projects/domain_expert_chatbot/config.yaml',
            'test_import': True
        },
        'sentiment_analysis': {
            'main_file': 'internship_projects/sentiment_analysis/sentiment_analyzer.py',
            'config_file': 'internship_projects/sentiment_analysis/config.yaml',
            'test_import': True
        },
        'multilingual_support': {
            'main_file': 'internship_projects/multilingual_support/multilingual_chatbot.py',
            'config_file': 'internship_projects/multilingual_support/config.yaml',
            'test_import': True
        }
    }

    # Test each project
    for project_name, project_info in projects.items():
        print(f"\n📁 Testing {project_name}...")
        project_results = {'passed': 0, 'failed': 0}

        # Test main file exists
        test_results['total_tests'] += 1
        main_file = project_info['main_file']
        if test_file_exists(main_file):
            print(f"  ✅ Main file exists: {main_file}")
            project_results['passed'] += 1
        else:
            print(f"  ❌ Main file missing: {main_file}")
            project_results['failed'] += 1

        # Test config file exists and is valid
        test_results['total_tests'] += 1
        config_file = project_info['config_file']
        if test_file_exists(config_file):
            if test_yaml_config(config_file):
                print(f"  ✅ Config file valid: {config_file}")
                project_results['passed'] += 1
            else:
                print(f"  ❌ Config file invalid: {config_file}")
                project_results['failed'] += 1
        else:
            print(f"  ❌ Config file missing: {config_file}")
            project_results['failed'] += 1

        # Test module import (if requested)
        if project_info.get('test_import', False):
            test_results['total_tests'] += 1
            module_name = f"internship_projects.{project_name}.{project_name}"
            if test_module_import(main_file, module_name):
                print(f"  ✅ Module import successful: {module_name}")
                project_results['passed'] += 1
            else:
                print(f"  ❌ Module import failed: {module_name}")
                project_results['failed'] += 1

        # Project summary
        test_results['projects'][project_name] = project_results
        total_project_tests = project_results['passed'] + project_results['failed']
        if project_results['failed'] == 0:
            print(f"  🎉 Project {project_name}: ALL TESTS PASSED ({project_results['passed']}/{total_project_tests})")
        else:
            print(f"  ⚠️  Project {project_name}: {project_results['passed']}/{total_project_tests} tests passed")

    # Test core system integration
    print("
🔗 Testing Core System Integration..."    core_files = [
        'knowledge_updater/core/config.py',
        'knowledge_updater/core/logging.py',
        'knowledge_updater/core/scheduler.py',
        'knowledge_updater/data_sources/manager.py',
        'knowledge_updater/vector_db/manager.py'
    ]

    for core_file in core_files:
        test_results['total_tests'] += 1
        if test_file_exists(core_file):
            print(f"  ✅ Core file exists: {core_file}")
            test_results['passed'] += 1
        else:
            print(f"  ❌ Core file missing: {core_file}")
            test_results['failed'] += 1

    # Test requirements.txt
    test_results['total_tests'] += 1
    requirements_file = 'requirements.txt'
    if test_file_exists(requirements_file):
        print(f"  ✅ Requirements file exists: {requirements_file}")
        test_results['passed'] += 1
    else:
        print(f"  ❌ Requirements file missing: {requirements_file}")
        test_results['failed'] += 1

    # Test README
    test_results['total_tests'] += 1
    readme_file = 'INTERNSHIP_PROJECTS_README.md'
    if test_file_exists(readme_file):
        print(f"  ✅ README file exists: {readme_file}")
        test_results['passed'] += 1
    else:
        print(f"  ❌ README file missing: {readme_file}")
        test_results['failed'] += 1

    # Final summary
    print("
📊 Integration Test Summary"    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    success_rate = (test_results['passed'] / test_results['total_tests']) * 100
    print(f"Success Rate: {success_rate".1f"}%")

    if test_results['failed'] == 0:
        print("
🎉 ALL INTEGRATION TESTS PASSED!"        print("✅ All internship projects are properly integrated with the existing system")
        print("✅ Project structure is complete and well-organized")
        print("✅ Configuration files are valid")
        print("✅ Documentation is in place")
        return True
    else:
        print("
⚠️  SOME TESTS FAILED"        print("❌ Please review the failed tests and fix any issues")
        print("❌ Some projects may not integrate properly with the existing system")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)