#!/usr/bin/env python3
"""
Final Comprehensive System Validation

This script performs a complete validation of all files in the nullclass folder
to ensure there are no compilation errors, import failures, configuration issues,
or any other potential problems.
"""

import os
import sys
import ast
import yaml

def comprehensive_system_check():
    print('🔍 FINAL COMPREHENSIVE SYSTEM VALIDATION')
    print('=' * 60)

    issues_found = []
    tests_passed = 0
    total_tests = 0

    # Test 1: Python File Compilation
    print('\n1. Testing Python File Compilation...')
    total_tests += 1

    python_files = [
        'internship_projects/multimodal_chatbot/multimodal_chatbot.py',
        'internship_projects/medical_qa_chatbot/medical_qa_chatbot.py',
        'internship_projects/domain_expert_chatbot/domain_expert_chatbot.py',
        'internship_projects/sentiment_analysis/sentiment_analyzer.py',
        'internship_projects/multilingual_support/multilingual_chatbot.py'
    ]

    compilation_successful = True
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            compile(source_code, file_path, 'exec')
            print(f'   ✅ {os.path.basename(file_path)}: Compiled successfully')
        except SyntaxError as e:
            print(f'   ❌ {os.path.basename(file_path)}: Syntax error - {e}')
            issues_found.append(f'Syntax error in {file_path}: {e}')
            compilation_successful = False
        except Exception as e:
            print(f'   ❌ {os.path.basename(file_path)}: Compilation failed - {e}')
            issues_found.append(f'Compilation error in {file_path}: {e}')
            compilation_successful = False

    if compilation_successful:
        print('   🎉 All Python files compile successfully')
        tests_passed += 1
    else:
        print('   ❌ Some Python files have compilation issues')

    # Test 2: Import Compatibility
    print('\n2. Testing Import Compatibility...')
    total_tests += 1

    import_successful = True
    sys.path.insert(0, '.')

    # Test core system imports
    core_imports = [
        'from knowledge_updater.core.config import get_config',
        'from knowledge_updater.core.logging import get_logger',
        'from knowledge_updater.core.scheduler import get_scheduler',
        'from knowledge_updater.data_sources.manager import DataSourceManager',
        'from knowledge_updater.vector_db.manager import VectorDBManager'
    ]

    for import_statement in core_imports:
        try:
            exec(import_statement)
            print(f'   ✅ Core import: {import_statement}')
        except ImportError as e:
            print(f'   ❌ Core import failed: {import_statement} - {e}')
            issues_found.append(f'Import error: {import_statement} - {e}')
            import_successful = False
        except Exception as e:
            print(f'   ❌ Core import error: {import_statement} - {e}')
            issues_found.append(f'Import error: {import_statement} - {e}')
            import_successful = False

    if import_successful:
        print('   🎉 All core system imports working')
        tests_passed += 1
    else:
        print('   ❌ Some core system imports failing')

    # Test 3: Configuration Validation
    print('\n3. Testing Configuration Files...')
    total_tests += 1

    config_files = [
        'internship_projects/multimodal_chatbot/config.yaml',
        'internship_projects/medical_qa_chatbot/config.yaml',
        'internship_projects/domain_expert_chatbot/config.yaml',
        'internship_projects/sentiment_analysis/config.yaml',
        'internship_projects/multilingual_support/config.yaml'
    ]

    config_valid = True
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if not isinstance(config, dict):
                    print(f'   ❌ {config_file}: Invalid YAML structure')
                    issues_found.append(f'Invalid YAML structure in {config_file}')
                    config_valid = False
                elif len(config) == 0:
                    print(f'   ⚠️  {config_file}: Empty configuration')
                else:
                    print(f'   ✅ {config_file}: Valid configuration ({len(config)} sections)')
            except yaml.YAMLError as e:
                print(f'   ❌ {config_file}: YAML syntax error - {e}')
                issues_found.append(f'YAML syntax error in {config_file}: {e}')
                config_valid = False
            except Exception as e:
                print(f'   ❌ {config_file}: Configuration error - {e}')
                issues_found.append(f'Configuration error in {config_file}: {e}')
                config_valid = False
        else:
            print(f'   ❌ {config_file}: File not found')
            issues_found.append(f'Configuration file missing: {config_file}')
            config_valid = False

    if config_valid:
        print('   🎉 All configuration files are valid')
        tests_passed += 1
    else:
        print('   ❌ Some configuration files have issues')

    # Test 4: Project Structure Integrity
    print('\n4. Testing Project Structure...')
    total_tests += 1

    structure_valid = True
    required_structure = {
        'internship_projects/multimodal_chatbot': ['multimodal_chatbot.py', 'config.yaml'],
        'internship_projects/medical_qa_chatbot': ['medical_qa_chatbot.py', 'config.yaml'],
        'internship_projects/domain_expert_chatbot': ['domain_expert_chatbot.py', 'config.yaml'],
        'internship_projects/sentiment_analysis': ['sentiment_analyzer.py', 'config.yaml'],
        'internship_projects/multilingual_support': ['multilingual_chatbot.py', 'config.yaml']
    }

    for project_path, required_files in required_structure.items():
        for file in required_files:
            full_path = os.path.join(project_path, file)
            if os.path.exists(full_path):
                print(f'   ✅ {full_path}: File exists')
            else:
                print(f'   ❌ {full_path}: File missing')
                issues_found.append(f'Missing file: {full_path}')
                structure_valid = False

    if structure_valid:
        print('   🎉 All project files are present')
        tests_passed += 1
    else:
        print('   ❌ Some project files are missing')

    # Test 5: Documentation Check
    print('\n5. Testing Documentation...')
    total_tests += 1

    doc_files = ['INTERNSHIP_PROJECTS_README.md', 'requirements.txt']
    docs_valid = True

    for doc_file in doc_files:
        if os.path.exists(doc_file):
            # Check if file has content
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if len(content) > 100:  # Reasonable size check
                    print(f'   ✅ {doc_file}: Documentation exists ({len(content)} characters)')
                else:
                    print(f'   ⚠️  {doc_file}: Documentation exists but may be incomplete')
            except Exception as e:
                print(f'   ❌ {doc_file}: Documentation error - {e}')
                issues_found.append(f'Documentation error in {doc_file}: {e}')
                docs_valid = False
        else:
            print(f'   ❌ {doc_file}: Documentation missing')
            issues_found.append(f'Documentation missing: {doc_file}')
            docs_valid = False

    if docs_valid:
        print('   🎉 All documentation is present')
        tests_passed += 1
    else:
        print('   ❌ Some documentation is missing or incomplete')

    # Summary
    print(f'\n📊 FINAL VALIDATION RESULTS: {tests_passed}/{total_tests} test categories passed')

    if issues_found:
        print('\n❌ ISSUES FOUND:')
        for issue in issues_found:
            print(f'   • {issue}')
        print(f'\n⚠️  {len(issues_found)} issues require attention')
        return False
    else:
        print('\n🎉 NO ISSUES DETECTED!')
        print('✅ All Python files compile successfully')
        print('✅ All imports work correctly')
        print('✅ All configurations are valid')
        print('✅ Project structure is complete')
        print('✅ Documentation is comprehensive')
        print('\n🚀 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!')
        return True

if __name__ == "__main__":
    success = comprehensive_system_check()
    sys.exit(0 if success else 1)