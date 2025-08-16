#!/bin/bash
echo "🚀 Activating Korean LLM environment..."
source korean-llm-env/bin/activate
echo "✅ Environment activated!"
echo ""
echo "💡 Quick commands:"
echo "  python quick_env_check.py     # 빠른 환경 검사"
echo "  python check_environment.py   # 전체 환경 검증 (torch 설치 후)"
echo "  python setup_environment.py   # 환경 재설정"
echo ""
echo "📋 Current environment: $(which python)"