#!/usr/bin/env python3
"""
Quick environment validation for Ollama GPT-OSS-20B setup
빠른 환경 검증 (PyTorch 없이)
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python_version():
    """Python 버전 검사"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python 버전 요구사항 충족")
        return True
    else:
        print("❌ Python 3.8+ 필요")
        return False

def check_virtual_environment():
    """가상환경 확인"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = sys.prefix
        print(f"✅ 가상환경 활성화됨: {Path(venv_path).name}")
        return True
    else:
        print("⚠️  가상환경이 활성화되지 않음")
        return False

def check_gpu():
    """GPU 확인"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line:
                name, total, free = line.split(', ')
                print(f"🎮 GPU: {name}")
                print(f"   📊 메모리: {free}MB 사용가능 / {total}MB 총용량")
                
                if int(free) >= 4000:  # 4GB 이상
                    print("✅ GPU 메모리 충분")
                    return True
                else:
                    print("⚠️  GPU 메모리 부족 (4GB+ 권장)")
                    return False
                    
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NVIDIA GPU 또는 드라이버 미발견")
        return False

def check_disk_space():
    """디스크 공간 확인"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024**3)
        
        print(f"💾 디스크 여유공간: {free_gb:.1f}GB")
        
        if free_gb >= 20:
            print("✅ 디스크 공간 충분")
            return True
        else:
            print("⚠️  디스크 공간 부족 (20GB+ 권장)")
            return False
    except Exception as e:
        print(f"❌ 디스크 공간 확인 실패: {e}")
        return False

def check_system_memory():
    """시스템 메모리 확인 (psutil 없이)"""
    try:
        # Linux의 /proc/meminfo 읽기
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
                total_gb = total_kb / (1024**2)
                print(f"🧠 시스템 메모리: {total_gb:.1f}GB")
                
                if total_gb >= 8:
                    print("✅ 시스템 메모리 충분")
                    return True
                else:
                    print("⚠️  시스템 메모리 부족 (8GB+ 권장)")
                    return False
    except Exception as e:
        print(f"⚠️  메모리 정보 확인 실패: {e}")
        return True  # 확인 실패해도 진행

def main():
    """메인 검사 함수"""
    print("🔍 Ollama GPT-OSS-20B 빠른 환경 검사\n")
    
    checks = [
        ("Python 버전", check_python_version),
        ("가상환경", check_virtual_environment), 
        ("GPU 상태", check_gpu),
        ("디스크 공간", check_disk_space),
        ("시스템 메모리", check_system_memory)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n📋 {name} 검사:")
        if check_func():
            passed += 1
    
    print(f"\n📊 검사 결과: {passed}/{total} 통과")
    
    if passed >= 4:  # 5개 중 4개 이상 통과
        print("🎉 환경 설정 준비 완료!")
        print("\n💡 다음 단계:")
        print("1. torch 설치: pip install torch")
        print("2. 전체 환경 검증: python check_environment.py")
        return True
    else:
        print("⚠️  환경 설정이 필요합니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)