#!/usr/bin/env python3
"""
Automated environment setup script for Ollama GPT-OSS-20B Unsloth fine-tuning
자동화된 환경 설정: 가상환경 생성, 패키지 설치, 검증
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class EnvironmentSetup:
    def __init__(self, venv_name="korean-llm-env"):
        self.venv_name = venv_name
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / venv_name
        
    def run_command(self, command, description="", check=True):
        """명령어 실행 및 결과 출력"""
        print(f"🔧 {description}")
        print(f"   $ {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            if e.stderr:
                print(f"   stderr: {e.stderr.strip()}")
            return False
    
    def check_prerequisites(self):
        """사전 요구사항 확인"""
        print("🔍 사전 요구사항 확인 중...")
        
        # Python 버전 확인
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            print(f"❌ Python 3.8+ 필요 (현재: {python_version.major}.{python_version.minor})")
            return False
        
        # pip 확인
        if not self.run_command("pip --version", "pip 설치 확인", check=False):
            print("❌ pip가 설치되지 않음")
            return False
        
        # git 확인 (선택사항)
        git_available = self.run_command("git --version", "git 설치 확인", check=False)
        if not git_available:
            print("⚠️  git 미설치 - 일부 패키지 설치가 제한될 수 있음")
        
        return True
    
    def create_virtual_environment(self):
        """가상환경 생성"""
        if self.venv_path.exists():
            print(f"⚠️  가상환경이 이미 존재함: {self.venv_path}")
            response = input("기존 환경을 삭제하고 재생성하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                self.run_command(f"rm -rf {self.venv_path}", "기존 가상환경 삭제")
            else:
                print("기존 환경을 유지합니다.")
                return True
        
        return self.run_command(f"python -m venv {self.venv_name}", "가상환경 생성")
    
    def get_activation_command(self):
        """OS별 가상환경 활성화 명령어 반환"""
        if os.name == 'nt':  # Windows
            return f"{self.venv_path}\\Scripts\\activate"
        else:  # Linux/macOS
            return f"source {self.venv_path}/bin/activate"
    
    def install_requirements(self):
        """필수 패키지 설치"""
        # pip 업그레이드
        pip_cmd = f"{self.venv_path}/bin/pip" if os.name != 'nt' else f"{self.venv_path}\\Scripts\\pip"
        
        if not self.run_command(f"{pip_cmd} install --upgrade pip", "pip 업그레이드"):
            return False
        
        # requirements.txt가 있으면 사용, 없으면 개별 설치
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            return self.run_command(f"{pip_cmd} install -r requirements.txt", 
                                  "requirements.txt에서 패키지 설치")
        else:
            # 개별 패키지 설치
            packages = [
                "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "\"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
                "\"xformers<0.0.26\"",
                "trl peft accelerate bitsandbytes",
                "transformers datasets",
                "requests beautifulsoup4",
                "psutil"
            ]
            
            for package in packages:
                if not self.run_command(f"{pip_cmd} install {package}", 
                                      f"패키지 설치: {package.split()[0]}"):
                    print(f"⚠️  {package} 설치 실패 - 계속 진행")
        
        return True
    
    def create_project_structure(self):
        """프로젝트 디렉토리 구조 생성"""
        directories = [
            "data",
            "models", 
            "scripts",
            "logs",
            "checkpoints"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                print(f"📁 디렉토리 생성: {directory}")
    
    def create_activation_script(self):
        """환경 활성화 스크립트 생성"""
        if os.name == 'nt':  # Windows
            script_content = f"""@echo off
echo Activating Korean LLM environment...
call {self.venv_path}\\Scripts\\activate.bat
echo Environment activated. Run 'python check_environment.py' to validate setup.
"""
            script_path = self.project_root / "activate_env.bat"
        else:  # Linux/macOS
            script_content = f"""#!/bin/bash
echo "Activating Korean LLM environment..."
source {self.venv_path}/bin/activate
echo "Environment activated. Run 'python check_environment.py' to validate setup."
"""
            script_path = self.project_root / "activate_env.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        
        print(f"📄 활성화 스크립트 생성: {script_path.name}")
    
    def run_environment_check(self):
        """환경 검증 실행"""
        python_cmd = f"{self.venv_path}/bin/python" if os.name != 'nt' else f"{self.venv_path}\\Scripts\\python"
        
        check_script = self.project_root / "check_environment.py"
        if check_script.exists():
            print("\n🔍 환경 검증 실행 중...")
            return self.run_command(f"{python_cmd} check_environment.py", 
                                  "환경 검증", check=False)
        else:
            print("⚠️  check_environment.py를 찾을 수 없습니다.")
            return False
    
    def setup(self, skip_check=False):
        """전체 환경 설정 실행"""
        print("🚀 Ollama GPT-OSS-20B Unsloth 환경 설정 시작\n")
        
        # 사전 요구사항 확인
        if not self.check_prerequisites():
            print("❌ 사전 요구사항이 충족되지 않았습니다.")
            return False
        
        # 가상환경 생성
        if not self.create_virtual_environment():
            print("❌ 가상환경 생성 실패")
            return False
        
        # 패키지 설치
        if not self.install_requirements():
            print("❌ 패키지 설치 실패")
            return False
        
        # 프로젝트 구조 생성
        self.create_project_structure()
        
        # 활성화 스크립트 생성
        self.create_activation_script()
        
        # 환경 검증 (선택적)
        if not skip_check:
            self.run_environment_check()
        
        print(f"\n🎉 환경 설정 완료!")
        print(f"💡 다음 명령어로 환경을 활성화하세요:")
        print(f"   {self.get_activation_command()}")
        
        return True

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Ollama GPT-OSS-20B 환경 설정")
    parser.add_argument("--venv-name", default="korean-llm-env", 
                       help="가상환경 이름 (기본값: korean-llm-env)")
    parser.add_argument("--skip-check", action="store_true",
                       help="환경 검증 단계 건너뛰기")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(args.venv_name)
    success = setup.setup(skip_check=args.skip_check)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()