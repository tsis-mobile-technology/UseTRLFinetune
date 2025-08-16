#!/usr/bin/env python3
"""
Environment validation script for Ollama GPT-OSS-20B Unsloth fine-tuning
검증: GPU, CUDA, Python 패키지, 메모리 상태
"""

import sys
import subprocess
import importlib
import platform
import psutil
import torch
import json
from pathlib import Path

class EnvironmentChecker:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        
    def log_check(self, name, status, message="", details=None):
        """검사 결과 로깅"""
        result = {
            "check": name,
            "status": status,
            "message": message,
            "details": details
        }
        self.checks.append(result)
        
        if status == "PASS":
            print(f"✅ {name}: {message}")
            self.passed += 1
        elif status == "FAIL":
            print(f"❌ {name}: {message}")
            self.failed += 1
        else:
            print(f"⚠️  {name}: {message}")
    
    def check_python_version(self):
        """Python 버전 검사"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log_check("Python Version", "PASS", 
                         f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.log_check("Python Version", "FAIL", 
                         f"Python {version.major}.{version.minor}.{version.micro} (Required: 3.8+)")
    
    def check_gpu_availability(self):
        """GPU 및 CUDA 가용성 검사"""
        try:
            # NVIDIA-SMI 실행
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    name, total, used, free = line.split(', ')
                    gpu_info.append({
                        "name": name,
                        "total_memory": int(total),
                        "used_memory": int(used),
                        "free_memory": int(free)
                    })
            
            if gpu_info:
                primary_gpu = gpu_info[0]
                self.log_check("GPU Detection", "PASS", 
                             f"{primary_gpu['name']} - {primary_gpu['free_memory']}MB free",
                             gpu_info)
            else:
                self.log_check("GPU Detection", "FAIL", "No GPU detected")
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_check("GPU Detection", "FAIL", "NVIDIA drivers not found")
    
    def check_cuda_pytorch(self):
        """PyTorch CUDA 지원 검사"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                cuda_version = torch.version.cuda
                
                self.log_check("PyTorch CUDA", "PASS", 
                             f"CUDA {cuda_version}, {device_count} device(s), Current: {device_name}")
            else:
                self.log_check("PyTorch CUDA", "FAIL", "CUDA not available in PyTorch")
        except Exception as e:
            self.log_check("PyTorch CUDA", "FAIL", f"Error checking CUDA: {str(e)}")
    
    def check_memory_requirements(self):
        """시스템 메모리 요구사항 검사"""
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        if total_gb >= 16:
            status = "PASS"
            message = f"{total_gb:.1f}GB total, {available_gb:.1f}GB available"
        elif total_gb >= 8:
            status = "WARN"
            message = f"{total_gb:.1f}GB total (minimum), {available_gb:.1f}GB available"
        else:
            status = "FAIL"
            message = f"{total_gb:.1f}GB total (insufficient), {available_gb:.1f}GB available"
        
        self.log_check("System Memory", status, message)
    
    def check_required_packages(self):
        """필수 패키지 설치 확인"""
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Hugging Face Transformers',
            'peft': 'Parameter-Efficient Fine-Tuning',
            'trl': 'Transformer Reinforcement Learning',
            'bitsandbytes': 'Quantization library',
            'accelerate': 'Accelerate library',
            'datasets': 'Datasets library',
            'requests': 'HTTP library',
            'beautifulsoup4': 'Web scraping library'
        }
        
        for package, description in required_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.log_check(f"Package: {package}", "PASS", f"{description} v{version}")
            except ImportError:
                self.log_check(f"Package: {package}", "FAIL", f"{description} not installed")
    
    def check_disk_space(self):
        """디스크 공간 확인"""
        current_path = Path.cwd()
        disk_usage = psutil.disk_usage(current_path)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb >= 50:
            status = "PASS"
            message = f"{free_gb:.1f}GB free space available"
        elif free_gb >= 20:
            status = "WARN"
            message = f"{free_gb:.1f}GB free space (may be tight)"
        else:
            status = "FAIL"
            message = f"{free_gb:.1f}GB free space (insufficient)"
        
        self.log_check("Disk Space", status, message)
    
    def check_virtual_environment(self):
        """가상환경 활성화 상태 확인"""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_path = sys.prefix
            self.log_check("Virtual Environment", "PASS", f"Active: {venv_path}")
        else:
            self.log_check("Virtual Environment", "WARN", "Not in virtual environment")
    
    def save_report(self, filename="environment_check_report.json"):
        """검사 결과를 JSON 파일로 저장"""
        report = {
            "timestamp": subprocess.check_output(['date'], text=True).strip(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture()[0]
            },
            "summary": {
                "total_checks": len(self.checks),
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": f"{(self.passed/len(self.checks)*100):.1f}%"
            },
            "checks": self.checks
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 상세 리포트 저장됨: {filename}")
    
    def run_all_checks(self):
        """모든 환경 검사 실행"""
        print("🔍 Ollama GPT-OSS-20B Unsloth 환경 검사 시작...\n")
        
        self.check_python_version()
        self.check_virtual_environment()
        self.check_gpu_availability()
        self.check_cuda_pytorch()
        self.check_memory_requirements()
        self.check_disk_space()
        self.check_required_packages()
        
        print(f"\n📊 검사 완료: {self.passed}개 통과, {self.failed}개 실패")
        
        if self.failed == 0:
            print("🎉 모든 환경 요구사항이 충족되었습니다!")
            return True
        else:
            print("⚠️  일부 요구사항이 충족되지 않았습니다. 위 내용을 확인해주세요.")
            return False

def main():
    """메인 실행 함수"""
    checker = EnvironmentChecker()
    success = checker.run_all_checks()
    checker.save_report()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()