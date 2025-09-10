"""
Setup script for HRM Autonomous Agent
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hrm-autonomous-agent",
    version="1.0.0",
    author="HRM Development Team",
    author_email="dev@hrm-agent.com",
    description="Autonomous agent system for continuous training of Hierarchical Reasoning Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/hrm-autonomous-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
        "advanced": [
            "faiss-cpu>=1.7.4",
            "sentence-transformers>=2.2.0",
            "openai>=0.27.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hrm-agent=main:main",
            "hrm-demo=run_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hrm_autonomous_agent": [
            "config/*.json",
            "web_interface/templates/*.html",
            "web_interface/static/*",
        ],
    },
)