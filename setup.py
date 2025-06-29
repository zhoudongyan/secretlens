"""
Setup script for SecretLens
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version from package
version = "0.1.0"

setup(
    name="secretlens",
    version=version,
    author="SecretLens Team",
    author_email="contact@secretlens.dev",
    description="LLM-Enhanced Secret Detection Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/secretlens/secretlens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "typing-extensions>=4.0.0",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "regex>=2023.0.0",
        "GitPython>=3.1.0",
        "pathspec>=0.11.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "secretlens=secretlens.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
