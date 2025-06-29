# SecretLens Development Makefile
# For installation, use: ./install.sh (Unix/macOS) or .\install.ps1 (Windows)

.PHONY: help dev-setup clean test lint format check run demo pre-commit ci

# Default target
help:
	@echo "🔧 SecretLens Development Makefile"
	@echo "=================================="
	@echo ""
	@echo "📦 Installation:"
	@echo "  Use ./install.sh (Unix/macOS) or .\\install.ps1 (Windows)"
	@echo "  For development setup: ./install.sh --mode=full"
	@echo ""
	@echo "🛠️  Development Commands:"
	@echo "  dev-setup    - Install dev dependencies in current environment"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks (mypy, black, isort)"
	@echo "  format       - Format code with black and isort"
	@echo "  check        - Run all checks (lint + test)"
	@echo "  pre-commit   - Setup pre-commit hooks"
	@echo "  clean        - Clean build artifacts and cache"
	@echo ""
	@echo "🚀 Runtime Commands:"
	@echo "  run          - Run secretlens CLI help"
	@echo "  demo         - Run demo script"
	@echo ""
	@echo "🔄 CI/Development:"
	@echo "  ci           - Run CI checks (lint + test + clean)"

# Development setup (assumes virtual environment is already activated)
dev-setup:
	@echo "🔧 Installing development dependencies..."
	@echo "Note: Make sure your virtual environment is activated"
	pip install -e ".[dev,llm]"
	@echo "✅ Development dependencies installed"

# Pre-commit hooks setup
pre-commit:
	@echo "🔗 Setting up pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Code quality targets
test:
	@echo "🧪 Running tests..."
	pytest

lint:
	@echo "🔍 Running linting checks..."
	@echo "  → mypy type checking..."
	mypy secretlens/
	@echo "  → black code style checking..."
	black --check secretlens/
	@echo "  → isort import sorting checking..."
	isort --check-only secretlens/

format:
	@echo "✨ Formatting code..."
	@echo "  → black code formatting..."
	black secretlens/
	@echo "  → isort import sorting..."
	isort secretlens/
	@echo "✅ Code formatted"

check: lint test
	@echo "✅ All checks passed!"

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleanup complete"

# Runtime targets
run:
	@echo "🚀 Running SecretLens CLI..."
	secretlens --help

demo:
	@echo "🎬 Running demo script..."
	python demo.py

# CI target for automated testing
ci: clean lint test
	@echo "🤖 CI checks complete!"

# Development workflow helpers
dev-check: format check
	@echo "👨‍💻 Development check complete!"

dev-clean: clean dev-setup
	@echo "🔄 Development environment refreshed!" 