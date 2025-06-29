# SecretLens Installation Guide

This guide provides multiple ways to install and set up SecretLens for development and usage.

## Prerequisites

- Python 3.8 or later
- Git (for cloning the repository)

## Quick Start

### Option 1: Unified Installation Script (Recommended)

**Unix/macOS:**
```bash
# Quick setup (Python environment only)
chmod +x install.sh
./install.sh

# Full setup (includes tools and configuration)
./install.sh --mode=full

# Use uv for faster installation
./install.sh --method=uv

# Show all options
./install.sh --help
```

**Windows PowerShell:**
```powershell
# Quick setup (Python environment only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1

# Full setup (includes tools and configuration)  
.\install.ps1 -Mode full

# Use uv for faster installation
.\install.ps1 -Method uv

# Show all options
.\install.ps1 -Help
```

### Option 2: Manual Setup with uv (Fast)

1. Install uv (if not already installed):

   ```bash
   # Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Set up the project:

   ```bash
   uv venv
   source .venv/bin/activate  # Unix/macOS
   # or
   .venv\Scripts\activate     # Windows

   uv pip install -e ".[dev,llm]"
   ```

### Option 3: Traditional pip + venv

```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -e ".[dev,llm]"
```

### Option 4: Manual Installation + Development Tools

```bash
# Manual setup with pip + venv
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev,llm]"

# Set up development tools (optional)
make dev-setup    # Install additional dev dependencies
make pre-commit   # Set up pre-commit hooks
```

## Installation Options

### Core Installation (minimal dependencies)

```bash
pip install -e .
```

### Development Installation (includes dev tools)

```bash
pip install -e ".[dev]"
```

### Full Installation (includes LLM APIs)

```bash
pip install -e ".[dev,llm]"
```

### LLM Only Installation

```bash
pip install -e ".[llm]"
```

## Verification

After installation, verify that SecretLens is working:

```bash
secretlens --help
```

Or run the demo:

```bash
python demo.py
```

## Development Setup

For development work, use the full installation with development dependencies:

```bash
# Install with development dependencies
pip install -e ".[dev,llm]"

# Set up pre-commit hooks (optional)
pre-commit install

# Set up development tools (optional)
make dev-setup    # Install additional dev dependencies  
make pre-commit   # Set up pre-commit hooks

# Verify development setup
make check        # Run linting and tests
```

## Development Tools (Makefile)

After installation, if you have `make` available, you can use development commands:

- `make help` - Show all available development commands
- `make dev-setup` - Install dev dependencies in current environment
- `make test` - Run tests
- `make lint` - Run linting checks (mypy, black, isort)  
- `make format` - Format code with black and isort
- `make check` - Run all checks (lint + test)
- `make pre-commit` - Setup pre-commit hooks
- `make clean` - Clean build artifacts and cache
- `make run` - Run SecretLens CLI help
- `make demo` - Run demo script

**Note:** For installation, use the unified installation script (`./install.sh` or `.\install.ps1`). Makefile is focused on development workflow.

## Environment Management

### Using pyenv (recommended for Python version management)

```bash
# Install Python 3.11 (recommended version)
pyenv install 3.11.0
pyenv local 3.11.0
```

The `.python-version` file in the project root will automatically use Python 3.11.0 if you have pyenv installed.

### Using conda/mamba

```bash
conda create -n secretlens python=3.11
conda activate secretlens
pip install -e ".[dev,llm]"
```

## Troubleshooting

### Common Issues

1. **Permission errors on Unix/macOS:**

   ```bash
   chmod +x install.sh
   ```

2. **PowerShell execution policy error:**

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Python version issues:**

   - Ensure you're using Python 3.8+
   - Check with `python --version`
   - Use pyenv or conda to manage Python versions

4. **uv installation fails:**
   - Make sure you have curl installed
   - Check your internet connection
   - Try manual installation from https://github.com/astral-sh/uv

### Getting Help

- Check the [README.md](README.md) for usage examples
- Check the [technical-guide.md](technical-guide.md) for technical details
- Open an issue on GitHub if you encounter problems

## Next Steps

After installation:

1. Read the [README.md](README.md) for usage examples
2. Try the demo: `python demo.py`
3. Check available commands: `secretlens --help`
4. Configure your LLM settings (see Configuration section below)

## Configuration

### Environment Variables Setup

SecretLens supports configuration via environment variables or `.env` files.

#### Option 1: Using .env File (Recommended)

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your settings:

   ```bash
   # LLM API Keys (choose one or more)
   OPENAI_API_KEY=your-openai-api-key-here
   ANTHROPIC_API_KEY=your-anthropic-api-key-here

   # LLM Configuration
   LLM_MODEL=gpt-4o-mini

   # Custom Base URLs (optional, for proxies or compatible APIs)
   LLM_BASE_URL=https://your-proxy.com/v1

   # Provider-specific base URLs (optional, for advanced configurations)
   # OPENAI_BASE_URL=https://your-openai-proxy.com/v1
   # ANTHROPIC_BASE_URL=https://your-anthropic-proxy.com
   ```

3. The `.env` file will be automatically loaded when running SecretLens.

#### Option 2: System Environment Variables

```bash
# Unix/macOS
export OPENAI_API_KEY="your-key-here"
export LLM_BASE_URL="https://your-proxy.com/v1"

# Windows
set OPENAI_API_KEY=your-key-here
set LLM_BASE_URL=https://your-proxy.com/v1
```

#### Option 3: Command Line Arguments

```bash
secretlens enhance ./my-project --api-key "your-key-here" --llm-base-url "https://your-proxy.com/v1"
```

### Configuration Priority

SecretLens uses the following priority order for configuration:

1. Command line arguments (highest priority)
2. `.env` file variables
3. System environment variables
4. Default values (lowest priority)

### Available Configuration Options

| Environment Variable | CLI Option       | Description                                                   |
| -------------------- | ---------------- | ------------------------------------------------------------- |
| `OPENAI_API_KEY`     | `--api-key`      | OpenAI API key                                                |
| `ANTHROPIC_API_KEY`  | `--api-key`      | Anthropic API key                                             |
| `LLM_MODEL`          | `--model`        | LLM model to use (e.g., gpt-4o-mini, claude-3-haiku-20240307) |
| `LLM_BASE_URL`       | `--llm-base-url` | Custom LLM API base URL                                       |
| `OPENAI_BASE_URL`    | -                | OpenAI-specific base URL                                      |
| `ANTHROPIC_BASE_URL` | -                | Anthropic-specific base URL                                   |

### Security Notes

- Never commit your `.env` file to version control
- The `.env` file is already included in `.gitignore`
- Use `.env.example` as a template for sharing configurations
- For production deployments, prefer system environment variables over `.env` files
