#!/bin/bash

# SecretLens Unified Installation Script
# Supports macOS, Linux, and Windows (via Git Bash/WSL)
# Usage: ./install.sh [--mode=quick|full] [--method=pip|uv] [--skip-tools] [--skip-keys]

set -e # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
MODE="quick"  # Default mode
METHOD=""     # Auto-detect
SKIP_TOOLS=false
SKIP_KEYS=false

# Print colored output
print_info() {
    echo -e "${BLUE}📍${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_step() {
    echo -e "${CYAN}🚀${NC} $1"
}

# Parse command line arguments
parse_args() {
    for arg in "$@"; do
        case $arg in
            --mode=*)
                MODE="${arg#*=}"
                ;;
            --method=*)
                METHOD="${arg#*=}"
                ;;
            --skip-tools)
                SKIP_TOOLS=true
                ;;
            --skip-keys)
                SKIP_KEYS=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_help() {
    echo "🔍 SecretLens Unified Installation Script"
    echo "========================================"
    echo ""
    echo "Usage: ./install.sh [options]"
    echo ""
    echo "Options:"
    echo "  --mode=quick|full     Installation mode (default: quick)"
    echo "  --method=pip|uv       Installation method (auto-detect if not specified)"
    echo "  --skip-tools          Skip external tools installation (Gitleaks)"
    echo "  --skip-keys           Skip API key setup"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Modes:"
    echo "  quick    Fast setup with Python environment only"
    echo "  full     Complete setup with tools and configuration"
    echo ""
    echo "Methods:"
    echo "  pip      Standard pip + venv setup"
    echo "  uv       Fast uv-based setup (recommended)"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_info "Checking Python installation..."

    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python $PYTHON_MIN_VERSION or higher."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python $PYTHON_VERSION is installed, but SecretLens requires Python $PYTHON_MIN_VERSION or higher."
        exit 1
    fi

    print_success "Found Python $PYTHON_VERSION"
}

# Auto-detect best installation method
detect_method() {
    if [ -z "$METHOD" ]; then
        if command_exists uv; then
            METHOD="uv"
            print_info "Auto-detected method: uv (already installed)"
        else
            # Check if we should install uv
            echo ""
            echo "Choose installation method:"
            echo "1) uv (fast, recommended)"
            echo "2) pip + venv (standard)"
            
            read -p "Enter choice [1-2]: " choice
            case $choice in
                1) METHOD="uv" ;;
                2) METHOD="pip" ;;
                *) 
                    print_warning "Invalid choice, defaulting to pip"
                    METHOD="pip"
                    ;;
            esac
        fi
    fi
    
    print_info "Using installation method: $METHOD"
}

# Install uv if needed
install_uv() {
    if [ "$METHOD" = "uv" ] && ! command_exists uv; then
        print_step "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command_exists uv; then
            print_success "uv installed successfully"
        else
            print_error "Failed to install uv, falling back to pip"
            METHOD="pip"
        fi
    fi
}

# Setup Python environment
setup_python_env() {
    print_step "Setting up Python environment with $METHOD..."
    
    case $METHOD in
        "uv")
            uv venv
            source .venv/bin/activate
            uv pip install -e ".[dev,llm]"
            VENV_ACTIVATE_CMD="source .venv/bin/activate"
            ;;
        "pip")
            $PYTHON_CMD -m venv venv
            source venv/bin/activate
            pip install --upgrade pip
            pip install -e ".[dev,llm]"
            VENV_ACTIVATE_CMD="source venv/bin/activate"
            ;;
        *)
            print_error "Unknown installation method: $METHOD"
            exit 1
            ;;
    esac
    
    print_success "Python environment setup complete!"
}

# Install external tools (full mode only)
install_tools() {
    if [ "$MODE" = "full" ] && [ "$SKIP_TOOLS" = false ]; then
        print_step "Installing external tools..."
        
        # Install Gitleaks
        if command_exists gitleaks; then
            print_warning "Gitleaks is already installed"
            gitleaks version
        else
            print_info "Installing Gitleaks..."
            
            # Detect OS for Gitleaks installation
            if [[ "$OSTYPE" == "darwin"* ]]; then
                if command_exists brew; then
                    brew install gitleaks
                else
                    install_gitleaks_manual "darwin" "amd64"
                fi
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                install_gitleaks_manual "linux" "amd64"
            else
                print_warning "Please install Gitleaks manually: https://github.com/gitleaks/gitleaks/releases"
            fi
            
            if command_exists gitleaks; then
                print_success "Gitleaks installed successfully"
            fi
        fi
    fi
}

# Install Gitleaks manually
install_gitleaks_manual() {
    local os_type=$1
    local arch=$2
    
    GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks-${os_type}-${arch}.tar.gz"
    
    curl -sSfL "$GITLEAKS_URL" | tar xz
    sudo mv gitleaks /usr/local/bin/
    sudo chmod +x /usr/local/bin/gitleaks
}

# Setup configuration (full mode only)
setup_config() {
    if [ "$MODE" = "full" ] && [ "$SKIP_KEYS" = false ]; then
        print_step "Setting up configuration..."
        
        if [ ! -f ".env" ]; then
            if [ -f ".env.example" ]; then
                cp .env.example .env
                print_info "Created .env file from .env.example"
                print_warning "Please edit .env file to add your API keys"
            else
                print_warning "No .env.example found. Please create .env file manually"
            fi
        else
            print_info ".env file already exists"
        fi
    fi
}

# Verify installation
verify_installation() {
    print_step "Verifying installation..."
    
    if command -v secretlens >/dev/null 2>&1; then
        print_success "SecretLens is installed and available"
        secretlens --help >/dev/null 2>&1 && print_success "SecretLens help command works"
    else
        print_error "SecretLens command not found. Installation may have failed."
        exit 1
    fi
}

# Show completion message
show_completion() {
    echo ""
    echo "🎉 SecretLens installation complete!"
    echo "=================================="
    echo ""
    print_info "To activate the environment: $VENV_ACTIVATE_CMD"
    print_info "To run SecretLens: secretlens --help"
    
    if [ "$MODE" = "full" ]; then
        print_info "📚 Check README.md for usage examples"
        print_info "🔧 Edit .env file to add your API keys"
    fi
    
    echo ""
    print_info "Available commands:"
    echo "  secretlens scan <path>       - Scan for secrets"
    echo "  secretlens enhance <path>    - Enhanced analysis with LLM"
    echo "  python demo.py               - Run demo script"
    
    if command_exists make; then
        echo "  make help                    - Show development commands"
    fi
}

# Main installation function
main() {
    echo "🔍 SecretLens Unified Installation Script"
    echo "========================================"
    echo ""
    
    parse_args "$@"
    
    print_info "Installation mode: $MODE"
    
    check_python
    detect_method
    install_uv
    setup_python_env
    
    if [ "$MODE" = "full" ]; then
        install_tools
        setup_config
    fi
    
    verify_installation
    show_completion
}

# Run main function
main "$@"
