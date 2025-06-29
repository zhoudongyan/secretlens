# SecretLens Unified Installation Script for Windows
# Usage: .\install.ps1 [-Mode quick|full] [-Method pip|uv] [-SkipTools] [-SkipKeys] [-Help]

param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",
    
    [ValidateSet("pip", "uv")]
    [string]$Method = "",
    
    [switch]$SkipTools,
    [switch]$SkipKeys,
    [switch]$Help
)

# Configuration
$PythonMinVersion = "3.8"
$ErrorActionPreference = "Stop"

# Output functions with emojis
function Write-Info {
    param([string]$Message)
    Write-Host "📍 $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "🚀 $Message" -ForegroundColor Cyan
}

# Show help
function Show-Help {
    Write-Host "🔍 SecretLens Unified Installation Script" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [options]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Mode <quick|full>    Installation mode (default: quick)"
    Write-Host "  -Method <pip|uv>      Installation method (auto-detect if not specified)"
    Write-Host "  -SkipTools           Skip external tools installation (Gitleaks)"
    Write-Host "  -SkipKeys            Skip API key setup"
    Write-Host "  -Help                Show this help message"
    Write-Host ""
    Write-Host "Modes:"
    Write-Host "  quick    Fast setup with Python environment only"
    Write-Host "  full     Complete setup with tools and configuration"
    Write-Host ""
    Write-Host "Methods:"
    Write-Host "  pip      Standard pip + venv setup"
    Write-Host "  uv       Fast uv-based setup (recommended)"
    Write-Host ""
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check Python installation
function Test-Python {
    Write-Info "Checking Python installation..."
    
    $pythonCmd = $null
    if (Test-Command "python") {
        $pythonCmd = "python"
    }
    elseif (Test-Command "python3") {
        $pythonCmd = "python3"
    }
    elseif (Test-Command "py") {
        $pythonCmd = "py"
    }
    else {
        Write-Error "Python is not installed. Please install Python $PythonMinVersion or higher from https://python.org"
        Write-Host "Or use: winget install Python.Python.3.11"
        exit 1
    }
    
    # Check Python version
    try {
        $versionOutput = & $pythonCmd --version 2>&1
        $version = ($versionOutput -split " ")[1]
        $versionParts = $version -split "\."
        $major = [int]$versionParts[0]
        $minor = [int]$versionParts[1]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Error "Python $version is installed, but SecretLens requires Python $PythonMinVersion or higher."
            exit 1
        }
        
        Write-Success "Found Python $version"
        return $pythonCmd
    }
    catch {
        Write-Error "Failed to check Python version: $_"
        exit 1
    }
}

# Auto-detect best installation method
function Get-InstallMethod {
    if ([string]::IsNullOrEmpty($Method)) {
        if (Test-Command "uv") {
            $script:Method = "uv"
            Write-Info "Auto-detected method: uv (already installed)"
        }
        else {
            # Interactive selection
            Write-Host ""
            Write-Host "Choose installation method:" -ForegroundColor Yellow
            Write-Host "1) uv (fast, recommended)"
            Write-Host "2) pip + venv (standard)"
            
            do {
                $choice = Read-Host "Enter choice [1-2]"
            } while ($choice -notin @("1", "2"))
            
            switch ($choice) {
                "1" { $script:Method = "uv" }
                "2" { $script:Method = "pip" }
            }
        }
    }
    
    Write-Info "Using installation method: $Method"
}

# Install uv if needed
function Install-Uv {
    if ($Method -eq "uv" -and -not (Test-Command "uv")) {
        Write-Step "Installing uv..."
        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
            # Refresh PATH
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
            
            if (Test-Command "uv") {
                Write-Success "uv installed successfully"
            }
            else {
                Write-Error "Failed to install uv, falling back to pip"
                $script:Method = "pip"
            }
        }
        catch {
            Write-Error "Failed to install uv: $_"
            Write-Info "Falling back to pip method"
            $script:Method = "pip"
        }
    }
}

# Setup Python environment
function Set-PythonEnvironment {
    param([string]$PythonCmd)
    
    Write-Step "Setting up Python environment with $Method..."
    
    switch ($Method) {
        "uv" {
            uv venv
            .\.venv\Scripts\Activate.ps1
            uv pip install -e ".[dev,llm]"
            $script:VenvActivateCmd = ".\.venv\Scripts\Activate.ps1"
        }
        "pip" {
            & $PythonCmd -m venv venv
            .\venv\Scripts\Activate.ps1
            python -m pip install --upgrade pip
            pip install -e ".[dev,llm]"
            $script:VenvActivateCmd = ".\venv\Scripts\Activate.ps1"
        }
        default {
            Write-Error "Unknown installation method: $Method"
            exit 1
        }
    }
    
    Write-Success "Python environment setup complete!"
}

# Install external tools (full mode only)
function Install-Tools {
    if ($Mode -eq "full" -and -not $SkipTools) {
        Write-Step "Installing external tools..."
        
        # Install Gitleaks
        if (Test-Command "gitleaks") {
            Write-Warning "Gitleaks is already installed"
            & gitleaks version
        }
        else {
            Write-Info "Installing Gitleaks..."
            
            # Try winget first
            if (Test-Command "winget") {
                try {
                    & winget install gitleaks --silent
                    Write-Success "Gitleaks installed via winget"
                }
                catch {
                    Write-Warning "winget installation failed, trying manual installation..."
                    Install-GitleaksManual
                }
            }
            else {
                Install-GitleaksManual
            }
            
            if (Test-Command "gitleaks") {
                Write-Success "Gitleaks installed successfully"
            }
        }
    }
}

# Install Gitleaks manually
function Install-GitleaksManual {
    try {
        $gitleaksUrl = "https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks-windows-amd64.zip"
        $tempPath = "$env:TEMP\gitleaks.zip"
        $extractPath = "$env:TEMP\gitleaks"
        
        # Download
        Invoke-WebRequest -Uri $gitleaksUrl -OutFile $tempPath
        
        # Extract
        Expand-Archive -Path $tempPath -DestinationPath $extractPath -Force
        
        # Move to Program Files
        $installPath = "$env:ProgramFiles\Gitleaks"
        if (-not (Test-Path $installPath)) {
            New-Item -ItemType Directory -Path $installPath -Force | Out-Null
        }
        
        Copy-Item "$extractPath\gitleaks.exe" "$installPath\gitleaks.exe" -Force
        
        # Add to PATH
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if ($currentPath -notlike "*$installPath*") {
            [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$installPath", "Machine")
            $env:PATH += ";$installPath"
        }
        
        # Cleanup
        Remove-Item $tempPath -Force
        Remove-Item $extractPath -Recurse -Force
        
        Write-Success "Gitleaks installed manually"
    }
    catch {
        Write-Error "Failed to install Gitleaks: $_"
        Write-Host "Please download manually from: https://github.com/gitleaks/gitleaks/releases"
    }
}

# Setup configuration (full mode only)
function Set-Configuration {
    if ($Mode -eq "full" -and -not $SkipKeys) {
        Write-Step "Setting up configuration..."
        
        if (-not (Test-Path ".env")) {
            if (Test-Path ".env.example") {
                Copy-Item ".env.example" ".env"
                Write-Info "Created .env file from .env.example"
                Write-Warning "Please edit .env file to add your API keys"
            }
            else {
                Write-Warning "No .env.example found. Please create .env file manually"
            }
        }
        else {
            Write-Info ".env file already exists"
        }
    }
}

# Verify installation
function Test-Installation {
    Write-Step "Verifying installation..."
    
    try {
        $null = & secretlens --help 2>&1
        Write-Success "SecretLens is installed and available"
        Write-Success "SecretLens help command works"
    }
    catch {
        Write-Error "SecretLens command not found. Installation may have failed."
        exit 1
    }
}

# Show completion message
function Show-Completion {
    Write-Host ""
    Write-Host "🎉 SecretLens installation complete!" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
    Write-Host ""
    Write-Info "To activate the environment: $VenvActivateCmd"
    Write-Info "To run SecretLens: secretlens --help"
    
    if ($Mode -eq "full") {
        Write-Info "📚 Check README.md for usage examples"
        Write-Info "🔧 Edit .env file to add your API keys"
    }
    
    Write-Host ""
    Write-Info "Available commands:"
    Write-Host "  secretlens scan <path>       - Scan for secrets"
    Write-Host "  secretlens enhance <path>    - Enhanced analysis with LLM"
    Write-Host "  python demo.py               - Run demo script"
}

# Main installation function
function Main {
    Write-Host "🔍 SecretLens Unified Installation Script" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($Help) {
        Show-Help
        exit 0
    }
    
    Write-Info "Installation mode: $Mode"
    
    $pythonCmd = Test-Python
    Get-InstallMethod
    Install-Uv
    Set-PythonEnvironment -PythonCmd $pythonCmd
    
    if ($Mode -eq "full") {
        Install-Tools
        Set-Configuration
    }
    
    Test-Installation
    Show-Completion
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Installation failed: $_"
    exit 1
} 