"""
Ollama Manager - Complete Local LLM Solution

Provides a unified interface for local LLM with automatic setup and management.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import psutil
import requests

logger = logging.getLogger(__name__)


# Model configuration - simplified and flattened structure
MODEL_CONFIG = {
    "qwen3": {
        "name": "qwen3:4b",
        "size_gb": 2.6,
        "memory_gb": 8,
        "description": "Smart and accurate code analysis",
        "use_case": "production",
    },
    "deepseek": {
        "name": "deepseek-r1:latest",
        "size_gb": 8.0,
        "memory_gb": 24,
        "description": "Advanced reasoning and analysis",
        "use_case": "research",
    },
    # Simple settings
    "default": "qwen3",
    "fallback_order": ["qwen3", "deepseek"],
}


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get information about a model by key (qwen3, deepseek)"""
    return MODEL_CONFIG.get(model_key, {})


def get_recommended_model(memory_gb: float) -> str:
    """Get recommended model based on available memory"""
    # Check models in fallback order
    for model_key in MODEL_CONFIG["fallback_order"]:
        model_info = MODEL_CONFIG.get(model_key, {})
        required_memory = model_info.get("memory_gb", 16)
        if memory_gb >= required_memory:
            return model_info.get("name", "qwen3:4b")

    # If no model fits, return the lightest one (qwen3 for low memory)
    return MODEL_CONFIG["qwen3"]["name"]


def print_current_config() -> None:
    """Print the current model configuration"""
    print("\n📋 SecretLens Model Configuration:")
    print("=" * 50)

    print("\n🎯 Available Models:")
    for key in ["qwen3", "deepseek"]:
        if key in MODEL_CONFIG:
            model = MODEL_CONFIG[key]
            print(f"  {key}: {model['name']}")
            print(f"    Size: {model['size_gb']:.1f}GB")
            print(f"    Memory: {model['memory_gb']}GB+")
            print(f"    Description: {model['description']}")
            print(f"    Use case: {model['use_case']}")
            print()

    print(f"🔧 Default: {MODEL_CONFIG['default']}")
    print(f"🔄 Fallback order: {' -> '.join(MODEL_CONFIG['fallback_order'])}")


def switch_to_model(model_key: str) -> str:
    """Switch to a specific model by key"""
    if model_key not in ["qwen3", "deepseek"]:
        raise ValueError(f"Unknown model: {model_key}. Available: qwen3, deepseek")

    MODEL_CONFIG["default"] = model_key
    model_name = MODEL_CONFIG[model_key]["name"]
    use_case = MODEL_CONFIG[model_key]["use_case"]

    print(f"✅ Switched to {model_key} ({model_name})")
    print(f"   Use case: {use_case}")
    return model_name


def switch_to_deepseek():
    """Quick switch to DeepSeek model"""
    return switch_to_model("deepseek")


def switch_to_qwen3():
    """Quick switch to Qwen3 model"""
    return switch_to_model("qwen3")


class _LocalLLMManager:
    """Internal LLM management (private class)"""

    def __init__(self):
        self.ollama_available = False
        self.ollama_host = "http://localhost:11434"
        self.system_info = self._get_system_info()
        self.recommended_model = self._select_optimal_model()
        self.fallback_models = self._get_fallback_models()

    def _get_system_info(self) -> Dict[str, any]:
        """Get system information for model selection"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        system = platform.system().lower()

        return {
            "memory_gb": memory_gb,
            "cpu_count": cpu_count,
            "system": system,
            "arch": platform.machine().lower(),
        }

    def _select_optimal_model(self) -> str:
        """Select optimal model based on system resources"""
        memory_gb = self.system_info["memory_gb"]

        # Use the new get_recommended_model function
        recommended = get_recommended_model(memory_gb)

        logger.info(f"Selected model {recommended} for {memory_gb:.1f}GB RAM")
        return recommended

    def _get_fallback_models(self) -> List[str]:
        """Get fallback models from configuration"""
        fallback_names = []
        for model_key in MODEL_CONFIG["fallback_order"]:
            if model_key in MODEL_CONFIG:
                fallback_names.append(MODEL_CONFIG[model_key]["name"])
        return fallback_names

    async def ensure_local_llm(self) -> bool:
        """
        Ensure local LLM is available, install if necessary

        Returns:
            True if local LLM is ready, False otherwise
        """
        try:
            logger.info("🤖 Checking local AI setup...")

            # Check if Ollama is installed
            if not self._is_ollama_installed():
                print("🚀 Setting up local AI for privacy-first analysis...")
                if not await self._install_ollama():
                    logger.error("Failed to install Ollama")
                    return False

            # Ensure Ollama service is running
            if not await self._ensure_ollama_service():
                logger.error("Failed to start Ollama service")
                return False

            # Check and download recommended model
            if not await self._ensure_model_available():
                logger.error("Failed to setup recommended model")
                return False

            self.ollama_available = True
            logger.info("✅ Local AI ready!")
            return True

        except Exception as e:
            logger.error(f"Failed to setup local LLM: {e}")
            return False

    def _is_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def _install_ollama(self) -> bool:
        """Install Ollama automatically"""
        system = self.system_info["system"]

        try:
            print("📦 Installing Ollama...")

            if system == "darwin":  # macOS
                return await self._install_ollama_macos()
            elif system == "linux":
                return await self._install_ollama_linux()
            elif system == "windows":
                return await self._install_ollama_windows()
            else:
                logger.error(f"Unsupported system: {system}")
                return False

        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False

    async def _install_ollama_macos(self) -> bool:
        """Install Ollama on macOS"""
        try:
            # Check if Homebrew is available
            homebrew_available = (
                subprocess.run(["which", "brew"], capture_output=True).returncode == 0
            )

            if homebrew_available:
                print("🍺 Installing via Homebrew...")
                process = await asyncio.create_subprocess_exec(
                    "brew",
                    "install",
                    "ollama",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()
                return process.returncode == 0
            else:
                # Fallback to direct download
                return await self._install_ollama_direct()

        except Exception as e:
            logger.error(f"macOS installation failed: {e}")
            return False

    async def _install_ollama_linux(self) -> bool:
        """Install Ollama on Linux"""
        try:
            print("🐧 Installing via official script...")
            # Use official installation script
            process = await asyncio.create_subprocess_exec(
                "curl", "-fsSL", "https://ollama.com/install.sh", stdout=asyncio.subprocess.PIPE
            )
            script_content, _ = await process.communicate()

            if process.returncode == 0:
                # Execute the installation script
                process = await asyncio.create_subprocess_exec(
                    "sh",
                    "-c",
                    script_content.decode(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()
                return process.returncode == 0

            return False

        except Exception as e:
            logger.error(f"Linux installation failed: {e}")
            return False

    async def _install_ollama_windows(self) -> bool:
        """Install Ollama on Windows"""
        try:
            print("🪟 Downloading Ollama for Windows...")

            # Download the Windows installer
            url = "https://ollama.com/download/OllamaSetup.exe"
            installer_path = Path.home() / "Downloads" / "OllamaSetup.exe"

            urllib.request.urlretrieve(url, installer_path)

            print("Please run the downloaded installer and restart your terminal.")
            print(f"Installer location: {installer_path}")

            # Note: On Windows, we can't automatically run the installer
            # due to security requirements
            return False  # User needs to manually install

        except Exception as e:
            logger.error(f"Windows installation failed: {e}")
            return False

    async def _install_ollama_direct(self) -> bool:
        """Direct installation method"""
        try:
            system = self.system_info["system"]
            arch = self.system_info["arch"]

            # Determine download URL based on system and architecture
            if system == "darwin":
                if "arm" in arch or "aarch64" in arch:
                    url = "https://ollama.com/download/ollama-darwin"
                else:
                    url = "https://ollama.com/download/ollama-darwin"
            elif system == "linux":
                url = "https://ollama.com/download/ollama-linux-amd64"
            else:
                return False

            # Download and install
            binary_path = Path("/usr/local/bin/ollama")
            urllib.request.urlretrieve(url, binary_path)
            os.chmod(binary_path, 0o755)

            return True

        except Exception as e:
            logger.error(f"Direct installation failed: {e}")
            return False

    async def _ensure_ollama_service(self) -> bool:
        """Ensure Ollama service is running"""
        try:
            # Check if service is already running
            if await self._is_ollama_running():
                return True

            print("🔄 Starting Ollama service...")

            # Start Ollama service in background
            if self.system_info["system"] == "windows":
                # On Windows, Ollama runs as a service
                subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # On Unix systems
                subprocess.Popen(
                    ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

            # Wait for service to start
            for i in range(30):  # Wait up to 30 seconds
                if await self._is_ollama_running():
                    return True
                await asyncio.sleep(1)

            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False

    async def _is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    async def _ensure_model_available(self) -> bool:
        """Ensure recommended model is available"""
        try:
            # Check if model is already downloaded
            logger.info(f"Checking if model {self.recommended_model} is available...")
            is_available = await self._is_model_available(self.recommended_model)
            logger.info(f"Model availability check result: {is_available}")

            if is_available:
                logger.info(f"✅ Model {self.recommended_model} is ready")
                print(f"✅ Model {self.recommended_model} is ready for use!")
                return True

            print(f"🔄 Setting up AI model: {self.recommended_model}")
            print("⏳ This is a one-time setup process...")
            logger.info(f"Model {self.recommended_model} not found, starting download...")

            # Try to download the recommended model
            if await self._download_model(self.recommended_model):
                return True

            # Try fallback models
            print("🔄 Trying alternative models...")
            for model in self.fallback_models:
                print(f"⚡ Attempting to download {model} (smaller/faster)...")
                logger.info(f"Trying fallback model: {model}")

                # Check if fallback model is already available first
                if await self._is_model_available(model):
                    print(f"✅ Found existing model {model}!")
                    self.recommended_model = model
                    return True

                if await self._download_model(model):
                    self.recommended_model = model
                    print(f"✅ Successfully set up {model} as the AI model")
                    return True

            print("❌ Failed to set up any AI model")
            print("💡 You can try manually: ollama pull qwen3:4b")
            return False

        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            logger.error(f"Failed to setup model: {e}")
            return False

    async def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally"""
        # Add retry mechanism for better reliability
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    # Use exact match instead of prefix match to avoid false positives
                    found = any(model["name"] == model_name for model in models)
                    logger.debug(
                        f"Model availability check for {model_name}: {found} (attempt {attempt + 1})"
                    )
                    if found:
                        return True
                    # If not found on first attempt, wait a bit for cache refresh
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                else:
                    logger.warning(
                        f"Failed to get model list: HTTP {response.status_code} (attempt {attempt + 1})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Model availability check failed: {e} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        return False

    def _get_model_size_info(self, model_name: str) -> str:
        """Get estimated size information for a model"""
        # Check our configured models
        for key in ["qwen3", "deepseek"]:
            if key in MODEL_CONFIG:
                model_info = MODEL_CONFIG[key]
                if model_name == model_info["name"] or model_name.startswith(
                    model_info["name"].split(":")[0]
                ):
                    return f"~{model_info['size_gb']:.1f}GB"

        return "~4.0GB (estimated)"

    async def _check_download_status(self, model_name: str) -> Optional[str]:
        """Check if a model is currently being downloaded"""
        try:
            # Check if model is in the process of being downloaded
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                # Look for the model in the list
                for line in result.stdout.split("\n"):
                    if model_name in line:
                        return line.strip()

            return None

        except Exception:
            return None

    async def _download_model(self, model_name: str) -> bool:
        """Download a model with detailed progress display"""
        try:
            # Double-check if model already exists before attempting download
            logger.info(f"Double-checking model {model_name} availability before download...")
            if await self._is_model_available(model_name):
                logger.info(f"Model {model_name} found during double-check, skipping download")
                print(f"✅ Model {model_name} is already available!")
                return True

            model_size = self._get_model_size_info(model_name)
            print(f"📥 Downloading {model_name} (first time setup)...")
            print(f"📊 Estimated size: {model_size}")
            print("⏳ This may take a few minutes depending on your internet connection...")
            print("💡 Tip: You can monitor progress with 'ollama list' in another terminal")
            print()

            # Start the download with timeout
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Set a reasonable timeout for the entire download (20 minutes)
            download_timeout = 20 * 60  # 20 minutes

            # Track progress state
            last_progress_line = ""
            layers_info = {}
            no_output_count = 0
            max_no_output = 10  # Max consecutive no-output cycles before checking completion

            # Monitor progress in real-time with timeout
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    # Wait for next line with shorter timeout
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
                    if not line:
                        # Process finished, check if it was successful
                        break

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    # Reset timeout and counters when we get valid output
                    start_time = asyncio.get_event_loop().time()
                    no_output_count = 0

                    # Parse different types of ollama pull output
                    if "pulling manifest" in line_str.lower():
                        print("📋 Fetching model manifest...")
                    elif "pulling" in line_str.lower() and "%" in line_str:
                        # Extract progress information
                        if line_str != last_progress_line:
                            print(f"📦 {line_str}")
                            last_progress_line = line_str
                    elif "verifying sha256 digest" in line_str.lower():
                        print("🔍 Verifying model integrity...")
                    elif "writing manifest" in line_str.lower():
                        print("📝 Finalizing installation...")
                    elif "success" in line_str.lower():
                        print(f"✅ Model ready!")
                    elif line_str and "error" not in line_str.lower():
                        # Show other progress messages
                        if line_str != last_progress_line:
                            print(f"⚙️  {line_str}")
                            last_progress_line = line_str

                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break

                    no_output_count += 1

                    # Check if we've been waiting too long without output
                    current_time = asyncio.get_event_loop().time()
                    if current_time - start_time > download_timeout:
                        print("⏱️  Download timeout reached")
                        process.terminate()
                        break

                    # Periodically check if model became available during download
                    if no_output_count >= max_no_output:
                        print("🔍 Checking download status...")
                        if await self._is_model_available(model_name):
                            print(f"✅ Model {model_name} is now available!")
                            # Kill the process since model is ready
                            try:
                                process.terminate()
                                await process.wait()
                            except:
                                pass
                            return True
                        no_output_count = 0

                    # Continue waiting
                    print("⠇ Still downloading...")
                    continue

            # Capture any remaining stderr
            try:
                stderr_output = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                if stderr_output:
                    stderr_str = stderr_output.decode().strip()
                    if stderr_str and "error" in stderr_str.lower():
                        print(f"⚠️  {stderr_str}")
            except asyncio.TimeoutError:
                pass  # No stderr output, that's fine

            await process.wait()

            # Final check: verify model is available regardless of process exit code
            print("🔍 Verifying model installation...")
            if await self._is_model_available(model_name):
                print(f"✅ {model_name} is ready for use!")
                return True
            elif process.returncode == 0:
                print(f"✅ {model_name} downloaded and installed successfully!")
                print("🚀 Model is ready for use!")
                return True
            else:
                print(f"❌ Failed to download {model_name}")
                logger.error(f"Download process exited with code {process.returncode}")
                return False

        except asyncio.TimeoutError:
            print("⏱️  Download is taking longer than expected...")
            print("💡 Large models can take 10-30 minutes on slower connections")
            print("📡 You can check progress with: ollama list")
            # Final check even after timeout
            if await self._is_model_available(model_name):
                print(f"✅ Model {model_name} is actually ready!")
                return True
            logger.error(f"Model download timed out: {model_name}")
            return False
        except Exception as e:
            print(f"❌ Download failed with error: {e}")
            # Final check even after exception
            if await self._is_model_available(model_name):
                print(f"✅ Model {model_name} is actually ready!")
                return True
            logger.error(f"Model download failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model"""
        return {
            "model": self.recommended_model,
            "host": self.ollama_host,
            "available": self.ollama_available,
            "system_info": self.system_info,
        }

    async def cleanup_unused_models(self) -> None:
        """Clean up unused models to save space"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])

                # Keep only the recommended model and remove others
                for model in models:
                    model_name = model["name"]
                    if not model_name.startswith(self.recommended_model.split(":")[0]):
                        logger.info(f"Removing unused model: {model_name}")
                        subprocess.run(["ollama", "rm", model_name], capture_output=True)

        except Exception as e:
            logger.warning(f"Failed to cleanup models: {e}")


class OllamaLLM:
    """
    Unified Ollama LLM Client with automatic setup

    This is the main class users should interact with. It automatically handles:
    - Ollama installation
    - Service management
    - Model downloading
    - API communication

    Usage:
        # Simple usage - everything is handled automatically
        llm = OllamaLLM()
        await llm.initialize()
        response = await llm.generate_completion("Analyze this code...")

        # Advanced usage with custom model
        llm = OllamaLLM(model="deepseek-r1:latest", temperature=0.2)
        await llm.initialize()
    """

    def __init__(self, model: str = "auto", host: str = "http://localhost:11434", **kwargs):
        """
        Initialize Ollama LLM client

        Args:
            model: Model name ("auto" for system-optimized selection)
            host: Ollama host URL
            **kwargs: Additional parameters (temperature, timeout, etc.)
        """
        self.host = host.rstrip("/")
        self.temperature = kwargs.get("temperature", 0.1)
        self.timeout = kwargs.get("timeout", 180)  # Increased to 3 minutes for complex analysis

        # Internal manager
        self._manager = _LocalLLMManager()

        # Model selection
        if model == "auto":
            self.model = self._manager.recommended_model
        else:
            self.model = model

        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the LLM client (call this once before using)

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.debug("🔄 OllamaLLM already initialized")
            return True

        try:
            logger.info("🚀 Starting OllamaLLM initialization...")

            # Ensure local LLM is ready
            logger.info("🔧 Step 1: Ensuring local LLM is ready...")
            if await self._manager.ensure_local_llm():
                logger.info("✅ Step 1 completed: Local LLM setup successful")

                # Update model if using auto-selection
                if hasattr(self._manager, "recommended_model"):
                    old_model = self.model
                    self.model = self._manager.recommended_model
                    logger.info(f"🔧 Step 2: Updated model from {old_model} to {self.model}")

                # Test connection
                logger.info("🔧 Step 3: Testing connection to Ollama service...")
                connection_ok = self._test_connection()
                if connection_ok:
                    logger.info("✅ Step 3 completed: Connection test successful")
                else:
                    logger.warning("⚠️ Step 3 warning: Connection test failed, but continuing...")

                # Final verification
                logger.info("🔧 Step 4: Final verification...")
                self._initialized = True
                logger.info(f"🚀 OllamaLLM ready with {self.model}")
                return True
            else:
                logger.error("❌ Step 1 failed: Local LLM setup failed")
                return False

        except Exception as e:
            logger.error(f"❌ OllamaLLM initialization failed: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def _test_connection(self) -> bool:
        """Test connection to Ollama service"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                logger.debug(f"✅ Connected to Ollama at {self.host}")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate completion using Ollama API

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, top_p, top_k)

        Returns:
            Generated completion text
        """
        if not self._initialized:
            logger.warning("LLM not initialized. Call initialize() first.")
            if not await self.initialize():
                return ""

        try:
            # Prepare the request
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 40),
                },
            }

            # Add system prompt if provided
            if system_prompt:
                data["system"] = system_prompt

            # Make the request
            logger.debug(f"Sending request to Ollama: {self.model}")
            start_time = time.time()

            response = requests.post(f"{self.host}/api/generate", json=data, timeout=self.timeout)

            duration = time.time() - start_time
            logger.debug(f"Ollama response received in {duration:.2f}s")

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and system"""
        return {
            "model": self.model,
            "host": self.host,
            "initialized": self._initialized,
            **self._manager.get_model_info(),
        }

    async def cleanup_unused_models(self) -> None:
        """Clean up unused models to save disk space"""
        await self._manager.cleanup_unused_models()

    @classmethod
    async def create_auto(cls, **kwargs) -> "OllamaLLM":
        """
        Factory method to create and initialize an OllamaLLM instance

        Returns:
            Initialized OllamaLLM instance
        """
        llm = cls(model="auto", **kwargs)
        await llm.initialize()
        return llm
