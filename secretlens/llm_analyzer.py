"""
LLM Analyzer

Uses Language Models to enhance secret detection accuracy and reduce false positives.
"""

import asyncio
import logging
import os

from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import SecretMatch, DetectionMethod, ConfidenceLevel, SecretType, FileType
from .ollama_manager import OllamaLLM

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Enhanced secret analysis using Language Models"""

    def __init__(
        self,
        provider: str = "auto",
        model: str = "auto",
        temperature: float = 0.1,
        llm_base_url: Optional[str] = None,
        individual_timeout: int = 120,  # Timeout per individual analysis (seconds)
        batch_timeout: int = 600,  # Timeout per batch (seconds)
        max_retries: int = 2,  # Max retries for failed analyses
        **kwargs,
    ):
        """
        Initialize LLM Analyzer

        Args:
            provider: LLM provider ("auto", "ollama", "openai", "anthropic")
            model: Model name
            temperature: Temperature for LLM responses
            llm_base_url: Base URL for LLM API
            individual_timeout: Timeout for individual analysis (seconds)
            batch_timeout: Timeout for batch processing (seconds)
            max_retries: Maximum retries for failed analyses
            **kwargs: Additional parameters
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.llm_base_url = llm_base_url
        self.individual_timeout = individual_timeout
        self.batch_timeout = batch_timeout
        self.max_retries = max_retries
        self.llm_client = None
        self.initialized = False

    async def initialize(self):
        """Public method to initialize the LLM client"""
        if not self.initialized:
            logger.info("🔧 Initializing LLM analyzer...")
            await self._initialize_llm_client()
            self.initialized = True
            logger.info("✅ LLM analyzer initialization completed")
        else:
            logger.debug("🔄 LLM analyzer already initialized")

    async def _initialize_llm_client(self):
        """Initialize the LLM client with auto-detection"""
        try:
            # Auto-detect LLM preference
            if self.provider == "auto":
                logger.info("🔧 Auto-detecting LLM provider...")
                self.provider = self._detect_llm_preference()
                logger.info(f"🔧 Auto-detected provider: {self.provider}")

            logger.info(f"🤖 Initializing LLM provider: {self.provider}")

            if self.provider == "ollama":
                logger.info("🔧 Initializing Ollama client...")
                await self._setup_local_llm()
            elif self.provider == "openai":
                logger.info("🔧 Initializing OpenAI client...")
                await self._setup_openai()
            elif self.provider == "anthropic":
                logger.info("🔧 Initializing Anthropic client...")
                await self._setup_anthropic()
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")

            logger.info(f"✅ LLM client setup completed for {self.provider}")

            if self.llm_client:
                logger.info(f"✅ LLM client is ready: {type(self.llm_client).__name__}")
            else:
                logger.warning("⚠️ LLM client is None after initialization")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.info("🔄 Falling back to Gitleaks-only mode")
            import traceback

            logger.debug(f"LLM initialization traceback: {traceback.format_exc()}")

    def _detect_llm_preference(self) -> str:
        """Detect user's LLM preference based on environment"""

        # Check for explicit API keys
        if os.getenv("OPENAI_API_KEY"):
            logger.info("🔑 Found OpenAI API key, using OpenAI")
            return "openai"

        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("🔑 Found Anthropic API key, using Anthropic")
            return "anthropic"

        # Default to local LLM for privacy
        logger.info("🔒 No remote API keys found, using local AI for privacy")
        return "ollama"

    async def _setup_local_llm(self):
        """Setup local Ollama LLM"""
        try:
            # Create and initialize unified OllamaLLM client
            self.llm_client = OllamaLLM(model="auto", temperature=self.temperature)

            # Initialize the client (handles installation, service, model download)
            if await self.llm_client.initialize():
                model_info = self.llm_client.get_model_info()
                logger.info(f"✅ Local AI ready with {model_info['model']}")
            else:
                logger.warning("❌ Failed to setup local LLM")
                self.llm_client = None

        except Exception as e:
            logger.error(f"Local LLM setup failed: {e}")
            self.llm_client = None

    async def _setup_openai(self):
        """Setup OpenAI LLM"""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")

            # Get base URL from parameter or environment variables (prioritize parameter > LLM_BASE_URL > OPENAI_BASE_URL)
            base_url = (
                self.llm_base_url
                or os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            # Initialize client with optional base URL
            if base_url:
                self.llm_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"✅ OpenAI initialized with custom base URL: {base_url}")
            else:
                self.llm_client = openai.AsyncOpenAI(api_key=api_key)
                logger.info("✅ OpenAI initialized with default base URL")

            # Use default model if auto
            if self.model == "auto":
                self.model = "gpt-4o-mini"  # Cost-effective choice

            logger.info(f"✅ OpenAI ready with model: {self.model}")

        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            self.llm_client = None
        except Exception as e:
            logger.error(f"OpenAI setup failed: {e}")
            self.llm_client = None

    async def _setup_anthropic(self):
        """Setup Anthropic LLM"""
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found in environment")

            # Get base URL from parameter or environment variables (prioritize parameter > LLM_BASE_URL > ANTHROPIC_BASE_URL)
            base_url = (
                self.llm_base_url
                or os.getenv("LLM_BASE_URL")
                or os.getenv("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com/v1"
            )

            # Initialize client with optional base URL
            if base_url:
                self.llm_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
                logger.info(f"✅ Anthropic initialized with custom base URL: {base_url}")
            else:
                self.llm_client = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("✅ Anthropic initialized with default base URL")

            # Use default model if auto
            if self.model == "auto":
                self.model = "claude-3-haiku-20240307"  # Cost-effective choice

            logger.info(f"✅ Anthropic ready with model: {self.model}")

        except ImportError:
            logger.error("Anthropic library not installed. Run: pip install anthropic")
            self.llm_client = None
        except Exception as e:
            logger.error(f"Anthropic setup failed: {e}")
            self.llm_client = None

    async def analyze_matches(
        self,
        matches: List[SecretMatch],
        context_window: int = 200,
        batch_size: int = 3,  # Reduced batch size for better stability
    ) -> List[SecretMatch]:
        """
        Analyze secret matches using LLM to reduce false positives

        Args:
            matches: List of secret matches to analyze
            context_window: Context window size for analysis
            batch_size: Number of matches to process concurrently (reduced for stability)

        Returns:
            List of analyzed SecretMatch objects with updated confidence and reasoning
        """
        if not self.llm_client:
            logger.info("No LLM available for analysis, returning original matches")
            return matches

        if not matches:
            return matches

        logger.info(f"🔍 Analyzing {len(matches)} potential secrets using {self.provider}...")

        analyzed_matches = []
        total_batches = (len(matches) + batch_size - 1) // batch_size

        # Process matches in batches
        for batch_idx, i in enumerate(range(0, len(matches), batch_size)):
            batch = matches[i : i + batch_size]

            # Show progress
            logger.info(
                f"📊 Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} matches)..."
            )

            # Analyze batch concurrently with timeout protection
            tasks = [self._analyze_single_match(match, context_window) for match in batch]

            try:
                # Add timeout for the entire batch (using configured timeout)
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.batch_timeout,  # Use configured batch timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"⏰ Batch {batch_idx + 1} timed out after {self.batch_timeout}s, using original matches"
                )
                batch_results = [Exception("Batch timeout")] * len(batch)

            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Analysis failed for match {i+j}: {result}")
                    analyzed_matches.append(batch[j])  # Use original match
                else:
                    analyzed_matches.append(result)

        # Log analysis summary
        false_positives = sum(1 for m in analyzed_matches if m.is_likely_false_positive)
        logger.info(
            f"📊 Analysis complete: {false_positives}/{len(analyzed_matches)} likely false positives"
        )

        return analyzed_matches

    async def _analyze_single_match(self, match: SecretMatch, context_window: int) -> SecretMatch:
        """Analyze a single secret match with retry mechanism"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Collect context around the match
                context = await self._collect_context(match, context_window)

                # Create analysis prompt
                prompt = self._create_analysis_prompt(match, context)

                # Get LLM analysis with timeout protection
                try:
                    analysis_result = await asyncio.wait_for(
                        self._call_llm(prompt),
                        timeout=self.individual_timeout,  # Use configured timeout
                    )
                except asyncio.TimeoutError:
                    timeout_msg = f"Analysis timed out after {self.individual_timeout}s for {match.file_path}:{match.line_number}"
                    if attempt < self.max_retries:
                        logger.warning(
                            f"⏰ {timeout_msg} (attempt {attempt + 1}/{self.max_retries + 1})"
                        )
                        last_error = f"Timeout after {self.individual_timeout}s"
                        continue
                    else:
                        logger.warning(f"⏰ {timeout_msg} (final attempt)")
                        return match  # Return original match if timeout after all retries

                # Parse and apply the analysis
                updated_match = self._apply_analysis_result(match, analysis_result)

                # If we reach here, analysis was successful
                if attempt > 0:
                    logger.info(
                        f"✅ Analysis succeeded for {match.file_path}:{match.line_number} on attempt {attempt + 1}"
                    )

                return updated_match

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    logger.warning(
                        f"⚠️ Analysis attempt {attempt + 1} failed for {match.file_path}:{match.line_number}: {e}"
                    )
                    # Wait a bit before retrying
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.warning(
                        f"❌ All analysis attempts failed for {match.file_path}:{match.line_number}: {e}"
                    )
                    return match

        # Should not reach here, but just in case
        logger.warning(
            f"Analysis failed after {self.max_retries + 1} attempts for {match.file_path}:{match.line_number}: {last_error}"
        )
        return match

    async def _collect_context(self, match: SecretMatch, context_window: int) -> str:
        """Collect surrounding context for a secret match"""
        try:
            file_path = Path(match.file_path)
            if not file_path.exists():
                return match.surrounding_context

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Calculate context window around the match
            start_line = max(0, match.line_number - 10)
            end_line = min(len(lines), match.line_number + 10)

            context_lines = lines[start_line:end_line]
            context = "".join(context_lines)

            # Truncate if too long
            if len(context) > context_window:
                context = context[:context_window] + "... [truncated]"

            return context

        except Exception as e:
            logger.warning(f"Failed to collect context for {match.file_path}: {e}")
            return match.surrounding_context

    def _create_analysis_prompt(self, match: SecretMatch, context: str) -> str:
        """Create a prompt for LLM analysis"""

        # Escape curly braces in all dynamic content to prevent f-string formatting issues
        safe_matched_text = str(match.matched_text).replace("{", "{{").replace("}", "}}")
        safe_pattern = str(match.pattern_matched).replace("{", "{{").replace("}", "}}")
        safe_context = str(context).replace("{", "{{").replace("}", "}}")

        prompt = f"""You are a security expert analyzing potential secret leaks in code. 

**TASK**: Analyze whether the following detected secret is a true positive (real secret) or false positive (test data, example, template, etc.).

**SECRET DETAILS:**
- Type: {match.secret_type.value}
- Matched Text: {safe_matched_text}
- File: {match.file_path}
- Line: {match.line_number}
- Detection Rule: {safe_pattern}
- Current Confidence: {match.confidence:.2f}

**CODE CONTEXT:**
```
{safe_context}
```

**ANALYSIS REQUIRED:**
Please analyze this potential secret and provide:

1. **is_likely_false_positive** (true/false): Whether this appears to be a false positive
2. **confidence_score** (0.0-1.0): Your confidence that this is a real secret
3. **reasoning**: Detailed explanation of your analysis
4. **risk_level** (low/medium/high): Risk level if this is a real secret
5. **recommendations**: Specific actions to take

**COMMON FALSE POSITIVE PATTERNS TO CONSIDER:**
- Test data, mock values, placeholders
- Configuration templates with example values
- Documentation examples
- Hard-coded development/staging credentials clearly marked as such
- Base64-encoded non-secret data
- Generated test tokens that are obviously fake

**REAL SECRET INDICATORS:**
- Production-like values in configuration files
- Credentials in environment variables
- Private keys without "test" or "example" indicators
- API keys with realistic formats and entropy
- Database URLs with real-looking hostnames

**RESPONSE FORMAT (XML):**
Return the analysis results in XML format, please strictly follow the following format:

<analysis>
  <is_likely_false_positive>true/false</is_likely_false_positive>
  <confidence_score>0.0-1.0</confidence_score>
  <reasoning>Your detailed explanation of the analysis</reasoning>
  <risk_level>low/medium/high</risk_level>
  <recommendations>Specific actions to take</recommendations>
</analysis>

Respond with ONLY the XML, no additional text."""

        return prompt

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM with the analysis prompt"""
        try:
            if self.provider == "ollama":
                response = await self._call_ollama(prompt)
            elif self.provider == "openai":
                response = await self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Parse XML response
            # Extract XML from code block or thinking content if present
            xml_content = self._extract_xml_from_response(response)
            return self._parse_xml_response(xml_content)

        except Exception as e:
            logger.warning(f"Failed to parse LLM response as XML: {e}")
            logger.warning(f"Raw LLM response: {repr(response)}")
            # Return default analysis
            return {
                "is_likely_false_positive": False,
                "confidence_score": 0.5,
                "reasoning": "Failed to parse LLM response",
                "risk_level": "medium",
                "recommendations": "Manual review required",
            }

    def _extract_xml_from_response(self, response: str) -> str:
        """Extract XML content from LLM response, handling code blocks and thinking content"""
        import re

        # Clean response text, remove possible code block markers
        cleaned_text = response.strip()

        # Try to extract content surrounded by ```xml and ```
        xml_pattern = r"```xml\s*([\s\S]*?)\s*```"
        match = re.search(xml_pattern, cleaned_text)
        if match:
            return match.group(1).strip()

        # Try to extract content surrounded by ``` and ```
        code_pattern = r"```(?:xml)?\s*([\s\S]*?)\s*```"
        match = re.search(code_pattern, cleaned_text)
        if match:
            content = match.group(1).strip()
            # Check if it looks like XML
            if "<analysis>" in content and "</analysis>" in content:
                return content

        # Try to find XML directly in the response (for thinking models)
        # Look for analysis XML
        analysis_pattern = r"<analysis>([\s\S]*?)</analysis>"
        match = re.search(analysis_pattern, cleaned_text, re.DOTALL)
        if match:
            return f"<analysis>{match.group(1)}</analysis>"

        # Look for discoveries XML
        discoveries_pattern = r"<discoveries>([\s\S]*?)</discoveries>"
        match = re.search(discoveries_pattern, cleaned_text, re.DOTALL)
        if match:
            return f"<discoveries>{match.group(1)}</discoveries>"

        # If no XML found, return the response as-is
        return cleaned_text

    def _extract_xml_tag(self, text: str, tag_name: str) -> str:
        """Extract XML tag content from text"""
        import re

        pattern = rf"<{tag_name}>([\s\S]*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_xml_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML response and extract analysis results"""
        try:
            result = {}

            # Extract each field from XML
            is_false_positive_str = self._extract_xml_tag(xml_content, "is_likely_false_positive")
            if is_false_positive_str.lower() in ["true", "yes", "1"]:
                result["is_likely_false_positive"] = True
            elif is_false_positive_str.lower() in ["false", "no", "0"]:
                result["is_likely_false_positive"] = False
            else:
                result["is_likely_false_positive"] = False  # Default to false for safety

            # Extract confidence score
            confidence_str = self._extract_xml_tag(xml_content, "confidence_score")
            try:
                result["confidence_score"] = float(confidence_str)
            except (ValueError, TypeError):
                result["confidence_score"] = 0.5  # Default confidence

            # Extract other fields
            result["reasoning"] = (
                self._extract_xml_tag(xml_content, "reasoning") or "No reasoning provided"
            )
            result["risk_level"] = self._extract_xml_tag(xml_content, "risk_level") or "medium"
            result["recommendations"] = (
                self._extract_xml_tag(xml_content, "recommendations") or "Manual review required"
            )

            # Validate risk level
            if result["risk_level"] not in ["low", "medium", "high"]:
                result["risk_level"] = "medium"

            return result

        except Exception as e:
            logger.error(f"Error parsing XML response: {e}")
            # Return default analysis
            return {
                "is_likely_false_positive": False,
                "confidence_score": 0.5,
                "reasoning": "Failed to parse XML response",
                "risk_level": "medium",
                "recommendations": "Manual review required",
            }

    def _parse_discovery_xml_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse XML response for discovery functionality and extract list of secrets"""
        import re

        try:
            discoveries = []

            # Find all <secret> tags
            secret_pattern = r"<secret>([\s\S]*?)</secret>"
            secret_matches = re.findall(secret_pattern, xml_content, re.DOTALL)

            for secret_content in secret_matches:
                try:
                    discovery = {}

                    # Extract each field from the secret XML
                    discovery["type"] = (
                        self._extract_xml_tag(secret_content, "type") or "generic_secret"
                    )
                    discovery["value"] = self._extract_xml_tag(secret_content, "value") or ""
                    discovery["context"] = self._extract_xml_tag(secret_content, "context") or ""
                    discovery["reasoning"] = (
                        self._extract_xml_tag(secret_content, "reasoning")
                        or "No reasoning provided"
                    )

                    # Extract line number
                    line_number_str = self._extract_xml_tag(secret_content, "line_number")
                    try:
                        discovery["line_number"] = int(line_number_str) if line_number_str else 0
                    except (ValueError, TypeError):
                        discovery["line_number"] = 0

                    # Extract confidence
                    confidence_str = self._extract_xml_tag(secret_content, "confidence")
                    try:
                        discovery["confidence"] = float(confidence_str) if confidence_str else 0.5
                    except (ValueError, TypeError):
                        discovery["confidence"] = 0.5

                    # Only add if we have a value
                    if discovery["value"]:
                        discoveries.append(discovery)

                except Exception as e:
                    logger.warning(f"Failed to parse individual secret from XML: {e}")
                    continue

            return discoveries

        except Exception as e:
            logger.error(f"Error parsing discovery XML response: {e}")
            return []

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        system_prompt = "You are a security expert specializing in secret detection and false positive analysis."

        response = await self.llm_client.generate_completion(
            prompt=prompt, system_prompt=system_prompt, temperature=self.temperature
        )
        return response

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a security expert specializing in secret detection and false positive analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _apply_analysis_result(self, match: SecretMatch, analysis: Dict[str, Any]) -> SecretMatch:
        """Apply LLM analysis results to SecretMatch"""
        # Create updated match
        updated_match = match.model_copy()

        # Update with LLM analysis
        updated_match.is_likely_false_positive = analysis.get("is_likely_false_positive", False)
        updated_match.llm_reasoning = analysis.get("reasoning", "")
        updated_match.detection_method = DetectionMethod.HYBRID

        # Update confidence score
        llm_confidence = analysis.get("confidence_score", match.confidence)
        # Combine original confidence with LLM confidence (weighted average)
        combined_confidence = (match.confidence * 0.3) + (llm_confidence * 0.7)
        updated_match.confidence = combined_confidence
        updated_match.confidence_level = self._get_confidence_level(combined_confidence)

        # Store additional analysis data
        if not updated_match.context_analysis:
            updated_match.context_analysis = {}

        updated_match.context_analysis.update(
            {"llm_analysis": analysis, "llm_provider": self.provider, "llm_model": self.model}
        )

        return updated_match

    async def find_additional_secrets(
        self,
        file_paths: List[str],
        context_window: int = 3000,
    ) -> List[SecretMatch]:
        """
        Use LLM to find additional secrets that might be missed by traditional tools

        Args:
            file_paths: List of file paths to analyze
            context_window: Maximum content length to analyze per file

        Returns:
            List of additionally discovered SecretMatch objects
        """
        if not self.llm_client:
            logger.info("No LLM available for additional secret discovery")
            return []

        logger.info(f"Searching for additional secrets in {len(file_paths)} files")

        additional_secrets = []

        for file_path in file_paths:
            try:
                content = self._read_file_content(file_path, context_window)
                if not content:
                    continue

                # Create prompt for finding missed secrets
                prompt = self._create_discovery_prompt(file_path, content)

                # Analyze with LLM
                discovery_result = await self._call_llm_for_discovery(prompt)

                # Convert discovered items to SecretMatch objects
                new_matches = self._convert_discoveries_to_matches(
                    file_path, content, discovery_result
                )

                additional_secrets.extend(new_matches)

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path} for additional secrets: {e}")
                continue

        logger.info(f"Found {len(additional_secrets)} additional potential secrets")
        return additional_secrets

    def _read_file_content(self, file_path: str, max_length: int) -> str:
        """Read file content with length limit"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if len(content) > max_length:
                content = content[:max_length] + "... [truncated]"

            return content
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""

    def _create_discovery_prompt(self, file_path: str, content: str) -> str:
        """Create prompt for discovering additional secrets"""

        # Escape curly braces in content to prevent f-string formatting issues
        safe_content = str(content).replace("{", "{{").replace("}", "}}")

        prompt = f"""You are a security expert looking for secrets and credentials that might be missed by traditional regex-based tools.

**FILE**: {file_path}

**CONTENT**:
```
{safe_content}
```

**TASK**: Identify any secrets, credentials, or sensitive information that might be present, including:

1. **Non-standard formats**: Secrets that don't match typical patterns
2. **Context-based secrets**: Information that's sensitive based on context
3. **Encoded/obfuscated data**: Base64, hex, or other encoded credentials
4. **Comments and strings**: Credentials mentioned in comments or string literals
5. **Configuration values**: Database URLs, API endpoints with embedded credentials
6. **Private keys**: Any form of private key material
7. **Tokens**: Access tokens, session tokens, etc.

**FOCUS ON**:
- Unusual string patterns that might be credentials
- Comments containing sensitive information
- Configuration values that look like real secrets
- Any form of authentication material

**RESPONSE FORMAT** (XML):
Return the discovered secrets in XML format, please strictly follow the following format:

<discoveries>
  <secret>
    <type>api_key|password|private_key|token|database_url|generic_secret</type>
    <value>the actual secret value found</value>
    <line_number>number</line_number>
    <context>surrounding context</context>
    <reasoning>why this appears to be a secret</reasoning>
    <confidence>0.0-1.0</confidence>
  </secret>
  <secret>
    <type>password</type>
    <value>another secret value</value>
    <line_number>number</line_number>
    <context>surrounding context</context>
    <reasoning>why this appears to be a secret</reasoning>
    <confidence>0.0-1.0</confidence>
  </secret>
  <!-- more secrets as needed -->
</discoveries>

If no additional secrets are found, return <discoveries></discoveries>.

Respond with ONLY the XML, no additional text."""

        return prompt

    def _convert_discoveries_to_matches(
        self, file_path: str, content: str, discoveries: List[Dict[str, Any]]
    ) -> List[SecretMatch]:
        """Convert LLM discoveries to SecretMatch objects"""
        matches = []

        if not isinstance(discoveries, list):
            return matches

        for discovery in discoveries:
            try:
                from .models import SecretType, FileType

                # Map type string to enum
                type_mapping = {
                    "api_key": SecretType.API_KEY,
                    "password": SecretType.PASSWORD,
                    "private_key": SecretType.PRIVATE_KEY,
                    "token": SecretType.ACCESS_TOKEN,
                    "database_url": SecretType.DATABASE_URL,
                    "generic_secret": SecretType.GENERIC_SECRET,
                }

                secret_type = type_mapping.get(
                    discovery.get("type", "generic_secret"), SecretType.GENERIC_SECRET
                )

                # Determine file type
                file_type = self._determine_file_type(file_path)

                confidence = float(discovery.get("confidence", 0.5))

                match = SecretMatch(
                    secret_type=secret_type,
                    matched_text=discovery.get("value", ""),
                    line_number=int(discovery.get("line_number", 0)),
                    column_start=0,
                    column_end=len(discovery.get("value", "")),
                    file_path=file_path,
                    file_type=file_type,
                    surrounding_context=discovery.get("context", ""),
                    detection_method=DetectionMethod.LLM_ANALYSIS,
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    llm_reasoning=discovery.get("reasoning", ""),
                    context_analysis={
                        "llm_discovery": discovery,
                        "llm_provider": self.provider,
                        "llm_model": self.model,
                    },
                )

                matches.append(match)

            except Exception as e:
                logger.warning(f"Failed to convert discovery to SecretMatch: {e}")
                continue

        return matches

    def _determine_file_type(self, file_path: str) -> "FileType":
        """Determine file type (duplicate from gitleaks_integration, should be refactored)"""
        from .models import FileType

        path = Path(file_path)
        suffix = path.suffix.lower()

        source_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".rb",
            ".php",
        }
        if suffix in source_extensions:
            return FileType.SOURCE_CODE

        config_extensions = {".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".config"}
        config_names = {"dockerfile", "makefile", ".env", ".gitignore"}
        if suffix in config_extensions or path.name.lower() in config_names:
            return FileType.CONFIG_FILE

        doc_extensions = {".md", ".txt", ".rst", ".adoc"}
        if suffix in doc_extensions:
            return FileType.DOCUMENTATION

        data_extensions = {".csv", ".sql", ".db", ".sqlite"}
        if suffix in data_extensions:
            return FileType.DATA_FILE

        return FileType.UNKNOWN

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the current LLM setup"""
        info = {
            "provider": self.provider,
            "model": self.model,
            "available": self.llm_client is not None,
            "temperature": self.temperature,
        }

        if self.llm_client and hasattr(self.llm_client, "get_model_info"):
            info.update(self.llm_client.get_model_info())

        return info

    async def _call_llm_for_discovery(self, prompt: str) -> List[Dict[str, Any]]:
        """Call the LLM for secret discovery (expects XML list response)"""
        try:
            if self.provider == "ollama":
                response = await self._call_ollama(prompt)
            elif self.provider == "openai":
                response = await self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Parse XML response
            # Extract XML from code block or thinking content if present
            xml_content = self._extract_xml_from_response(response)
            discoveries = self._parse_discovery_xml_response(xml_content)

            return discoveries

        except Exception as e:
            logger.warning(f"Failed to parse LLM discovery response as XML: {e}")
            logger.warning(f"Raw LLM response: {repr(response)}")
            # Return empty list for discovery
            return []
