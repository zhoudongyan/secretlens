"""
Enhanced Scanner

Main module that combines Gitleaks and LLM analysis for improved accuracy.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import ScanConfig, ScanResult, SecretMatch, DetectionMethod
from .gitleaks_integration import GitleaksIntegration
from .llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class EnhancedScanner:
    """Main scanner that combines Gitleaks with LLM analysis"""

    def __init__(self, config: Optional[ScanConfig] = None):
        """
        Initialize the enhanced scanner

        Args:
            config: Optional ScanConfig, will use defaults if not provided
        """
        self.config = config or ScanConfig(target_path=".")
        self.gitleaks = GitleaksIntegration()

        # Initialize LLM analyzer with auto-detection
        self.llm_analyzer = None
        if self.config.enable_llm_analysis:
            self._initialize_llm_analyzer()

    def _initialize_llm_analyzer(self):
        """Initialize LLM analyzer with smart defaults"""
        try:
            # Determine LLM provider based on configuration
            provider = self._determine_llm_provider()
            model = self._determine_llm_model(provider)

            logger.info(f"🤖 Initializing AI analysis with {provider}")

            self.llm_analyzer = LLMAnalyzer(
                provider=provider,
                model=model,
                temperature=self.config.llm_temperature,
                llm_base_url=self.config.llm_base_url,
                individual_timeout=self.config.individual_timeout,
                batch_timeout=self.config.batch_timeout,
                max_retries=self.config.max_retries,
            )

        except Exception as e:
            logger.warning(f"Failed to initialize LLM analyzer: {e}")
            logger.info("🔄 Continuing with Gitleaks-only analysis")
            self.llm_analyzer = None

    def _determine_llm_provider(self) -> str:
        """Determine which LLM provider to use"""
        # If user explicitly configured a provider, use it
        if self.config.llm_provider != "auto":
            return self.config.llm_provider

        # Auto-detect based on API keys and preferences
        import os

        if os.getenv("OPENAI_API_KEY"):
            logger.info("🔑 Using OpenAI (API key found)")
            return "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            logger.info("🔑 Using Anthropic (API key found)")
            return "anthropic"
        else:
            logger.info("🔒 Using local AI for privacy (no remote API keys found)")
            return "ollama"

    def _determine_llm_model(self, provider: str) -> str:
        """Determine which model to use for the provider"""
        if self.config.llm_model != "auto":
            return self.config.llm_model

        # Auto-select cost-effective models
        model_defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "ollama": "auto",  # Will be determined by ollama llm
        }

        return model_defaults.get(provider, "auto")

    async def scan(self) -> ScanResult:
        """
        Perform enhanced secret scanning

        Returns:
            ScanResult with all findings and analysis
        """
        start_time = time.time()

        # Print friendly banner
        self._print_scan_banner()

        logger.info(f"🔍 Starting enhanced scan of {self.config.target_path}")

        try:
            # Step 1: Run Gitleaks scan
            logger.info("📡 Running initial secret detection...")
            gitleaks_findings = self.gitleaks.scan_directory(
                self.config.target_path, config=self.config, verbose=self.config.verbose
            )

            # Convert to SecretMatch objects
            initial_matches = self.gitleaks.convert_to_secret_matches(gitleaks_findings)
            logger.info(f"🎯 Found {len(initial_matches)} potential secrets")

            # Step 2: LLM analysis of Gitleaks results
            analyzed_matches = initial_matches
            if self.llm_analyzer and initial_matches:
                logger.info("🧠 Analyzing with AI to reduce false positives...")

                # Wait for LLM to be ready
                await self._ensure_llm_ready()

                if self.llm_analyzer.llm_client:
                    analyzed_matches = await self.llm_analyzer.analyze_matches(
                        initial_matches,
                        context_window=self.config.max_context_length,
                        batch_size=self.config.batch_size,
                    )

                    # Log analysis results
                    false_positives = sum(1 for m in analyzed_matches if m.is_likely_false_positive)
                    logger.info(f"✨ AI identified {false_positives} likely false positives")
                else:
                    logger.info("🔄 AI analysis unavailable, using Gitleaks results")

            # Step 3: Additional LLM-based discovery (optional)
            additional_matches = []
            if (
                self.llm_analyzer
                and self.llm_analyzer.llm_client
                and not self.config.analysis_only
                and self.config.enable_additional_discovery
            ):

                logger.info("🔍 Searching for additional secrets with AI...")

                # Collect file paths for additional analysis
                file_paths = self._collect_file_paths()
                if file_paths:
                    # Limit files based on configuration
                    limited_files = file_paths[: self.config.max_discovery_files]
                    logger.info(f"🎯 Analyzing {len(limited_files)} files for additional secrets")

                    additional_matches = await self.llm_analyzer.find_additional_secrets(
                        limited_files,
                        context_window=self.config.max_context_length,
                    )
                    if additional_matches:
                        logger.info(
                            f"💡 AI discovered {len(additional_matches)} additional potential secrets"
                        )
                    else:
                        logger.info("💡 No additional secrets found by AI")
                else:
                    logger.info("⚠️  No files to analyze for additional secrets")

            # Step 4: Combine and filter results
            all_matches = analyzed_matches + additional_matches

            # Filter by confidence threshold
            if not self.config.include_low_confidence:
                all_matches = [
                    m for m in all_matches if m.confidence >= self.config.confidence_threshold
                ]

            # Filter out likely false positives if LLM analysis was performed
            if self.llm_analyzer and self.llm_analyzer.llm_client:
                high_confidence_matches = [m for m in all_matches if not m.is_likely_false_positive]
                logger.info(
                    f"🎯 Final results: {len(high_confidence_matches)} high-confidence secrets"
                )
            else:
                high_confidence_matches = all_matches

            # Step 5: Generate scan result
            scan_duration = time.time() - start_time

            # Count files scanned
            total_files = len(self._collect_file_paths())

            # Create summary
            summary = self._generate_summary(
                gitleaks_matches=len(initial_matches),
                llm_analyzed=len(analyzed_matches),
                additional_found=len(additional_matches),
                final_matches=len(high_confidence_matches),
                false_positives_filtered=sum(
                    1 for m in analyzed_matches if m.is_likely_false_positive
                ),
            )

            result = ScanResult(
                config=self.config,
                matches=high_confidence_matches,
                scan_summary=summary,
                total_files_scanned=total_files,
                total_matches_found=len(high_confidence_matches),
                scan_duration_seconds=scan_duration,
                timestamp=datetime.now().isoformat(),
            )

            # Print completion message
            self._print_completion_message(result)

            return result

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            raise

    def _print_scan_banner(self):
        """Print a friendly scan banner"""
        print()
        print("🔍 SecretLens - AI-Enhanced Secret Detection")
        print("=" * 50)

        # Show LLM status
        if self.llm_analyzer:
            if hasattr(self.llm_analyzer, "provider"):
                if self.llm_analyzer.provider == "ollama":
                    print("🤖 Using local AI (privacy-protected)")
                elif self.llm_analyzer.provider == "openai":
                    print("🤖 Using OpenAI for analysis")
                elif self.llm_analyzer.provider == "anthropic":
                    print("🤖 Using Anthropic for analysis")
                else:
                    print("🤖 AI analysis enabled")
            else:
                print("🤖 AI analysis enabled")
        else:
            print("📡 Using Gitleaks detection only")

        print(f"📂 Scanning: {self.config.target_path}")
        print()

    async def _ensure_llm_ready(self):
        """Ensure LLM is ready for analysis"""
        if not self.llm_analyzer:
            return

        # Initialize the LLM analyzer if not already done
        if not self.llm_analyzer.initialized:
            logger.info("🔧 Initializing AI analysis...")

            # Show setup message for local LLM
            if self.llm_analyzer.provider == "ollama":
                print(
                    "⏳ Setting up local AI (this may take a few minutes for first-time setup)..."
                )

            await self.llm_analyzer.initialize()

    def _print_completion_message(self, result: ScanResult):
        """Print scan completion message"""
        print()
        print("✅ Scan completed!")
        print(f"⏱️  Duration: {result.scan_duration_seconds:.2f} seconds")
        print(f"📁 Files scanned: {result.total_files_scanned}")
        print(f"🎯 Secrets found: {result.total_matches_found}")

        if result.total_matches_found > 0:
            print()
            print("🚨 Action required: Review detected secrets")
            high_conf = sum(1 for m in result.matches if m.confidence >= 0.8)
            if high_conf > 0:
                print(f"⚠️  High confidence: {high_conf} secrets")
        else:
            print("🎉 No secrets detected!")

        print()

    def _collect_file_paths(self) -> List[str]:
        """Collect file paths for scanning based on include/exclude patterns"""
        import fnmatch

        target_path = Path(self.config.target_path)
        if not target_path.exists():
            logger.warning(f"Target path does not exist: {target_path}")
            return []

        file_paths = []

        # Collect all files
        if target_path.is_file():
            file_paths = [str(target_path)]
        else:
            for file_path in target_path.rglob("*"):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(target_path))

                    # Check file size limit
                    if file_path.stat().st_size > self.config.max_file_size:
                        continue

                    # Check include patterns
                    if self.config.include_patterns:
                        if not any(
                            fnmatch.fnmatch(relative_path, pattern)
                            for pattern in self.config.include_patterns
                        ):
                            continue

                    # Check exclude patterns
                    if any(
                        fnmatch.fnmatch(relative_path, pattern)
                        for pattern in self.config.exclude_patterns
                    ):
                        continue

                    file_paths.append(str(file_path))

        logger.debug(f"Collected {len(file_paths)} files for scanning")
        return file_paths

    def _generate_summary(
        self,
        gitleaks_matches: int,
        llm_analyzed: int,
        additional_found: int,
        final_matches: int,
        false_positives_filtered: int,
    ) -> Dict[str, Any]:
        """Generate scan summary statistics"""
        summary = {
            "gitleaks_matches": gitleaks_matches,
            "llm_analyzed": llm_analyzed,
            "additional_found": additional_found,
            "final_matches": final_matches,
            "false_positives_filtered": false_positives_filtered,
            "false_positive_rate": 0.0,
            "enhancement_enabled": self.config.enable_llm_analysis,
            "llm_info": None,
        }

        # Calculate false positive rate
        if gitleaks_matches > 0:
            summary["false_positive_rate"] = false_positives_filtered / gitleaks_matches

        # Add LLM info if available
        if self.llm_analyzer:
            try:
                summary["llm_info"] = self.llm_analyzer.get_llm_info()
            except:
                pass

        return summary

    def generate_detailed_report(self, scan_result: ScanResult) -> str:
        """Generate a human-readable detailed report"""
        report = []
        report.append(f"# SecretLens Enhanced Scan Report")
        report.append(f"")
        report.append(f"**Scan Target:** {scan_result.config.target_path}")
        report.append(f"**Scan Time:** {scan_result.timestamp}")
        report.append(f"**Duration:** {scan_result.scan_duration_seconds:.2f} seconds")
        report.append(f"**Files Scanned:** {scan_result.total_files_scanned}")
        report.append(f"")

        # AI Analysis info
        llm_info = scan_result.scan_summary.get("llm_info")
        if llm_info and llm_info.get("available"):
            provider = llm_info.get("provider", "unknown")
            model = llm_info.get("name", llm_info.get("model", "unknown"))
            if provider == "ollama":
                report.append(f"**AI Analysis:** Local AI ({model}) - Privacy Protected")
            else:
                report.append(f"**AI Analysis:** {provider.title()} ({model})")
        else:
            report.append(f"**AI Analysis:** Not available")
        report.append(f"")

        # Summary section
        summary = scan_result.scan_summary
        report.append(f"## Summary")
        report.append(f"- **Total Matches Found:** {scan_result.total_matches_found}")
        report.append(f"- **Gitleaks Initial Matches:** {summary.get('gitleaks_matches', 0)}")
        report.append(
            f"- **False Positives Filtered:** {summary.get('false_positives_filtered', 0)}"
        )
        report.append(f"- **Additional Secrets Found:** {summary.get('additional_found', 0)}")
        report.append(f"- **False Positive Rate:** {summary.get('false_positive_rate', 0.0):.1%}")
        report.append(f"")

        # Detailed findings
        if scan_result.matches:
            report.append(f"## Detailed Findings")
            report.append(f"")

            # Group by confidence level
            high_conf = [m for m in scan_result.matches if m.confidence >= 0.9]
            med_conf = [m for m in scan_result.matches if 0.7 <= m.confidence < 0.9]
            low_conf = [m for m in scan_result.matches if m.confidence < 0.7]

            for level, matches in [("High", high_conf), ("Medium", med_conf), ("Low", low_conf)]:
                if matches:
                    report.append(f"### {level} Confidence ({len(matches)} matches)")
                    report.append(f"")

                    for i, match in enumerate(matches[:10], 1):  # Limit to first 10
                        report.append(
                            f"**{i}. {match.secret_type.value.replace('_', ' ').title()}**"
                        )
                        report.append(f"   - **File:** {match.file_path}:{match.line_number}")
                        report.append(f"   - **Confidence:** {match.confidence:.2f}")
                        report.append(f"   - **Method:** {match.detection_method.value}")

                        if match.llm_reasoning:
                            report.append(f"   - **Analysis:** {match.llm_reasoning}")

                        report.append(f"   - **Context:** `{match.surrounding_context.strip()}`")
                        report.append(f"")

                    if len(matches) > 10:
                        report.append(f"   ... and {len(matches) - 10} more matches")
                        report.append(f"")
        else:
            report.append(f"## No secrets found! 🎉")
            report.append(f"")

        # Recommendations
        report.append(f"## Recommendations")
        if scan_result.matches:
            high_risk = [m for m in scan_result.matches if m.confidence >= 0.8]
            if high_risk:
                report.append(
                    f"- **Immediate Action Required:** {len(high_risk)} high-confidence secrets detected"
                )
                report.append(f"- Review and rotate any confirmed secrets")
                report.append(f"- Add secrets to your secret management system")
                report.append(f"- Consider using environment variables for configuration")
        else:
            report.append(f"- Great job! No secrets detected in the codebase")
            report.append(f"- Continue following security best practices")

        report.append(f"")
        report.append(f"---")
        report.append(f"*Generated by SecretLens - AI-Enhanced Secret Detection*")

        return "\n".join(report)
