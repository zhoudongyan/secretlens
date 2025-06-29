"""
Gitleaks Integration Module

Handles calling Gitleaks and parsing its output for further LLM analysis.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .models import SecretMatch, SecretType, DetectionMethod, FileType, ConfidenceLevel, ScanConfig

logger = logging.getLogger(__name__)


class GitleaksIntegration:
    """Integration with Gitleaks for initial secret detection"""

    def __init__(self, gitleaks_binary: str = "gitleaks"):
        self.gitleaks_binary = gitleaks_binary
        self._verify_gitleaks_installation()

    def _verify_gitleaks_installation(self):
        """Verify that Gitleaks is installed and accessible"""
        try:
            result = subprocess.run(
                [self.gitleaks_binary, "version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Gitleaks not found or not working: {result.stderr}")
            logger.info(f"Gitleaks version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Gitleaks not found. Please install it first: {e}")

    def scan_directory(
        self, target_path: str, config: Optional[ScanConfig] = None, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run Gitleaks scan on directory and return raw results

        Args:
            target_path: Path to scan
            config: ScanConfig object with Gitleaks settings
            verbose: Whether to run in verbose mode

        Returns:
            List of raw Gitleaks findings
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Determine scan command based on target type and config
            cmd = self._build_gitleaks_command(target_path, output_path, config, verbose)

            logger.info(f"Running Gitleaks: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60 * 60  # 1 hour timeout
            )

            # Gitleaks returns exit code 1 when secrets are found, which is expected
            if result.returncode not in [0, 1]:
                logger.error(f"Gitleaks failed: {result.stderr}")
                raise RuntimeError(f"Gitleaks scan failed: {result.stderr}")

            # Parse the JSON output
            try:
                with open(output_path, "r") as f:
                    content = f.read().strip()
                    if not content:
                        logger.info("No secrets found by Gitleaks")
                        return []

                    findings = json.loads(content)
                    logger.info(f"Gitleaks found {len(findings)} potential secrets")
                    return findings
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gitleaks output: {e}")
                return []

        finally:
            # Clean up temp file
            try:
                Path(output_path).unlink()
            except FileNotFoundError:
                pass

    def _build_gitleaks_command(
        self,
        target_path: str,
        output_path: str,
        config: Optional[ScanConfig] = None,
        verbose: bool = False,
    ) -> List[str]:
        """
        Build Gitleaks command with all appropriate flags based on configuration

        Args:
            target_path: Path to scan
            output_path: Path for output file
            config: ScanConfig object with Gitleaks settings
            verbose: Whether to run in verbose mode

        Returns:
            Complete command list for subprocess
        """
        # Determine scan mode
        scan_mode = self._determine_scan_mode(target_path, config)

        # Base command with scan mode and target path
        cmd = [self.gitleaks_binary, scan_mode, str(target_path)]

        # Report settings
        cmd.extend(
            [
                "--report-format",
                "json",
                "--report-path",
                output_path,
            ]
        )

        # Add configuration-based flags
        if config:
            self._add_config_flags(cmd, config)

        # Add verbose flag
        if verbose:
            cmd.append("--verbose")

        return cmd

    def _determine_scan_mode(self, target_path: str, config: Optional[ScanConfig] = None) -> str:
        """
        Determine whether to use 'dir' or 'git' scan mode

        Args:
            target_path: Path to scan
            config: ScanConfig object

        Returns:
            'dir' or 'git' command
        """
        if config and config.scan_mode != "auto":
            return config.scan_mode

        # Auto-detect based on whether target is a git repository
        if self._is_git_repository(target_path):
            return "git"
        else:
            return "dir"

    def _is_git_repository(self, path: str) -> bool:
        """Check if the path is a git repository"""
        git_dir = Path(path) / ".git"
        return git_dir.exists()

    def _add_config_flags(self, cmd: List[str], config: ScanConfig) -> None:
        """
        Add configuration flags to the Gitleaks command

        Args:
            cmd: Command list to modify
            config: ScanConfig object with settings
        """
        # Configuration file
        if config.gitleaks_config_path:
            cmd.extend(["--config", config.gitleaks_config_path])

        # Baseline file for ignoring known issues
        if config.gitleaks_baseline_path:
            cmd.extend(["--baseline-path", config.gitleaks_baseline_path])

        # Gitleaks ignore file
        if config.gitleaks_ignore_path:
            cmd.extend(["--gitleaks-ignore-path", config.gitleaks_ignore_path])

        # Enable specific rules only
        if config.enable_rules:
            for rule in config.enable_rules:
                cmd.extend(["--enable-rule", rule])

        # Decoding settings
        if config.max_decode_depth > 0:
            cmd.extend(["--max-decode-depth", str(config.max_decode_depth)])

        # Archive depth settings
        if config.max_archive_depth > 0:
            cmd.extend(["--max-archive-depth", str(config.max_archive_depth)])

        # File size limit
        if config.max_target_megabytes:
            cmd.extend(["--max-target-megabytes", str(config.max_target_megabytes)])

        # Redaction settings
        if config.redact_percentage != 100:
            cmd.append(f"--redact={config.redact_percentage}")

        # Log level
        if config.gitleaks_log_level != "info":
            cmd.extend(["--log-level", config.gitleaks_log_level])

        # Additional flags
        if config.ignore_gitleaks_allow:
            cmd.append("--ignore-gitleaks-allow")

        if config.no_banner:
            cmd.append("--no-banner")

        if config.no_color:
            cmd.append("--no-color")

    def convert_to_secret_matches(
        self, gitleaks_findings: List[Dict[str, Any]]
    ) -> List[SecretMatch]:
        """
        Convert Gitleaks raw findings to SecretMatch objects

        Args:
            gitleaks_findings: Raw findings from Gitleaks

        Returns:
            List of SecretMatch objects
        """
        matches = []

        for finding in gitleaks_findings:
            try:
                # Map Gitleaks rule to our secret type
                secret_type = self._map_rule_to_secret_type(finding.get("RuleID", ""))

                # Determine file type
                file_type = self._determine_file_type(finding.get("File", ""))

                # Calculate confidence based on entropy and rule
                confidence = self._calculate_initial_confidence(finding)
                confidence_level = self._get_confidence_level(confidence)

                match = SecretMatch(
                    secret_type=secret_type,
                    matched_text=finding.get("Secret", ""),
                    line_number=finding.get("StartLine", 0),
                    column_start=finding.get("StartColumn", 0),
                    column_end=finding.get("EndColumn", 0),
                    file_path=finding.get("File", ""),
                    file_type=file_type,
                    surrounding_context=finding.get("Line", ""),
                    detection_method=DetectionMethod.REGEX_MATCH,
                    confidence=confidence,
                    confidence_level=confidence_level,
                    entropy_score=finding.get("Entropy", 0.0),
                    pattern_matched=finding.get("RuleID", ""),
                    # Store original Gitleaks data for reference
                    context_analysis={
                        "gitleaks_original": finding,
                        "commit": finding.get("Commit", ""),
                        "author": finding.get("Author", ""),
                        "fingerprint": finding.get("Fingerprint", ""),
                    },
                )

                matches.append(match)

            except Exception as e:
                logger.warning(f"Failed to convert Gitleaks finding: {e}")
                continue

        return matches

    def _map_rule_to_secret_type(self, rule_id: str) -> SecretType:
        """Map Gitleaks rule ID to our SecretType enum"""
        rule_mapping = {
            "aws-access-key-id": SecretType.API_KEY,
            "aws-secret-access-key": SecretType.API_KEY,
            "github-pat": SecretType.ACCESS_TOKEN,
            "github-oauth": SecretType.OAUTH_TOKEN,
            "slack-access-token": SecretType.ACCESS_TOKEN,
            "slack-bot-token": SecretType.ACCESS_TOKEN,
            "jwt": SecretType.JWT_TOKEN,
            "private-key": SecretType.PRIVATE_KEY,
            "rsa-private-key": SecretType.PRIVATE_KEY,
            "generic-api-key": SecretType.API_KEY,
            "password": SecretType.PASSWORD,
            "postgres": SecretType.DATABASE_URL,
            "mysql": SecretType.DATABASE_URL,
            "mongodb": SecretType.DATABASE_URL,
        }

        # Try exact match first
        if rule_id in rule_mapping:
            return rule_mapping[rule_id]

        # Try partial matching
        rule_lower = rule_id.lower()
        if "key" in rule_lower:
            return SecretType.API_KEY
        elif "token" in rule_lower:
            return SecretType.ACCESS_TOKEN
        elif "password" in rule_lower:
            return SecretType.PASSWORD
        elif "private" in rule_lower:
            return SecretType.PRIVATE_KEY
        else:
            return SecretType.GENERIC_SECRET

    def _determine_file_type(self, file_path: str) -> FileType:
        """Determine file type based on file extension and path"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Source code files
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

        # Configuration files
        config_extensions = {".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".config"}
        config_names = {"dockerfile", "makefile", ".env", ".gitignore"}
        if suffix in config_extensions or path.name.lower() in config_names:
            return FileType.CONFIG_FILE

        # Documentation
        doc_extensions = {".md", ".txt", ".rst", ".adoc"}
        if suffix in doc_extensions:
            return FileType.DOCUMENTATION

        # Data files
        data_extensions = {".csv", ".sql", ".db", ".sqlite"}
        if suffix in data_extensions:
            return FileType.DATA_FILE

        return FileType.UNKNOWN

    def _calculate_initial_confidence(self, finding: Dict[str, Any]) -> float:
        """Calculate initial confidence score based on Gitleaks data"""
        base_confidence = 0.7  # Base confidence for Gitleaks matches

        # Adjust based on entropy
        entropy = finding.get("Entropy", 0.0)
        if entropy > 5.0:
            base_confidence += 0.2
        elif entropy > 4.0:
            base_confidence += 0.1
        elif entropy < 3.0:
            base_confidence -= 0.2

        # Adjust based on rule specificity
        rule_id = finding.get("RuleID", "").lower()
        if "generic" in rule_id:
            base_confidence -= 0.1
        elif any(specific in rule_id for specific in ["aws", "github", "slack"]):
            base_confidence += 0.1

        return min(1.0, max(0.1, base_confidence))

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
