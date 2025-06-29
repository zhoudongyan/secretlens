"""
Data models for SecretLens
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os

class SecretType(str, Enum):
    """Types of secrets that can be detected"""

    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    PRIVATE_KEY = "private_key"
    PASSWORD = "password"
    DATABASE_URL = "database_url"
    OAUTH_TOKEN = "oauth_token"
    JWT_TOKEN = "jwt_token"
    GENERIC_SECRET = "generic_secret"


class ConfidenceLevel(str, Enum):
    """Confidence levels for detection results"""

    HIGH = "high"  # 90-100%
    MEDIUM = "medium"  # 70-89%
    LOW = "low"  # 50-69%
    VERY_LOW = "very_low"  # <50%


class DetectionMethod(str, Enum):
    """Methods used for detection"""

    REGEX_MATCH = "regex_match"
    ENTROPY_ANALYSIS = "entropy_analysis"
    LLM_ANALYSIS = "llm_analysis"
    HYBRID = "hybrid"


class FileType(str, Enum):
    """Supported file types"""

    SOURCE_CODE = "source_code"
    CONFIG_FILE = "config_file"
    DOCUMENTATION = "documentation"
    DATA_FILE = "data_file"
    UNKNOWN = "unknown"


class SecretMatch(BaseModel):
    """A potential secret match found in code"""

    # Basic match info
    secret_type: SecretType
    matched_text: str
    line_number: int
    column_start: int
    column_end: int

    # Context information
    file_path: str
    file_type: FileType
    surrounding_context: str = Field(description="Context around the match")

    # Detection metadata
    detection_method: DetectionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel

    # LLM analysis results
    llm_reasoning: Optional[str] = None
    is_likely_false_positive: bool = False
    context_analysis: Optional[Dict[str, Any]] = None

    # Additional metadata
    entropy_score: Optional[float] = None
    pattern_matched: Optional[str] = None
    created_at: Optional[str] = None


class ScanConfig(BaseModel):
    """Configuration for scanning"""

    # Scan scope
    target_path: str
    include_patterns: List[str] = ["*"]
    exclude_patterns: List[str] = [".git/*", "node_modules/*", "*.log"]
    max_file_size: int = 1024 * 1024  # 1MB

    # Detection settings
    enable_regex_detection: bool = True
    enable_entropy_analysis: bool = True
    enable_llm_analysis: bool = True
    entropy_threshold: float = 4.5

    # Gitleaks-specific settings
    gitleaks_config_path: Optional[str] = None
    gitleaks_baseline_path: Optional[str] = None
    gitleaks_ignore_path: Optional[str] = None
    enable_rules: List[str] = []  # Specific rules to enable
    scan_mode: str = "auto"  # "auto", "dir", "git"
    max_decode_depth: int = 0
    max_archive_depth: int = 0
    max_target_megabytes: Optional[int] = None
    redact_percentage: int = 100
    gitleaks_log_level: str = "info"
    ignore_gitleaks_allow: bool = False
    no_banner: bool = True
    no_color: bool = False

    # LLM settings
    llm_provider: str = "auto"  # "auto", "openai", "anthropic", "ollama"
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "auto")
    )  # Get from env or default to "auto"
    llm_temperature: float = 0.1
    max_context_length: int = 2000
    llm_base_url: Optional[str] = None  # Custom LLM API base URL (applies to current provider)

    # Performance settings
    max_workers: int = 4
    batch_size: int = 10
    individual_timeout: int = 120  # Timeout for individual LLM analysis (seconds)
    batch_timeout: int = 600  # Timeout for batch processing (seconds)
    max_retries: int = 2  # Maximum retries for failed LLM analyses

    # Output settings
    output_format: str = "json"
    include_low_confidence: bool = False
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for results"
    )
    verbose: bool = False
    analysis_only: bool = False
    enable_additional_discovery: bool = False  # Enable LLM-based additional secret discovery
    max_discovery_files: int = 50  # Maximum files to analyze for additional secrets


class ScanResult(BaseModel):
    """Complete scan results"""

    config: ScanConfig
    matches: List[SecretMatch]
    scan_summary: Dict[str, Any]
    total_files_scanned: int
    total_matches_found: int
    scan_duration_seconds: float
    timestamp: str
