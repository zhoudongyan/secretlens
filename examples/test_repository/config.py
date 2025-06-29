# Test configuration file with various secret patterns
import os

# Real API key (this should be detected as high confidence)
OPENAI_API_KEY = "sk-proj-1234567890abcdef1234567890abcdef1234567890abcdef12"

# Database URL with credentials (should be detected)
DATABASE_URL = "postgresql://user:p@ssw0rd123@db.example.com:5432/myapp"

# AWS credentials (should be detected)
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Test data that should be filtered as false positives
TEST_API_KEY = "test_key_12345"  # Obviously test data
EXAMPLE_PASSWORD = "password123"  # Generic example
MOCK_TOKEN = "mock_token_abcd1234"  # Mock token

# Configuration template (should be false positive)
SECRET_KEY_TEMPLATE = "your_secret_key_here"
API_ENDPOINT_TEMPLATE = "https://api.example.com/v1"

# Development/staging credentials (context-dependent)
DEV_API_KEY = "dev_sk_test_1234567890abcdef"  # Development key
STAGING_DB_URL = "postgresql://testuser:testpass@localhost:5432/test_db"

# Base64 encoded non-secret data (should be false positive)
LOGO_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# JWT token (format looks real but context suggests test)
TEST_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
