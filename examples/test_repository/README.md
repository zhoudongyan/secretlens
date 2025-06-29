# Test Repository for SecretLens

This directory contains test files designed to validate SecretLens' ability to:

1. **Detect real secrets** - Properly formatted API keys, database URLs, etc.
2. **Filter false positives** - Test data, examples, templates
3. **Understand context** - Distinguish between real and fake credentials based on context

## Test Files

- `config.py` - Configuration file with mixed real and fake secrets
- `test_secrets.txt` - Plain text file with various secret patterns
- `docker-compose.yml` - Docker configuration with environment variables

## Expected Behavior

### Should be detected as HIGH confidence:

- Real-looking API keys in production contexts
- Database URLs with realistic credentials
- Private keys without test indicators

### Should be filtered as FALSE POSITIVES:

- Obviously fake/test values
- Configuration templates
- Example/documentation snippets
- Mock data clearly marked as such

### Should be CONTEXT-DEPENDENT:

- Development/staging credentials
- Base64 encoded data (depends on content)
- JWT tokens (depends on context)

## Usage

Run SecretLens on this directory to test detection accuracy:

```bash
python -m secretlens enhance examples/test_repository --verbose
```
