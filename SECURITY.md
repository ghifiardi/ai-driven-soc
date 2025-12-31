# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the security team directly (configure your security contact email)
3. Include detailed steps to reproduce the vulnerability
4. Allow reasonable time for the issue to be addressed before public disclosure

## Security Practices

### Secrets Management

This project follows strict secrets management practices:

1. **No hardcoded secrets**: All sensitive values must be provided via environment variables
2. **Production enforcement**: Critical secrets (JWT_SECRET, API keys) will cause runtime errors if not set in production
3. **Development fallbacks**: Development environments use ephemeral secrets that change on restart

### Required Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JWT_SECRET` | Yes (production) | JWT signing secret. Generate with: `openssl rand -hex 32` |
| `SERVICE_API_KEY` | Yes (production) | Service-to-service authentication key |
| `FLASK_SECRET_KEY` | Yes (production) | Flask session secret key |
| `GEMINI_API_KEY` | Optional | Google Gemini API key for AI features |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to GCP service account JSON |

### Generating Secure Secrets

```bash
# Generate a secure JWT secret
openssl rand -hex 32

# Generate a secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate a Flask secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

### Pre-commit Hooks

Install pre-commit hooks to catch security issues before committing:

```bash
pip install pre-commit
pre-commit install
```

The hooks will:
- Detect private keys
- Detect AWS credentials
- Run Bandit security linter
- Check for large files

### CI/CD Security

Our CI/CD pipeline includes:

1. **Bandit**: Static security analysis for Python
2. **Safety/pip-audit**: Dependency vulnerability scanning
3. **Gitleaks**: Secret detection in git history
4. **Trivy**: Container image vulnerability scanning
5. **CodeQL**: GitHub's semantic code analysis

### Dependency Security

- All dependencies are pinned to specific versions in `requirements.txt`
- Weekly automated dependency updates via Dependabot (when configured)
- Security scanning runs on every PR

## Security Checklist for Contributors

Before submitting a PR:

- [ ] No secrets or API keys in code
- [ ] Environment variables used for all sensitive configuration
- [ ] Input validation for all user-provided data
- [ ] Parameterized queries for database operations
- [ ] HTTPS/TLS for all external communications
- [ ] Proper authentication/authorization checks
- [ ] No debug mode enabled in production code
- [ ] Logging does not include sensitive data

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |
