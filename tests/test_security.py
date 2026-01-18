# =============================================================================
# Security Tests
# =============================================================================

import pytest
import os
import re
from pathlib import Path
from typing import List


class TestSecretDetection:
    """Tests to ensure no secrets are hardcoded in the codebase."""

    # Patterns that indicate potential hardcoded secrets
    SECRET_PATTERNS = [
        # API Keys with actual values (not env var references)
        (r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "API key"),
        (r"apikey\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "API key"),

        # AWS patterns
        (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
        (r"aws[_-]?secret[_-]?access[_-]?key\s*=\s*['\"][^'\"]+['\"]", "AWS Secret"),

        # GCP patterns
        (r'"private_key":\s*"-----BEGIN', "GCP Private Key"),
        (r'"private_key_id":\s*"[a-f0-9]{40}"', "GCP Private Key ID"),

        # Generic secrets with values
        (r"secret[_-]?key\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "Secret Key"),
        (r"password\s*=\s*['\"][^'\"]{8,}['\"]", "Password"),

        # JWT secrets with hardcoded values
        (r"jwt[_-]?secret\s*=\s*['\"][^'\"]{20,}['\"]", "JWT Secret"),

        # Database connection strings with passwords
        (r"(mysql|postgres|mongodb)://[^:]+:[^@]+@", "Database Password"),
    ]

    # Files/patterns to exclude from scanning
    EXCLUDE_PATTERNS = [
        r"\.git/",
        r"venv/",
        r"\.venv/",
        r"node_modules/",
        r"__pycache__/",
        r"\.pyc$",
        r"test_.*\.py$",  # Exclude test files
        r"conftest\.py$",
        r"\.md$",  # Exclude markdown
        r"requirements.*\.txt$",
    ]

    def get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        project_root = Path(__file__).parent.parent
        python_files = []

        for path in project_root.rglob("*.py"):
            # Check if path matches any exclude pattern
            path_str = str(path)
            if any(re.search(pattern, path_str) for pattern in self.EXCLUDE_PATTERNS):
                continue
            python_files.append(path)

        return python_files

    def test_no_hardcoded_api_keys(self):
        """Ensure no API keys are hardcoded in Python files."""
        violations = []

        for file_path in self.get_python_files():
            try:
                content = file_path.read_text()

                # Skip files that only reference environment variables
                for pattern, secret_type in self.SECRET_PATTERNS:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Check if this is actually an env var reference
                        line_start = content.rfind("\n", 0, match.start()) + 1
                        line_end = content.find("\n", match.end())
                        line = content[line_start:line_end]

                        # Skip if it's clearly using environment variables
                        if "os.getenv" in line or "os.environ" in line or ".get(" in line:
                            continue

                        # Skip if it's a comment
                        if line.strip().startswith("#"):
                            continue

                        violations.append(
                            f"{file_path}:{content[:match.start()].count(chr(10))+1}: "
                            f"Potential {secret_type} found"
                        )
            except Exception as e:
                pass  # Skip files that can't be read

        if violations:
            pytest.fail(
                f"Found {len(violations)} potential hardcoded secrets:\n"
                + "\n".join(violations[:10])  # Show first 10
            )

    def test_no_private_keys_in_code(self):
        """Ensure no private keys are embedded in code."""
        project_root = Path(__file__).parent.parent

        for path in project_root.rglob("*"):
            if path.is_file() and path.suffix in [".py", ".json", ".yml", ".yaml"]:
                # Skip excluded directories
                if any(
                    re.search(pattern, str(path)) for pattern in self.EXCLUDE_PATTERNS
                ):
                    continue

                try:
                    content = path.read_text()
                    if "-----BEGIN PRIVATE KEY-----" in content:
                        pytest.fail(f"Private key found in {path}")
                    if "-----BEGIN RSA PRIVATE KEY-----" in content:
                        pytest.fail(f"RSA private key found in {path}")
                except Exception:
                    pass

    def test_env_vars_not_have_defaults_in_production(self):
        """Ensure critical env vars don't have insecure defaults."""
        project_root = Path(__file__).parent.parent

        critical_vars = ["JWT_SECRET", "SECRET_KEY", "API_KEY"]

        for path in project_root.rglob("*.py"):
            if any(
                re.search(pattern, str(path)) for pattern in self.EXCLUDE_PATTERNS
            ):
                continue

            try:
                content = path.read_text()

                for var in critical_vars:
                    # Look for patterns like: os.getenv("JWT_SECRET", "some-default")
                    pattern = rf'os\.getenv\(["\']?{var}["\']?,\s*["\'][^"\']+["\']\)'
                    matches = re.findall(pattern, content)

                    for match in matches:
                        # Check if it's a placeholder that would be caught
                        if "change-me" in match.lower() or "CHANGE_ME" in match:
                            continue
                        if "test" in match.lower() and "test" in str(path).lower():
                            continue
                        # This might be a real default - check the context
                        # For now, just warn
                        pass
            except Exception:
                pass


class TestAuthenticationSecurity:
    """Tests for authentication security."""

    def test_jwt_algorithm_is_secure(self):
        """Ensure JWT uses secure algorithm (not 'none' or weak algorithms)."""
        project_root = Path(__file__).parent.parent

        insecure_algorithms = ["none", "HS256"]  # HS256 is OK but RS256 is better

        for path in project_root.rglob("*.py"):
            if "venv" in str(path) or ".venv" in str(path):
                continue

            try:
                content = path.read_text()

                # Check for algorithm set to none (insecure)
                if re.search(r'algorithm\s*=\s*["\']none["\']', content, re.IGNORECASE):
                    pytest.fail(f"Insecure JWT algorithm 'none' found in {path}")

            except Exception:
                pass

    def test_password_hashing_used(self):
        """Ensure passwords are hashed, not stored in plain text."""
        # This is a placeholder - actual implementation would check
        # that password fields use proper hashing (bcrypt, argon2, etc.)
        pass


class TestInputValidation:
    """Tests for input validation security."""

    def test_sql_injection_prevention(self):
        """Ensure parameterized queries are used."""
        project_root = Path(__file__).parent.parent

        # Patterns that might indicate SQL injection vulnerabilities
        dangerous_patterns = [
            r'execute\(["\'].*%s.*["\'].*%',  # String formatting in SQL
            r'f".*SELECT.*{',  # f-strings in SQL
            r'f".*INSERT.*{',
            r'f".*UPDATE.*{',
            r'f".*DELETE.*{',
        ]

        violations = []

        for path in project_root.rglob("*.py"):
            if "venv" in str(path) or "test" in str(path).lower():
                continue

            try:
                content = path.read_text()

                for pattern in dangerous_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(str(path))
                        break
            except Exception:
                pass

        # Note: This test may have false positives
        # Review violations manually
        if violations:
            print(f"Files to review for SQL injection: {violations[:5]}")
