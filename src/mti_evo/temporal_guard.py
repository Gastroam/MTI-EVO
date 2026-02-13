"""
MTI Temporal Validation Lobe (LVT) - VersionGuard
==================================================

The "Cronos Gatekeeper" that fights LLM frequency bias.

Problem: LLMs trained on historical data suggest outdated libraries.
Solution: Real-time version validation against NPM/PyPI registries.

Components:
- VersionGuard: Intercepts proposals and validates versions
- TemporalContext: Injects current date and version constraints
- RegistryCache: Cached lookups to NPM/PyPI for latest versions
- CronosValidator: Validates code proposals against project reality
"""

import os
import re
import json
import time
import asyncio
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, date
from functools import lru_cache

logger = logging.getLogger(__name__)


# Current date - the "truth" anchor
CURRENT_DATE = date(2026, 1, 17)
CURRENT_YEAR = 2026


@dataclass
class PackageInfo:
    """Information about a package in the project."""
    name: str
    installed_version: str
    latest_version: Optional[str] = None
    is_outdated: bool = False
    major_behind: int = 0
    registry: str = "npm"  # npm, pypi
    last_checked: float = field(default_factory=time.time)


@dataclass
class VersionValidation:
    """Result of version validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    temporal_context: str = ""


# Forbidden/outdated patterns - libraries that should NOT be suggested
FORBIDDEN_PATTERNS = {
    # JavaScript/TypeScript
    "moment": {"reason": "Use date-fns v4+ or native Temporal API", "since": 2022},
    "request": {"reason": "Use native fetch or axios", "since": 2020},
    "jquery": {"reason": "Use vanilla JS or React", "since": 2020},
    "lodash": {"reason": "Use native ES6+ methods unless specific need", "since": 2023},
    "enzyme": {"reason": "Use React Testing Library", "since": 2022},
    "create-react-app": {"reason": "Use Vite or Next.js", "since": 2023},
    "tslint": {"reason": "Use ESLint with TypeScript plugin", "since": 2020},
    "node-sass": {"reason": "Use sass (dart-sass)", "since": 2022},
    
    # Python
    "urllib2": {"reason": "Use requests or httpx", "since": 2018},
    "optparse": {"reason": "Use argparse", "since": 2015},
    "imp": {"reason": "Use importlib", "since": 2018},
    "asyncore": {"reason": "Use asyncio", "since": 2020},
    
    # Deprecated React patterns
    "componentWillMount": {"reason": "Use useEffect hook", "since": 2019},
    "componentWillReceiveProps": {"reason": "Use useEffect with deps", "since": 2019},
    "componentWillUpdate": {"reason": "Use useEffect", "since": 2019},
}

# Version hints for common libraries (minimum recommended versions for 2026)
MINIMUM_VERSIONS = {
    # Node/JS ecosystem (2026 standards)
    "react": "19.0.0",
    "next": "15.0.0",
    "vite": "6.0.0",
    "typescript": "5.4.0",
    "tailwindcss": "4.0.0",
    "eslint": "9.0.0",
    "node": "22.0.0",
    
    # Python ecosystem
    "python": "3.12",
    "fastapi": "0.115.0",
    "pydantic": "2.9.0",
    "httpx": "0.28.0",
    "sqlalchemy": "2.0.0",
    
    # AI/ML
    "torch": "2.5.0",
    "transformers": "4.47.0",
    "langchain": "0.3.0",
}


class RegistryCache:
    """
    Cached lookups to package registries.
    
    Caches NPM/PyPI responses to avoid rate limiting.
    Cache TTL: 1 hour for version lookups.
    """
    
    CACHE_TTL = 3600  # 1 hour
    
    def __init__(self):
        self._cache: Dict[str, Tuple[str, float]] = {}  # name -> (version, timestamp)
        self._npm_available = self._check_npm()
        self._pip_available = self._check_pip()
    
    def _check_npm(self) -> bool:
        """Check if npm is available."""
        try:
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True, text=True, timeout=5,
                shell=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_pip(self) -> bool:
        """Check if pip is available."""
        try:
            result = subprocess.run(
                ["pip", "--version"],
                capture_output=True, text=True, timeout=5,
                shell=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_latest_npm(self, package: str) -> Optional[str]:
        """Get latest version from NPM registry."""
        cache_key = f"npm:{package}"
        
        # Check cache
        if cache_key in self._cache:
            version, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TTL:
                return version
        
        if not self._npm_available:
            return None
        
        try:
            result = subprocess.run(
                ["npm", "view", package, "version"],
                capture_output=True, text=True, timeout=10,
                shell=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self._cache[cache_key] = (version, time.time())
                return version
        except Exception as e:
            logger.debug(f"NPM lookup failed for {package}: {e}")
        
        return None
    
    def get_latest_pypi(self, package: str) -> Optional[str]:
        """Get latest version from PyPI registry."""
        cache_key = f"pypi:{package}"
        
        # Check cache
        if cache_key in self._cache:
            version, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.CACHE_TTL:
                return version
        
        if not self._pip_available:
            return None
        
        try:
            result = subprocess.run(
                ["pip", "index", "versions", package],
                capture_output=True, text=True, timeout=10,
                shell=True
            )
            if result.returncode == 0:
                # Parse output like "Available versions: 1.0.0, 0.9.0, ..."
                match = re.search(r'Available versions:\s*(\S+)', result.stdout)
                if match:
                    version = match.group(1).rstrip(',')
                    self._cache[cache_key] = (version, time.time())
                    return version
        except Exception as e:
            logger.debug(f"PyPI lookup failed for {package}: {e}")
        
        return None


class VersionGuard:
    """
    The Temporal Gatekeeper - validates proposals against modern standards.
    
    Intercepts LLM output in the Critic layer and:
    1. Detects forbidden/outdated library suggestions
    2. Validates version numbers against minimums
    3. Queries registries for latest versions
    4. Forces justification for non-latest versions
    """
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.registry = RegistryCache()
        self.project_packages: Dict[str, PackageInfo] = {}
        self._scanned = False
    
    def scan_project_dependencies(self) -> int:
        """
        Scan package.json and requirements.txt for current versions.
        """
        count = 0
        
        # Scan package.json (Node/JS)
        pkg_json = self.workspace / "package.json"
        if pkg_json.exists():
            try:
                with open(pkg_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                for name, version in deps.items():
                    # Clean version string (remove ^, ~, etc)
                    clean_version = re.sub(r'^[\^~>=<]+', '', str(version))
                    self.project_packages[name] = PackageInfo(
                        name=name,
                        installed_version=clean_version,
                        registry="npm"
                    )
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")
        
        # Scan requirements.txt (Python)
        requirements = self.workspace / "requirements.txt"
        if requirements.exists():
            try:
                with open(requirements, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse "package==version" or "package>=version"
                            match = re.match(r'^([a-zA-Z0-9_-]+)[=<>]+(.+)$', line)
                            if match:
                                name, version = match.groups()
                                self.project_packages[name] = PackageInfo(
                                    name=name,
                                    installed_version=version.strip(),
                                    registry="pypi"
                                )
                                count += 1
            except Exception as e:
                logger.warning(f"Failed to parse requirements.txt: {e}")
        
        self._scanned = True
        logger.info(f"[VersionGuard] Scanned {count} project dependencies")
        return count
    
    def validate_proposal(self, code_proposal: str) -> VersionValidation:
        """
        Validate a code proposal for temporal correctness.
        
        This is the core interception point in the Critic layer.
        
        Args:
            code_proposal: LLM-generated code to validate
            
        Returns:
            VersionValidation with issues and suggestions
        """
        if not self._scanned:
            self.scan_project_dependencies()
        
        issues = []
        suggestions = []
        blocked = []
        
        # Check for forbidden patterns
        for pattern, info in FORBIDDEN_PATTERNS.items():
            if pattern.lower() in code_proposal.lower():
                # Context check - is it actually using it or just mentioning?
                if re.search(rf'\b{re.escape(pattern)}\b', code_proposal, re.IGNORECASE):
                    issues.append(
                        f"⚠️ Outdated: '{pattern}' deprecated since {info['since']}. "
                        f"Reason: {info['reason']}"
                    )
                    blocked.append(pattern)
                    suggestions.append(info['reason'])
        
        # Check for version numbers in imports
        # Pattern: package@version or package==version
        version_matches = re.findall(
            r'([a-zA-Z0-9_-]+)(?:@|==|>=|~=)(\d+\.\d+(?:\.\d+)?)',
            code_proposal
        )
        
        for pkg_name, suggested_version in version_matches:
            if pkg_name.lower() in [k.lower() for k in MINIMUM_VERSIONS]:
                min_version = MINIMUM_VERSIONS.get(pkg_name.lower(), "")
                if min_version and self._version_lt(suggested_version, min_version):
                    issues.append(
                        f"📅 Version too old: {pkg_name}@{suggested_version} "
                        f"(minimum for 2026: {min_version})"
                    )
                    suggestions.append(f"Use {pkg_name}@{min_version} or higher")
        
        # Check against project reality
        for pkg_name, pkg_info in self.project_packages.items():
            if pkg_name in code_proposal:
                # Check if suggesting a downgrade
                suggested = self._extract_suggested_version(code_proposal, pkg_name)
                if suggested and self._version_lt(suggested, pkg_info.installed_version):
                    issues.append(
                        f"⬇️ Downgrade detected: {pkg_name} {suggested} "
                        f"(project uses {pkg_info.installed_version})"
                    )
        
        # Build temporal context
        temporal_context = self._build_temporal_context()
        
        is_valid = len(issues) == 0
        
        return VersionValidation(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            blocked_patterns=blocked,
            temporal_context=temporal_context
        )
    
    def _version_lt(self, v1: str, v2: str) -> bool:
        """Check if v1 < v2 (simple version comparison)."""
        try:
            parts1 = [int(x) for x in v1.split('.')[:3]]
            parts2 = [int(x) for x in v2.split('.')[:3]]
            
            # Pad to same length
            while len(parts1) < 3:
                parts1.append(0)
            while len(parts2) < 3:
                parts2.append(0)
            
            return parts1 < parts2
        except Exception:
            return False
    
    def _extract_suggested_version(self, code: str, pkg_name: str) -> Optional[str]:
        """Extract version number for a package from code."""
        patterns = [
            rf'{pkg_name}@(\d+\.\d+(?:\.\d+)?)',
            rf'{pkg_name}==(\d+\.\d+(?:\.\d+)?)',
            rf'{pkg_name}["\']:\s*["\'][\^~]?(\d+\.\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _build_temporal_context(self) -> str:
        """Build temporal context string for prompt injection."""
        context_parts = [
            f"TEMPORAL_CONTEXT: Current date is {CURRENT_DATE.strftime('%B %d, %Y')}.",
            f"YEAR: {CURRENT_YEAR}",
        ]
        
        # Add project stack info
        if self.project_packages:
            key_packages = []
            for pkg in ["react", "next", "vite", "python", "fastapi", "node"]:
                if pkg in self.project_packages:
                    info = self.project_packages[pkg]
                    key_packages.append(f"{pkg}@{info.installed_version}")
            
            if key_packages:
                context_parts.append(f"PROJECT_STACK: {', '.join(key_packages)}")
        
        context_parts.append(
            "CONSTRAINT: Do NOT suggest deprecated libraries or syntax from pre-2024. "
            "Use latest stable versions. If suggesting older version, JUSTIFY why."
        )
        
        return " | ".join(context_parts)
    
    def get_outdated_packages(self) -> List[PackageInfo]:
        """Get list of outdated packages in project."""
        if not self._scanned:
            self.scan_project_dependencies()
        
        outdated = []
        for pkg in self.project_packages.values():
            if pkg.registry == "npm":
                latest = self.registry.get_latest_npm(pkg.name)
            else:
                latest = self.registry.get_latest_pypi(pkg.name)
            
            if latest:
                pkg.latest_version = latest
                if self._version_lt(pkg.installed_version, latest):
                    pkg.is_outdated = True
                    outdated.append(pkg)
        
        return outdated
    
    def get_temporal_injection(self) -> str:
        """
        Get the temporal context to inject into LLM prompts.
        This is called by the Router before building system prompts.
        """
        if not self._scanned:
            self.scan_project_dependencies()
        
        return self._build_temporal_context()


class CronosValidator:
    """
    Validates code proposals with temporal awareness.
    
    Integrates with the Critic layer to intercept and fix
    temporally-incorrect suggestions.
    """
    
    def __init__(self, version_guard: VersionGuard):
        self.guard = version_guard
    
    async def validate_and_fix(
        self,
        proposal: str,
        generate_fn=None
    ) -> Tuple[str, VersionValidation]:
        """
        Validate a proposal and optionally regenerate if invalid.
        
        Args:
            proposal: The LLM output to validate
            generate_fn: Optional async function to regenerate
            
        Returns:
            Tuple of (fixed_proposal, validation_result)
        """
        validation = self.guard.validate_proposal(proposal)
        
        if validation.is_valid:
            return proposal, validation
        
        # If we have a generate function and there are issues, try to fix
        if generate_fn and validation.issues:
            fix_prompt = self._build_fix_prompt(proposal, validation)
            try:
                fixed = await generate_fn(fix_prompt)
                # Re-validate
                new_validation = self.guard.validate_proposal(fixed)
                return fixed, new_validation
            except Exception as e:
                logger.warning(f"[CronosValidator] Fix attempt failed: {e}")
        
        return proposal, validation
    
    def _build_fix_prompt(self, proposal: str, validation: VersionValidation) -> str:
        """Build a prompt to fix temporal issues."""
        issues_str = "\n".join(f"- {issue}" for issue in validation.issues)
        suggestions_str = "\n".join(f"- {s}" for s in validation.suggestions)
        
        return f"""[TEMPORAL CORRECTION REQUIRED]

The following code proposal contains outdated patterns or versions:

ISSUES:
{issues_str}

SUGGESTIONS:
{suggestions_str}

{validation.temporal_context}

ORIGINAL PROPOSAL:
```
{proposal[:2000]}
```

Please provide a corrected version using modern, 2026-appropriate libraries and patterns.
Do NOT use any deprecated or oudated libraries. Use latest stable versions.

CORRECTED CODE:"""


@dataclass
class PreInstallValidation:
    """Result of pre-install validation."""
    package_name: str
    requested_version: Optional[str]
    is_allowed: bool
    is_forbidden: bool
    is_latest: bool
    latest_version: Optional[str]
    warning_message: Optional[str] = None
    suggestion: Optional[str] = None
    registry: str = "npm"


class PreInstallValidator:
    """
    Pre-Install Hook - Validates packages BEFORE installation.
    
    Intercepts npm install / pip install commands and:
    1. Checks if package is in forbidden list
    2. Checks if requested version is latest
    3. Suggests modern alternatives
    4. Optionally blocks forbidden packages
    
    Usage:
        validator = PreInstallValidator(version_guard)
        result = validator.validate_install("moment", registry="npm")
        if not result.is_allowed:
            print(f"BLOCKED: {result.warning_message}")
    """
    
    def __init__(self, version_guard: VersionGuard):
        self.guard = version_guard
    
    def validate_install(
        self,
        package_name: str,
        requested_version: Optional[str] = None,
        registry: str = "npm"
    ) -> PreInstallValidation:
        """
        Validate a package before installation.
        
        Args:
            package_name: Name of package to install
            requested_version: Requested version (None = latest)
            registry: "npm" or "pypi"
            
        Returns:
            PreInstallValidation with decision
        """
        # Check if forbidden
        is_forbidden = False
        suggestion = None
        warning = None
        
        pkg_lower = package_name.lower()
        if pkg_lower in [k.lower() for k in FORBIDDEN_PATTERNS]:
            forbidden_info = FORBIDDEN_PATTERNS.get(pkg_lower) or FORBIDDEN_PATTERNS.get(package_name)
            if forbidden_info:
                is_forbidden = True
                suggestion = forbidden_info["reason"]
                warning = (
                    f"🚫 BLOCKED: '{package_name}' is deprecated (since {forbidden_info['since']}). "
                    f"Suggestion: {forbidden_info['reason']}"
                )
        
        # Get latest version
        latest_version = None
        if registry == "npm":
            latest_version = self.guard.registry.get_latest_npm(package_name)
        else:
            latest_version = self.guard.registry.get_latest_pypi(package_name)
        
        # Check if requesting latest
        is_latest = True
        if requested_version and latest_version:
            is_latest = not self.guard._version_lt(requested_version, latest_version)
            if not is_latest:
                if not warning:
                    warning = (
                        f"⚠️ WARNING: Installing {package_name}@{requested_version} "
                        f"but latest is {latest_version}"
                    )
                    suggestion = f"Consider using {package_name}@{latest_version}"
        
        # Check against minimum versions
        if pkg_lower in [k.lower() for k in MINIMUM_VERSIONS]:
            min_version = MINIMUM_VERSIONS.get(pkg_lower, "")
            check_version = requested_version or latest_version
            if check_version and min_version and self.guard._version_lt(check_version, min_version):
                if not warning:
                    warning = (
                        f"📅 WARNING: {package_name}@{check_version} is below "
                        f"2026 minimum ({min_version})"
                    )
        
        # Determine if allowed
        is_allowed = not is_forbidden
        
        return PreInstallValidation(
            package_name=package_name,
            requested_version=requested_version,
            is_allowed=is_allowed,
            is_forbidden=is_forbidden,
            is_latest=is_latest,
            latest_version=latest_version,
            warning_message=warning,
            suggestion=suggestion,
            registry=registry
        )
    
    def validate_install_command(self, command: str) -> List[PreInstallValidation]:
        """
        Parse and validate an install command.
        
        Supports:
        - npm install <package>
        - npm install <package>@<version>
        - pip install <package>
        - pip install <package>==<version>
        
        Args:
            command: Full install command string
            
        Returns:
            List of PreInstallValidation for each package
        """
        results = []
        
        # Detect registry
        registry = "npm" if "npm" in command.lower() else "pypi"
        
        # Parse packages from command
        if registry == "npm":
            # npm install pkg1 pkg2@version pkg3
            packages = re.findall(r'(?:install|add|i)\s+(.+?)(?:\s*$|\s+--)', command)
            if packages:
                pkg_list = packages[0].split()
                for pkg_spec in pkg_list:
                    if pkg_spec.startswith('-'):
                        continue
                    # Parse package@version
                    if '@' in pkg_spec and not pkg_spec.startswith('@'):
                        name, version = pkg_spec.rsplit('@', 1)
                    else:
                        name = pkg_spec
                        version = None
                    results.append(self.validate_install(name, version, registry))
        else:
            # pip install pkg1 pkg2==version pkg3
            packages = re.findall(r'install\s+(.+?)(?:\s*$|\s+-)', command)
            if packages:
                pkg_list = packages[0].split()
                for pkg_spec in pkg_list:
                    if pkg_spec.startswith('-'):
                        continue
                    # Parse package==version
                    if '==' in pkg_spec:
                        name, version = pkg_spec.split('==', 1)
                    elif '>=' in pkg_spec:
                        name, version = pkg_spec.split('>=', 1)
                    else:
                        name = pkg_spec
                        version = None
                    results.append(self.validate_install(name, version, registry))
        
        return results
    
    def get_install_report(self, validations: List[PreInstallValidation]) -> Dict[str, Any]:
        """Generate a report for multiple package validations."""
        blocked = [v for v in validations if not v.is_allowed]
        warnings = [v for v in validations if v.is_allowed and v.warning_message]
        clean = [v for v in validations if v.is_allowed and not v.warning_message]
        
        return {
            "total_packages": len(validations),
            "blocked_count": len(blocked),
            "warning_count": len(warnings),
            "clean_count": len(clean),
            "can_proceed": len(blocked) == 0,
            "blocked": [
                {
                    "package": v.package_name,
                    "reason": v.warning_message,
                    "suggestion": v.suggestion
                }
                for v in blocked
            ],
            "warnings": [
                {
                    "package": v.package_name,
                    "message": v.warning_message,
                    "latest": v.latest_version
                }
                for v in warnings
            ]
        }


# Singleton
_version_guard: Optional[VersionGuard] = None
_pre_install_validator: Optional[PreInstallValidator] = None


def get_version_guard(workspace_path: str = None) -> VersionGuard:
    """Get or create the version guard singleton."""
    global _version_guard
    if _version_guard is None:
        if not workspace_path:
            workspace_path = os.getcwd()
        _version_guard = VersionGuard(workspace_path)
    return _version_guard


def get_pre_install_validator(workspace_path: str = None) -> PreInstallValidator:
    """Get or create the pre-install validator singleton."""
    global _pre_install_validator
    if _pre_install_validator is None:
        guard = get_version_guard(workspace_path)
        _pre_install_validator = PreInstallValidator(guard)
    return _pre_install_validator

