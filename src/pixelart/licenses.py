from __future__ import annotations

OPEN_LICENSE_TOKENS = [
    "cc0",
    "creative commons zero",
    "public domain",
    "cc-by",
    "cc by",
    "cc-by-sa",
    "cc by-sa",
    "apache-2.0",
    "apache 2.0",
    "mit",
    "bsd",
    "unlicense",
]


def normalize_license(license_name: str) -> str:
    return " ".join(license_name.strip().split()).lower()


def is_open_license(license_name: str) -> bool:
    normalized = normalize_license(license_name)
    return any(token in normalized for token in OPEN_LICENSE_TOKENS)

