"""Lightweight named entity extraction — no LLM required.

Extracts entities using regex patterns and heuristics. Fast (~1ms).
Phase 4 adds optional spaCy support for better accuracy.

Entity types detected:
- Capitalized proper nouns (Fly.io, Python, AWS, PostgreSQL)
- Domain-like words (fly.io, github.com)
- Version strings (v3.12, Python 3.11)
- Known tech terms (case-insensitive matching against a curated list)
"""

from __future__ import annotations

import re

# Common tech terms that should be recognized as entities regardless of casing.
# Keep this list focused — it's for high-value terms agents commonly reference.
KNOWN_ENTITIES: set[str] = {
    # Cloud & infra
    "aws", "gcp", "azure", "fly.io", "vercel", "netlify", "heroku", "railway",
    "docker", "kubernetes", "k8s", "terraform", "pulumi", "cloudflare",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite",
    "dynamodb", "elasticsearch", "supabase", "neon", "pgvector",
    # Languages & runtimes
    "python", "javascript", "typescript", "golang", "rust", "java", "ruby",
    "node", "nodejs", "deno", "bun",
    # Frameworks
    "react", "nextjs", "vue", "angular", "django", "flask", "fastapi",
    "express", "rails", "spring", "laravel",
    # AI/ML
    "openai", "anthropic", "claude", "gpt", "gemini", "mistral", "ollama",
    "langchain", "langgraph", "pytorch", "tensorflow",
    # Tools
    "git", "github", "gitlab", "jira", "slack", "notion", "linear",
    "nginx", "caddy", "grafana", "prometheus", "datadog", "sentry",
    # Protocols
    "grpc", "graphql", "websocket", "oauth", "jwt", "saml",
}

# Stopwords to skip (common English words that happen to be capitalized at sentence start)
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    "and", "or", "but", "not", "nor", "so", "yet", "both", "either", "neither",
    "it", "its", "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "this", "that", "these", "those", "here", "there", "where", "when",
    "what", "which", "who", "whom", "how", "why",
    "if", "then", "else", "than", "because", "since", "while", "although",
    "also", "just", "only", "very", "too", "quite", "rather",
    "all", "each", "every", "any", "some", "no", "many", "few", "much",
    "more", "most", "other", "another", "such",
    "new", "old", "first", "last", "next", "same", "different",
    "now", "today", "yesterday", "tomorrow", "always", "never", "often",
    "use", "using", "used", "like", "want", "need", "make", "run", "set",
    "get", "got", "let", "try", "keep", "start", "stop", "check",
}

# Patterns
_DOMAIN_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:io|com|org|net|dev|app|ai|co)\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"\b(?:v?\d+\.\d+(?:\.\d+)?)\b")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:[.-][a-zA-Z0-9]+)*\b")

# Person name pattern: sequences of 2-3 capitalized words (e.g., "Caroline Smith", "John")
# Excludes common non-name words that appear capitalized mid-sentence
_NAME_EXCLUDE: set[str] = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "yes", "no", "ok", "hey", "hi", "hello", "thanks", "sure", "well",
    "oh", "wow", "great", "good", "nice", "cool", "right", "true", "false",
}

# Date patterns for temporal extraction
_DATE_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:[,\s]+(\d{4}))?\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:[,\s]+(\d{4}))\b",
    re.IGNORECASE,
)
_RELATIVE_DATE_RE = re.compile(
    r"\b(yesterday|last\s+(?:week|month|year)|next\s+(?:week|month|year)|"
    r"last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text using heuristics.

    Returns lowercase, deduplicated entity list.
    Extracts: tech terms, domains, versions, person names, and date references.
    """
    entities: set[str] = set()

    # 1. Known tech terms (case-insensitive)
    words = re.findall(r"[\w.-]+", text.lower())
    for word in words:
        if word in KNOWN_ENTITIES:
            entities.add(word)

    # 2. Domain-like patterns (fly.io, github.com)
    for match in _DOMAIN_RE.finditer(text):
        entities.add(match.group().lower())

    # 3. Version strings (Python 3.12, v2.1.0)
    for match in _VERSION_RE.finditer(text):
        start = match.start()
        prefix = text[:start].rstrip()
        prefix_word = prefix.split()[-1] if prefix else ""
        if prefix_word and prefix_word[0].isupper():
            entities.add(f"{prefix_word.lower()} {match.group()}")
        else:
            entities.add(match.group())

    # 4. Person names — capitalized words, including after speaker labels ("Caroline:")
    #    and at sentence boundaries. Captures multi-word names like "John Smith".
    #    Split on sentence boundaries AND speaker labels (e.g., "Caroline: Hey...")
    segments = re.split(r"[.!?]\s+|\n", text)
    for segment in segments:
        words_in_segment = segment.split()
        i = 0
        while i < len(words_in_segment):
            word = words_in_segment[i]

            # Skip words that are speaker labels ending with ":"
            if word.endswith(":") and len(word) > 2:
                # Extract the speaker name itself (before the colon)
                speaker = word[:-1]
                if speaker[0].isupper() and speaker.lower() not in _STOPWORDS:
                    entities.add(speaker.lower())
                i += 1
                continue

            # After timestamp brackets like [8:02 PM] skip non-name content
            if word.startswith("[") or word.endswith("]"):
                i += 1
                continue

            cap_match = _CAPITALIZED_RE.match(word)
            if cap_match:
                candidate = cap_match.group()
                lower = candidate.lower()

                if lower not in _STOPWORDS and lower not in _NAME_EXCLUDE and len(lower) > 1:
                    # Try to capture multi-word names (e.g., "John Smith")
                    name_parts = [candidate]
                    j = i + 1
                    while j < len(words_in_segment):
                        next_word = words_in_segment[j]
                        # Stop at punctuation, speaker labels, or non-capitalized words
                        if next_word.endswith(":") or next_word.startswith("["):
                            break
                        next_match = _CAPITALIZED_RE.match(next_word)
                        if (
                            next_match
                            and next_match.group().lower() not in _STOPWORDS
                            and next_match.group().lower() not in _NAME_EXCLUDE
                        ):
                            name_parts.append(next_match.group())
                            j += 1
                            if len(name_parts) >= 3:
                                break
                        else:
                            break

                    # Add full name and individual parts
                    if len(name_parts) >= 2:
                        entities.add(" ".join(name_parts).lower())
                    for part in name_parts:
                        part_lower = part.lower()
                        if part_lower not in _NAME_EXCLUDE and len(part_lower) > 1:
                            entities.add(part_lower)
                    i = j
                    continue
            i += 1

    # 5. Date references — extract month/year mentions for temporal matching
    for match in _DATE_MONTH_YEAR_RE.finditer(text):
        month = match.group(2).lower()
        year = match.group(3)
        entities.add(month)
        if year:
            entities.add(f"{month} {year}")

    for match in _MONTH_YEAR_RE.finditer(text):
        month = match.group(1).lower()
        year = match.group(2)
        entities.add(month)
        entities.add(f"{month} {year}")

    # Remove any empty strings
    entities.discard("")

    return sorted(entities)
