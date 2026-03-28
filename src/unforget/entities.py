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


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text using heuristics.

    Returns lowercase, deduplicated entity list.
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
        # Try to capture preceding word (e.g., "Python 3.12")
        start = match.start()
        prefix = text[:start].rstrip()
        prefix_word = prefix.split()[-1] if prefix else ""
        if prefix_word and prefix_word[0].isupper():
            entities.add(f"{prefix_word.lower()} {match.group()}")
        else:
            entities.add(match.group())

    # 4. Capitalized words (proper nouns) — skip sentence-initial and stopwords
    sentences = re.split(r"[.!?]\s+", text)
    for sentence in sentences:
        words_in_sentence = sentence.split()
        for i, word in enumerate(words_in_sentence):
            # Skip first word of sentence (always capitalized)
            if i == 0:
                continue
            for cap_match in _CAPITALIZED_RE.finditer(word):
                candidate = cap_match.group()
                lower = candidate.lower()
                if lower not in _STOPWORDS and len(lower) > 1:
                    entities.add(lower)

    # Remove any empty strings
    entities.discard("")

    return sorted(entities)
