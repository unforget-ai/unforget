"""Unit tests for entity extraction — no database needed."""

from unforget.entities import extract_entities


class TestKnownEntities:
    def test_tech_terms_lowercase(self):
        entities = extract_entities("we use postgresql and redis for caching")
        assert "postgresql" in entities
        assert "redis" in entities

    def test_tech_terms_mixed_case(self):
        entities = extract_entities("Deploy to AWS using Docker")
        assert "aws" in entities
        assert "docker" in entities

    def test_cloud_providers(self):
        entities = extract_entities("migrated from heroku to fly.io last week")
        assert "heroku" in entities
        assert "fly.io" in entities

    def test_languages(self):
        entities = extract_entities("the backend is python, frontend is typescript")
        assert "python" in entities
        assert "typescript" in entities

    def test_ai_tools(self):
        entities = extract_entities("switched from openai to anthropic claude")
        assert "openai" in entities
        assert "anthropic" in entities
        assert "claude" in entities

    def test_frameworks(self):
        entities = extract_entities("API built with fastapi, frontend with react")
        assert "fastapi" in entities
        assert "react" in entities


class TestDomainPatterns:
    def test_dotio(self):
        entities = extract_entities("deployed to fly.io successfully")
        assert "fly.io" in entities

    def test_dotcom(self):
        entities = extract_entities("check github.com for the repo")
        assert "github.com" in entities

    def test_dotdev(self):
        entities = extract_entities("documentation at docs.ninetrix.dev")
        assert "docs.ninetrix.dev" in entities


class TestCapitalizedWords:
    def test_proper_nouns(self):
        """Capitalized words mid-sentence are extracted as entities."""
        entities = extract_entities("the project uses Kubernetes for orchestration")
        assert "kubernetes" in entities

    def test_skips_sentence_start(self):
        """First word of sentence is skipped (always capitalized)."""
        entities = extract_entities("The server runs on Linux")
        assert "the" not in entities
        assert "linux" in entities

    def test_skips_stopwords(self):
        entities = extract_entities("We talked With John About the Deploy")
        assert "with" not in entities
        assert "about" not in entities
        assert "john" in entities


class TestVersionStrings:
    def test_version_with_prefix(self):
        entities = extract_entities("upgraded to Python 3.12 last week")
        assert "python 3.12" in entities

    def test_version_standalone(self):
        entities = extract_entities("running version 2.1.0 in prod")
        assert "2.1.0" in entities

    def test_v_prefix(self):
        entities = extract_entities("released v4.0 today")
        assert "v4.0" in entities


class TestEdgeCases:
    def test_empty_string(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        entities = extract_entities("the quick brown fox jumps over the lazy dog")
        # No known entities, no capitalized proper nouns mid-sentence
        assert len(entities) == 0

    def test_deduplication(self):
        entities = extract_entities("Python is great. I love python. python rocks")
        assert entities.count("python") == 1

    def test_sorted(self):
        entities = extract_entities("uses Redis, PostgreSQL, and AWS")
        assert entities == sorted(entities)

    def test_multiple_sentences(self):
        text = "We use Python for backend. The API runs on FastAPI. Database is PostgreSQL."
        entities = extract_entities(text)
        assert "python" in entities
        assert "fastapi" in entities
        assert "postgresql" in entities


class TestEntityInRetrieval:
    """Test that entities stored on write help retrieval."""

    def test_query_entity_extraction(self):
        """Entities extracted from queries match stored entities."""
        # Simulating: memory stored with "User deploys to Fly.io"
        stored_entities = extract_entities("User deploys to Fly.io")
        assert "fly.io" in stored_entities

        # Query: "Fly.io deployment"
        query_entities = extract_entities("Fly.io deployment")
        assert "fly.io" in query_entities

        # Overlap exists — entity channel will match
        overlap = set(stored_entities) & set(query_entities)
        assert "fly.io" in overlap
