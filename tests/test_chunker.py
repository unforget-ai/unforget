"""Unit tests for chunker — no database needed."""

from unforget.chunker import chunk_messages, chunk_text, split_sentences


class TestSplitSentences:
    def test_basic(self):
        text = "Hello world. This is a test. Final sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        assert split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_question_exclamation(self):
        text = "What is this? It's great! Really."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_abbreviations(self):
        text = "Mr. Smith went to Dr. Jones. They talked."
        sentences = split_sentences(text)
        # Should not split on Mr. or Dr.
        assert len(sentences) == 2

    def test_empty(self):
        assert split_sentences("") == []

    def test_no_period(self):
        assert split_sentences("No period here") == ["No period here"]


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_sentences=8)
        assert len(chunks) == 1

    def test_splits_at_max(self):
        sentences = [f"Sentence number {i}." for i in range(10)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_sentences=3, overlap=0)
        assert len(chunks) >= 3

    def test_overlap(self):
        sentences = [f"Sentence {i} here." for i in range(6)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_sentences=3, overlap=1)
        # With overlap=1, chunks should share boundary sentences
        assert len(chunks) >= 2

    def test_empty(self):
        assert chunk_text("") == []

    def test_min_sentences(self):
        text = "One. Two. Three. Four. Five."
        chunks = chunk_text(text, min_sentences=2, max_sentences=3, overlap=0)
        for chunk in chunks:
            sentences = split_sentences(chunk)
            assert len(sentences) >= 2 or chunk == chunks[-1]


class TestChunkMessages:
    def test_basic_conversation(self):
        messages = [
            {"role": "user", "content": "Hello there. How are you?"},
            {"role": "assistant", "content": "I'm good. What can I help with?"},
            {"role": "user", "content": "I need help with Python. Specifically FastAPI."},
        ]
        chunks = chunk_messages(messages, max_sentences=8)
        assert len(chunks) >= 1
        assert "user:" in chunks[0].lower()

    def test_empty_messages(self):
        assert chunk_messages([]) == []

    def test_single_message(self):
        messages = [{"role": "user", "content": "Short message."}]
        chunks = chunk_messages(messages)
        assert len(chunks) == 1

    def test_long_conversation(self):
        messages = [
            {"role": "user", "content": f"Message {i}. This has details about topic {i}."}
            for i in range(20)
        ]
        chunks = chunk_messages(messages, max_sentences=4, overlap=1)
        assert len(chunks) > 1

    def test_preserves_role(self):
        messages = [
            {"role": "user", "content": "I prefer Python."},
            {"role": "assistant", "content": "Noted."},
        ]
        chunks = chunk_messages(messages)
        assert "user:" in chunks[0].lower()
