import json
import os

import pytest

from memory.long_term import LongTermMemory


@pytest.fixture
def prefs_path(tmp_path):
    return str(tmp_path / "user_prefs.json")


def test_load_defaults_when_file_missing(prefs_path):
    mem = LongTermMemory(prefs_path)
    prefs = mem.load()
    assert prefs == {"language": "chinese", "style": "academic"}


def test_save_and_load(prefs_path):
    mem = LongTermMemory(prefs_path)
    mem.save({"language": "english", "style": "casual", "domain": "nlp"})
    prefs = mem.load()
    assert prefs["language"] == "english"
    assert prefs["domain"] == "nlp"


def test_update_partial(prefs_path):
    mem = LongTermMemory(prefs_path)
    mem.update("domain", "cv")
    prefs = mem.load()
    assert prefs["domain"] == "cv"
    assert prefs["language"] == "chinese"


def test_get_existing_key(prefs_path):
    mem = LongTermMemory(prefs_path)
    mem.update("language", "english")
    assert mem.get("language") == "english"


def test_get_missing_key(prefs_path):
    mem = LongTermMemory(prefs_path)
    assert mem.get("nonexistent") is None


def test_get_all(prefs_path):
    mem = LongTermMemory(prefs_path)
    mem.update("domain", "nlp")
    all_prefs = mem.get_all()
    assert "language" in all_prefs
    assert all_prefs["domain"] == "nlp"