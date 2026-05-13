import json
import os

_DEFAULT_PREFS = {"language": "chinese", "style": "academic"}


class LongTermMemory:
    def __init__(self, path: str = "user_prefs.json"):
        self._path = path

    def load(self) -> dict:
        if not os.path.exists(self._path):
            self.save(dict(_DEFAULT_PREFS))
            return dict(_DEFAULT_PREFS)
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, prefs: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)

    def update(self, key: str, value: str) -> None:
        prefs = self.load()
        prefs[key] = value
        self.save(prefs)

    def get(self, key: str) -> str | None:
        prefs = self.load()
        return prefs.get(key)

    def get_all(self) -> dict:
        return self.load()