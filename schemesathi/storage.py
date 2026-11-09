from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "schemesathi.sqlite3"


class Store:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY,
                    profile_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS family_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES profiles(user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id INTEGER PRIMARY KEY,
                    state TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS document_status (
                    user_id INTEGER NOT NULL,
                    scheme_slug TEXT NOT NULL,
                    document TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY(user_id, scheme_slug, document)
                );
                CREATE TABLE IF NOT EXISTS recommendation_pages (
                    user_id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    result_slugs_json TEXT NOT NULL DEFAULT '[]'
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(recommendation_pages)")
            }
            if "result_slugs_json" not in columns:
                connection.execute(
                    "ALTER TABLE recommendation_pages "
                    "ADD COLUMN result_slugs_json TEXT NOT NULL DEFAULT '[]'"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def get_profile(self, user_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        profile = json.loads(row["profile_json"]) if row else {}
        if "caste" in profile:
            profile.pop("caste", None)
            self.replace_profile(user_id, profile)
        return profile

    def update_profile(self, user_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        profile.update(updates)
        self.replace_profile(user_id, profile)
        return profile

    def replace_profile(self, user_id: int, profile: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO profiles(user_id, profile_json) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json",
                (user_id, json.dumps(profile, ensure_ascii=False)),
            )

    def add_family_member(self, user_id: int, name: str, profile: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO profiles(user_id, profile_json) VALUES(?, '{}') "
                "ON CONFLICT(user_id) DO NOTHING",
                (user_id,),
            )
            connection.execute(
                "INSERT INTO family_members(user_id, name, profile_json) VALUES(?, ?, ?)",
                (user_id, name[:80], json.dumps(profile, ensure_ascii=False)),
            )

    def get_family(self, user_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, profile_json FROM family_members WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()
        return [{"id": row["id"], "name": row["name"], **json.loads(row["profile_json"])} for row in rows]

    def update_family_member(
        self, user_id: int, member_id: int, name: str, profile: dict[str, Any]
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE family_members SET name=?, profile_json=? WHERE id=? AND user_id=?",
                (name[:80], json.dumps(profile, ensure_ascii=False), member_id, user_id),
            )
        return cursor.rowcount == 1

    def set_conversation(self, user_id: int, state: str, data: dict[str, Any] | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(user_id, state, data_json) VALUES(?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, data_json=excluded.data_json",
                (user_id, state, json.dumps(data or {}, ensure_ascii=False)),
            )

    def get_conversation(self, user_id: int) -> tuple[str | None, dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, data_json FROM conversations WHERE user_id=?", (user_id,)
            ).fetchone()
        return (row["state"], json.loads(row["data_json"])) if row else (None, {})

    def clear_conversation(self, user_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))

    def set_document_status(self, user_id: int, slug: str, document: str, status: str) -> None:
        if status not in {"available", "missing", "unknown"}:
            raise ValueError("Document status must be available, missing, or unknown.")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO document_status(user_id, scheme_slug, document, status) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(user_id, scheme_slug, document) DO UPDATE SET status=excluded.status",
                (user_id, slug, document, status),
            )

    def get_document_statuses(self, user_id: int, slug: str) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document, status FROM document_status WHERE user_id=? AND scheme_slug=?",
                (user_id, slug),
            ).fetchall()
        return {row["document"]: row["status"] for row in rows}

    def set_recommendation_page(
        self,
        user_id: int,
        query: str,
        profile: dict[str, Any],
        mode: str,
        result_slugs: list[str] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO recommendation_pages"
                "(user_id, query, profile_json, mode, result_slugs_json) VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET query=excluded.query, "
                "profile_json=excluded.profile_json, mode=excluded.mode, "
                "result_slugs_json=excluded.result_slugs_json",
                (
                    user_id,
                    query,
                    json.dumps(profile, ensure_ascii=False),
                    mode,
                    json.dumps(result_slugs or [], ensure_ascii=False),
                ),
            )

    def get_recommendation_page(self, user_id: int) -> tuple[str, dict[str, Any], str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT query, profile_json, mode FROM recommendation_pages WHERE user_id=?", (user_id,)
            ).fetchone()
        return (row["query"], json.loads(row["profile_json"]), row["mode"]) if row else None

    def get_recommendation_slugs(self, user_id: int) -> list[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_slugs_json FROM recommendation_pages WHERE user_id=?", (user_id,)
            ).fetchone()
        return json.loads(row["result_slugs_json"]) if row else []

    def delete_profile(self, user_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM document_status WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM family_members WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM recommendation_pages WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM profiles WHERE user_id=?", (user_id,))
