"""Schema migrations: fresh databases, ones from before versioning, new steps, failures and newer files."""

import json
import sqlite3

import pytest

from dainamo.server import db

TABLES = {"profiles", "datasets", "jobs", "settings", "sandbox_runs"}


def tables(c):
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_fresh_database_gets_every_step(tmp_path):
    c = sqlite3.connect(tmp_path / "fresh.db")
    assert db.migrate(c, backup_to=str(tmp_path / "fresh.bak")) == len(db.MIGRATIONS) == db.schema_version(c)
    assert not (tmp_path / "fresh.bak").exists()  # nothing to back up
    assert TABLES <= tables(c)
    assert db.migrate(c) == len(db.MIGRATIONS)  # nothing left to do


def test_database_from_before_versioning_keeps_its_rows(tmp_path):
    c = sqlite3.connect(tmp_path / "old.db")
    c.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT, family TEXT, data TEXT, created REAL, updated REAL)")
    c.execute("INSERT INTO profiles VALUES ('p1', 'mine', 'krea2', '{}', 1, 1)")
    c.commit()
    assert db.schema_version(c) == 0
    db.migrate(c, backup_to=str(tmp_path / "old.bak"))
    assert db.schema_version(c) == len(db.MIGRATIONS) and TABLES <= tables(c)
    assert c.execute("SELECT name FROM profiles").fetchall() == [("mine",)]
    assert sqlite3.connect(tmp_path / "old.bak").execute("SELECT name FROM profiles").fetchall() == [("mine",)]


def test_new_steps_run_once_in_order_after_a_backup(tmp_path):
    c = sqlite3.connect(tmp_path / "v1.db")
    db.migrate(c)
    c.execute("INSERT INTO jobs (id, name) VALUES ('j1', 'job')")
    c.commit()
    calls = []

    def fill(conn):
        calls.append(1)
        conn.execute("UPDATE jobs SET note = name || '!'")

    steps = db.MIGRATIONS + ["ALTER TABLE jobs ADD COLUMN note TEXT;\nCREATE INDEX jobs_note ON jobs(note);", fill]
    backup = tmp_path / "v1.bak"
    assert db.migrate(c, steps, backup_to=str(backup)) == len(steps)
    assert c.execute("SELECT note FROM jobs").fetchone() == ("job!",)
    db.migrate(c, steps, backup_to=str(backup))
    assert calls == [1]  # a step runs once
    old = sqlite3.connect(backup)  # the copy taken before the first new step
    assert db.schema_version(old) == len(db.MIGRATIONS)
    assert "note" not in [r[1] for r in old.execute("PRAGMA table_info(jobs)")]


def test_a_failed_step_changes_nothing(tmp_path):
    c = sqlite3.connect(tmp_path / "fail.db")
    db.migrate(c)
    steps = db.MIGRATIONS + ["CREATE TABLE half (x);\nINSERT INTO missing VALUES (1);"]
    with pytest.raises(sqlite3.OperationalError):
        db.migrate(c, steps)
    assert db.schema_version(c) == len(db.MIGRATIONS) and "half" not in tables(c)
    c.execute("INSERT INTO settings VALUES ('k', '1')")  # the connection still works normally
    c.commit()


def test_a_newer_database_is_refused(tmp_path):
    c = sqlite3.connect(tmp_path / "new.db")
    c.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="newer Dainamo Trainer"):
        db.migrate(c)


def _at_version_1(path, captioner=None):
    c = sqlite3.connect(path)
    db.migrate(c, db.MIGRATIONS[:1])
    c.execute("INSERT INTO datasets (id, name, path, managed, trigger, created) VALUES ('d1', 'cats', '/x', 0, '', 1)")
    if captioner is not None:
        c.execute("INSERT INTO settings VALUES ('captioner', ?)", (json.dumps(captioner),))
    c.commit()
    return c


def test_caption_styles_step_keeps_a_changed_caption_prompt(tmp_path):
    c = _at_version_1(tmp_path / "custom.db", {"endpoint": "http://x/v1", "prompt": "Tags please."})
    db.migrate(c)
    assert c.execute("SELECT id, caption_options FROM datasets").fetchall() == [("d1", None)]
    styles = json.loads(c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone()[0])
    assert styles["default"] == "mine" and styles["styles"][0] == {"id": "mine", "name": "My caption prompt",
                                                                   "prompt": "Tags please."}
    assert [st["id"] for st in styles["styles"][1:]] == [st["id"] for st in db.CAPTION_STYLES]
    assert sum(st["id"] == "moss" for st in styles["styles"]) == 1  # step 3 found it there already
    # the default prompt (or none saved) needs nothing kept: the built-in styles apply
    for name, captioner in (("same.db", {"prompt": db.DEFAULT_CAPTION_PROMPT}), ("none.db", None)):
        c = _at_version_1(tmp_path / name, captioner)
        db.migrate(c)
        assert c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone() is None


def test_a_new_default_caption_style_joins_saved_ones_once(tmp_path):
    c = sqlite3.connect(tmp_path / "styles.db")
    db.migrate(c, steps=db.MIGRATIONS[:2])
    saved = {"default": "tags", "styles": [{"id": "tags", "name": "My tags", "prompt": "Tags."}]}
    c.execute("INSERT INTO settings (key, value) VALUES ('caption_styles', ?)", (json.dumps(saved),))
    c.commit()
    db.migrate(c)
    styles = json.loads(c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone()[0])
    assert [st["id"] for st in styles["styles"]] == ["tags", "moss", "flux3"] and styles["default"] == "tags"
    assert db.MOSS_CAPTION_STYLE in db.CAPTION_STYLES  # and new installs have it among the defaults


def test_the_flux3_style_joins_saved_styles_and_stays_deleted(tmp_path):
    c = sqlite3.connect(tmp_path / "flux3.db")
    db.migrate(c, steps=db.MIGRATIONS[:4])
    saved = {"default": "mine", "styles": [{"id": "mine", "name": "Mine", "prompt": "Describe it."}]}
    c.execute("INSERT INTO settings (key, value) VALUES ('caption_styles', ?)", (json.dumps(saved),))
    c.commit()
    db.migrate(c)
    styles = json.loads(c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone()[0])
    assert styles["styles"] == [saved["styles"][0], db.FLUX3_CAPTION_STYLE] and styles["default"] == "mine"
    assert db.FLUX3_CAPTION_STYLE in db.CAPTION_STYLES and db.DEFAULT_SETTINGS["caption_styles"]["default"] == "detailed"
    c.execute("UPDATE settings SET value=? WHERE key='caption_styles'", (json.dumps(saved),))  # deleted later
    c.commit()
    db.migrate(c)
    assert json.loads(c.execute("SELECT value FROM settings WHERE key='caption_styles'").fetchone()[0]) == saved


def test_the_flux3_style_keeps_the_upsampler_rules_word_for_word():
    """Built from the prompt upsampler's text-to-image system prompt (the bundled pup_prompts.json): its own
    framing, rules about what an image shows where the upsampler's are about a request, and its caption format
    as written, less the one rule for inventing text a request needs."""
    import os
    import re

    path = os.path.join(os.path.dirname(db.__file__), "..", "assets", "flux3_pup_prompts.json")
    with open(path, encoding="utf-8") as f:
        t2i = json.load(f)["t2i"]
    style = db.FLUX3_CAPTION_STYLE
    framing, fidelity, caption_format, end = style["system"].split("\n\n")
    assert "Describe the attached image as ONE detailed, self-contained caption" in framing
    assert "Expand" not in style["system"] and "request" not in style["system"]
    assert framing.endswith(" Output only the caption, with no greetings, explanations, refusals, questions, Markdown, "
                            "headings, JSON, coordinates, ids or angle-bracket tokens, and no quotes around it.")
    assert framing.split(". ", 2)[-1] in t2i
    kept = ["Give a photograph a few ordinary, non-ideal specifics drawn from the scene's own objects instead of the "
            "stock-photo ideal.", "Time of day is lighting, not rendered text.", "turn moods into observable physical "
            "facts", "open with the medium convention (\"Full-color outdoor photograph, ...\"", "Do not invent names, "
            "brands, logos or celebrities", "drop appraisal and sales vocabulary"]
    assert all(k.lower() in t2i.lower() and k.lower() in fidelity.lower() for k in kept)
    request_only = " author concrete, domain-correct wording when the request needs text it does not supply, and"
    assert re.search(r"^Caption format: .*$", t2i, re.M)[0].replace(request_only, "") == caption_format
    assert end == "Output only the caption." and t2i.endswith(end)
    assert style["temperature"] == 0 and style["max_tokens"] == 1024  # the upsampler's greedy sampling, shorter
