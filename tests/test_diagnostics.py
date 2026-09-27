from __future__ import annotations

import json
import shutil
import unittest
import uuid
import zipfile
from pathlib import Path

from synpo.diagnostics import (
    DIAGNOSTIC_LEVEL_CORRECTION,
    DIAGNOSTIC_LEVEL_FULL,
    DiagnosticSession,
    diagnostic_protocol_path,
)


class DiagnosticSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / ".test-work" / uuid.uuid4().hex
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_full_session_writes_sanitized_evidence_and_summary(self) -> None:
        session = DiagnosticSession.create(
            self.root, level=DIAGNOSTIC_LEVEL_FULL
        )
        session.record(
            "test_event",
            source=Path.home() / "private" / "image.tif",
            comment="must not be collected",
            duration_seconds=1.25,
        )
        session.record("operation_started", kind="detection")
        session.record(
            "operation_progress", phase="Writing cache", current=2, total=10
        )
        session.sample_system()
        directory = session.finish(reason="test")

        self.assertTrue((directory / "environment.json").is_file())
        self.assertTrue((directory / "system-samples.csv").is_file())
        self.assertTrue((directory / "README.txt").is_file())
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        test_event = next(item for item in events if item["event"] == "test_event")
        self.assertTrue(
            str(test_event["details"]["source"]).startswith("%USERPROFILE%")
        )
        self.assertNotIn("comment", test_event["details"])
        summary = json.loads(
            (directory / "session-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["diagnostic_level"], DIAGNOSTIC_LEVEL_FULL)
        self.assertEqual(summary["system_sample_count"], 1)
        self.assertEqual(summary["event_counts"]["test_event"], 1)
        samples = (directory / "system-samples.csv").read_text(encoding="utf-8")
        self.assertIn("active_operation", samples)
        self.assertIn("detection", samples)
        self.assertIn("Writing cache", samples)

    def test_correction_level_filters_full_session_events(self) -> None:
        session = DiagnosticSession.create(
            self.root, level=DIAGNOSTIC_LEVEL_CORRECTION
        )
        session.record("full_event", scope="full")
        session.record("correction_event", scope="correction")
        directory = session.finish(reason="test")
        text = (directory / "events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("full_event", text)
        self.assertIn("correction_event", text)

    def test_package_contains_complete_session_folder(self) -> None:
        session = DiagnosticSession.create(
            self.root, level=DIAGNOSTIC_LEVEL_FULL
        )
        archive = session.package(reason="test")
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
        self.assertTrue(any(name.endswith("session-summary.json") for name in names))
        self.assertTrue(any(name.endswith("environment.json") for name in names))

    def test_standalone_protocol_is_bundled_pdf(self) -> None:
        protocol = diagnostic_protocol_path()
        self.assertTrue(protocol.is_file())
        contents = protocol.read_bytes()
        self.assertEqual(contents[:5], b"%PDF-")
        self.assertIn(b"/BaseFont /Helvetica", contents)
        self.assertIn(b"Synpo Performance Diagnostic Protocol", contents)
        self.assertNotIn(b"/FontFile", contents)
        self.assertEqual(contents.count(b"/Type /Page "), 3)
        self.assertGreater(protocol.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
