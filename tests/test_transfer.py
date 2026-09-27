from __future__ import annotations

import contextlib
import hashlib
import shutil
import unittest
import uuid
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from synpo.importer import scan_batch
from synpo.models import Calibration
from synpo.project import create_project_manifest, load_project, save_project
from synpo.transfer import (
    TransferCacheError,
    TransferError,
    TransferValidationError,
    create_transfer_archive,
    import_transfer_archive,
    inspect_transfer_archive,
)


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def make_project(root: Path) -> tuple[dict[str, object], Path, Path]:
    source = root / "source"
    source.mkdir()
    base = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    for offset, channel in enumerate(("ChanA", "ChanB"), start=1):
        tifffile.imwrite(
            source / f"batch_group_specimen_{channel}_registered.tif",
            base + offset,
            metadata={"axes": "ZYX"},
            photometric="minisblack",
        )
    manifest = create_project_manifest(
        scan_batch(source),
        output_directory=root / "old-output",
        channel_roles={"ChanA": "protein_clusters", "ChanB": "dendrite_spines"},
        calibration=Calibration("transfer", 0.05, 0.5),
    )
    specimen = manifest["specimens"][0]
    specimen["review"].update(
        {
            "state": "reviewed",
            "comment": "Preserve this specimen comment",
            "history": [{"operation": "trim", "object_id": 4}],
            "object_status": {"dendrite": {"1": "accepted"}, "spine": {}},
        }
    )
    specimen["distribution_review"] = {
        "spines": {"4": {"distribution_included": True, "endpoint_hint_zyx": [1, 2, 3]}},
        "updated_at": 123.0,
    }
    specimen["analysis"] = {
        "excluded": True,
        "exclusion_reason": "edge artifact",
        "rois_xy": [{"x": 1, "y": 2, "width": 4, "height": 5}],
    }
    manifest["preprocessing"]["settings_by_specimen"]["0"] = {
        "ChanA": {"background_percentile": 17.0}
    }
    project = save_project(root / "example.synpo.json", manifest)
    return manifest, project, source


class TransferTests(unittest.TestCase):
    def test_settings_only_with_raw_preserves_setup_and_comments_but_resets_state(self) -> None:
        with workspace_directory() as root:
            manifest, project, _source = make_project(root)
            archive = create_transfer_archive(
                manifest,
                project,
                root / "settings.synpo-transfer.zip",
                mode="settings_only",
                include_raw=True,
            )
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            info = inspect_transfer_archive(archive)
            self.assertEqual(info.mode, "settings_only")
            self.assertTrue(info.include_raw)

            result = import_transfer_archive(archive, root / "imports")
            loaded = load_project(result.project_path)
            specimen = loaded["specimens"][0]
            self.assertEqual(specimen["review"]["comment"], "Preserve this specimen comment")
            self.assertEqual(specimen["review"]["history"], [])
            self.assertEqual(
                specimen["review"]["object_status"], {"dendrite": {}, "spine": {}}
            )
            self.assertEqual(specimen["distribution_review"]["spines"], {})
            self.assertEqual(specimen["checkpoints"]["preprocessing"]["state"], "excluded")
            self.assertEqual(specimen["checkpoints"]["detection"]["state"], "not_started")
            self.assertEqual(specimen["analysis"]["exclusion_reason"], "edge artifact")
            self.assertEqual(len(specimen["analysis"]["rois_xy"]), 1)
            self.assertEqual(
                loaded["preprocessing"]["settings_by_specimen"]["0"]["ChanA"][
                    "background_percentile"
                ],
                17.0,
            )
            self.assertIsNone(loaded["cache"]["path"])
            for channel_data in specimen["channels"].values():
                raw_path = Path(channel_data["source_path"])
                self.assertTrue(raw_path.is_file())
                self.assertTrue(raw_path.is_relative_to(result.project_path.parent))
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), archive_digest)

    def test_full_transfer_restores_cache_and_recovers_renamed_external_tiffs(self) -> None:
        with workspace_directory() as root:
            manifest, project, source = make_project(root)
            cache_root = root / "old-output" / ".synpo-cache" / str(manifest["project_id"])
            (cache_root / "misc").mkdir(parents=True)
            (cache_root / "misc" / "state.bin").write_bytes(b"cached-state")
            manifest["cache"]["path"] = str(cache_root / "preprocessed.zarr")
            project = save_project(project, manifest)
            archive = create_transfer_archive(
                manifest, project, root / "full.synpo-transfer.zip", mode="full"
            )

            relocated = root / "renamed-raw"
            relocated.mkdir()
            original_paths = sorted(source.glob("*.tif"))
            for index, original in enumerate(original_paths, start=1):
                shutil.copy2(original, relocated / f"renamed-{index}.tiff")

            result = import_transfer_archive(
                archive,
                root / "imports",
                external_raw_directory=relocated,
            )
            loaded = load_project(result.project_path)
            self.assertEqual(result.renamed_source_matches, 2)
            self.assertEqual(
                loaded["specimens"][0]["review"]["history"],
                [{"operation": "trim", "object_id": 4}],
            )
            new_cache = Path(loaded["cache"]["path"]).parent
            self.assertTrue(new_cache.is_relative_to(result.project_path.parent))
            self.assertEqual((new_cache / "misc" / "state.bin").read_bytes(), b"cached-state")
            self.assertTrue(
                all(
                    Path(channel["source_path"]).name.startswith("renamed-")
                    for channel in loaded["specimens"][0]["channels"].values()
                )
            )

            second = import_transfer_archive(
                archive,
                root / "imports",
                external_raw_directory=relocated,
                conflict_policy="copy",
            )
            self.assertNotEqual(second.project_path.parent, result.project_path.parent)
            self.assertTrue(second.project_path.parent.name.endswith("-2"))

            marker = result.project_path.parent / "keep-on-failure.txt"
            marker.write_text("old project", encoding="utf-8")
            empty_raw = root / "empty-raw"
            empty_raw.mkdir()
            with self.assertRaises(TransferError):
                import_transfer_archive(
                    archive,
                    root / "imports",
                    external_raw_directory=empty_raw,
                    conflict_policy="replace",
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "old project")

            replaced = import_transfer_archive(
                archive,
                root / "imports",
                external_raw_directory=relocated,
                conflict_policy="replace",
            )
            self.assertEqual(replaced.project_path.parent, result.project_path.parent)
            self.assertFalse(marker.exists())

    def test_corrupt_full_cache_can_only_be_recovered_as_settings_only(self) -> None:
        with workspace_directory() as root:
            manifest, project, source = make_project(root)
            cache_root = root / "old-output" / ".synpo-cache" / str(manifest["project_id"])
            cache_root.mkdir(parents=True)
            (cache_root / "state.bin").write_bytes(b"original cache")
            manifest["cache"]["path"] = str(cache_root / "preprocessed.zarr")
            project = save_project(project, manifest)
            archive = create_transfer_archive(
                manifest, project, root / "full.synpo-transfer.zip", mode="full"
            )
            corrupt = root / "corrupt.synpo-transfer.zip"
            with zipfile.ZipFile(archive, "r") as source_zip, zipfile.ZipFile(
                corrupt, "w", compression=zipfile.ZIP_DEFLATED
            ) as target_zip:
                for name in source_zip.namelist():
                    payload = source_zip.read(name)
                    if name.startswith("cache/"):
                        payload = b"damaged cache"
                    target_zip.writestr(name, payload)

            with self.assertRaises(TransferCacheError):
                import_transfer_archive(
                    corrupt,
                    root / "failed-import",
                    external_raw_directory=source,
                )
            self.assertFalse((root / "failed-import" / "example").exists())

            recovered = import_transfer_archive(
                corrupt,
                root / "recovered-import",
                external_raw_directory=source,
                recover_as_settings_only=True,
            )
            loaded = load_project(recovered.project_path)
            self.assertTrue(recovered.recovered_as_settings_only)
            self.assertIsNone(loaded["cache"]["path"])
            self.assertEqual(loaded["specimens"][0]["review"]["history"], [])
            self.assertEqual(
                loaded["specimens"][0]["review"]["comment"],
                "Preserve this specimen comment",
            )

    def test_unsafe_archive_member_is_rejected(self) -> None:
        with workspace_directory() as root:
            archive = root / "unsafe.synpo-transfer.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", b"unsafe")
                output.writestr("synpo-transfer.json", b"{}")
            with self.assertRaises(TransferValidationError):
                inspect_transfer_archive(archive)


if __name__ == "__main__":
    unittest.main()
