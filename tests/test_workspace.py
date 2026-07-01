import pytest

from opendubbing.core.workspace import Workspace


class TestWorkspace:
    def test_creates_subdirectories(self, tmp_path):
        workspace = Workspace(tmp_path / "ws")
        for name in Workspace.SUBDIRECTORIES:
            assert (workspace.root / name).exists()

    def test_path_for(self, tmp_path):
        workspace = Workspace(tmp_path / "ws")
        path = workspace.path_for("timeline", "test.jsonl")
        assert path == workspace.root / "timeline" / "test.jsonl"

    def test_path_for_unknown_step(self, tmp_path):
        workspace = Workspace(tmp_path / "ws")
        with pytest.raises(ValueError):
            workspace.path_for("unknown", "file.txt")

    def test_exists(self, tmp_path):
        workspace = Workspace(tmp_path / "ws")
        assert not workspace.exists("timeline", "test.jsonl")
        path = workspace.path_for("timeline", "test.jsonl")
        path.write_text("{}")
        assert workspace.exists("timeline", "test.jsonl")

    def test_timeline_path_property(self, tmp_path):
        workspace = Workspace(tmp_path / "ws")
        assert workspace.timeline_path == workspace.root / "timeline" / "timeline.jsonl"
