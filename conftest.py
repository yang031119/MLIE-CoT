"""Platform capabilities required by individual regression tests."""
import pytest


@pytest.fixture
def require_symlink(tmp_path):
    target = tmp_path / "symlink_capability_target"
    target.mkdir()
    link = tmp_path / "symlink_capability_link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symbolic-link creation permission")
        raise
    link.unlink()
    target.rmdir()
