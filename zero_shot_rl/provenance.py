"""File hashes and no-clobber publication for generated data."""

from contextlib import contextmanager
import hashlib
import os


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def no_clobber(path):
    """Yield a temporary path to write; then hard-link it to `path`, which fails if
    `path` exists, even when another process creates it meanwhile."""
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    try:
        yield temporary
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot(files, repo, directory):
    """Hash and copy source files (paths under `repo`) into `directory`."""
    hashes = {}
    for file in files:
        relative = file.relative_to(repo)
        hashes[str(relative)] = sha256(file)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    return hashes
