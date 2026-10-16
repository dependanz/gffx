"""Storage accounting for one exclusively owned benchmark directory.

Only the declared SQLite scratch namespace may disappear normally. Entries are
classified from one lstat-equivalent stat attempt, never exists/is_file + stat.
Permission failures, links, unexplained loss and quota excess propagate.
"""
import os
from pathlib import Path
import re
import stat

COMBINED_LIMIT = 2 * 1024**3
_INDEX_NAME = re.compile(r"exact-index-[a-z0-9_]{8}\Z")
_SQLITE_FILES = frozenset(("held.sqlite", "held.sqlite-journal"))


def _scandir(path):
    return os.scandir(path)


def _entry_stat(entry):
    return entry.stat(follow_symlinks=False)


class StorageObserver:
    def __init__(self, root, *, limit=COMBINED_LIMIT, required=()):
        # abspath is lexical; do not resolve symlinks or make extra stat attempts.
        self.root = Path(os.path.abspath(root))
        self.limit = limit
        self.permanent = {self.root / name for name in required}
        if any(path.parent != self.root for path in self.permanent):
            raise ValueError("required permanent files must be immediate root entries")
        self.allowed_disappearances = 0
        self.samples = 0

    def _temporary(self, path):
        parts = path.relative_to(self.root).parts
        if parts == ("scratch",):
            return True
        if len(parts) in (2, 3) and parts[0] == "scratch" and _INDEX_NAME.fullmatch(parts[1]):
            return len(parts) == 2 or parts[2] in _SQLITE_FILES
        return False

    def _walk(self, directory, seen):
        total = 0
        try:
            with _scandir(directory) as entries:
                for entry in entries:
                    path = directory / entry.name
                    temporary = self._temporary(path)
                    try:
                        info = _entry_stat(entry)  # Exactly one attempt for this entry.
                    except FileNotFoundError:
                        if not temporary:
                            raise
                        self.allowed_disappearances += 1
                        continue
                    if stat.S_ISLNK(info.st_mode) or (
                        getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    ):
                        raise OSError("storage observer refuses links/reparse points: " + str(path))
                    if not temporary:
                        seen.add(path)
                    if stat.S_ISDIR(info.st_mode):
                        total += self._walk(path, seen)
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                    else:
                        raise OSError("storage observer refuses special entries: " + str(path))
        except FileNotFoundError as error:
            # Catch ONLY disappearance of the directory being opened/enumerated.
            # An unrelated missing child must not be swallowed by an ancestor.
            missing = getattr(error, "filename", None)
            if (not self._temporary(directory)
                    or missing is None
                    or Path(os.path.abspath(missing)) != directory):
                raise
            self.allowed_disappearances += 1
        return total

    def bytes_used(self):
        seen = set()
        total = self._walk(self.root, seen)
        missing = self.permanent - seen
        if missing:
            raise FileNotFoundError("permanent benchmark entry disappeared: " +
                                    ", ".join(str(p) for p in sorted(missing)))
        self.permanent.update(seen)
        self.samples += 1
        if total > self.limit:
            raise OSError("2GiB combined output/scratch refusal: " + str(total))
        return total


_OBSERVERS = {}


def bytes_used(root):
    key = os.path.abspath(root)
    observer = _OBSERVERS.get(key)
    if observer is None:
        observer = _OBSERVERS[key] = StorageObserver(key)
    return observer.bytes_used()
