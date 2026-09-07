"""Runtime compatibility patch for Ray/filelock on shared filesystems.

filelock 3.29 unlinks Unix lock files before calling flock(LOCK_UN). On the
container or network filesystems used by some GPU workers, that can raise
FileNotFoundError inside Ray initialization. Unlocking/closing before
unlinking preserves the same lock semantics.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import cast


try:
    from filelock import _unix

    def _safe_release(self) -> None:
        fd = cast("int", self._context.lock_file_fd)
        self._context.lock_file_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            Path(self.lock_file).unlink()
        except OSError:
            pass

    _unix.UnixFileLock._release = _safe_release
except Exception:
    pass

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass
