"""Durable, scoped keys for explicitly enabled gateway runtime authentication.

An approved key must survive SDK and process restarts. A corrupt/missing stored
key is an error, never an instruction to silently replace an approved identity.
The local descriptor remembers the storage backend so an unavailable keyring
cannot cause a fallback to a new file key on a later run.

:func:`runtime_key_available` answers, without loading or creating anything,
whether :func:`load_runtime_key` has what it needs in this process. Both apply
the same helpers below, so the two cannot drift apart.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Iterator

from ..dpop import DPoPKey
from ..errors import GovernanceConfigurationError

_MAX_KEY_BYTES = 16384
_STORE_LOCK = threading.RLock()
_KEYRING_SERVICE = "deepintshield.runtime.v1"


def _error() -> GovernanceConfigurationError:
    return GovernanceConfigurationError(
        framework="gateway-broker",
        reason="The runtime signing key could not be loaded safely; restore the existing key or explicitly enroll a replacement.",
        code="agent_runtime_key_unavailable",
    )


def _secure_keyring():
    """Only native protected keyrings; never a plaintext third-party backend."""
    try:
        import keyring

        backend = keyring.get_keyring()
        candidates = getattr(backend, "backends", None) or [backend]
        for candidate in candidates:
            module = type(candidate).__module__
            if module in {
                "keyring.backends.macOS", "keyring.backends.Windows",
                "keyring.backends.SecretService", "keyring.backends.libsecret",
                "keyring.backends.kwallet",
            } and candidate.priority > 0:
                return candidate
    except Exception:
        pass
    return None


def _require_cryptography() -> None:
    """Raise ``credential_dependency_missing`` without the ``runtime`` extra."""
    try:
        # Import before accessing storage so a missing optional dependency has
        # useful guidance instead of looking like a corrupt enrolled key.
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401
    except ImportError:
        raise GovernanceConfigurationError(
            framework="gateway-broker",
            reason="Install deepintshield[runtime] to use gateway runtime authentication.",
            code="credential_dependency_missing",
        ) from None


def _file_storage_supported() -> bool:
    """Whether a key may live in the 0600 descriptor file instead of a keyring.

    Windows has no owner-only file mode to rely on, so a key is kept there
    only in the native credential store.
    """
    return os.name != "nt"


def _store_directory() -> Path:
    """The durable descriptor directory for keys the SDK manages itself."""
    root = os.environ.get("DEEPINTSHIELD_RUNTIME_KEY_DIR", "").strip()
    return Path(root).expanduser() if root else Path.home() / ".deepintshield" / "runtime-keys"


def _private_bytes(key: DPoPKey) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key._key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _parse_key(raw: bytes) -> DPoPKey:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    if not raw or len(raw) > _MAX_KEY_BYTES:
        raise _error()
    try:
        return DPoPKey(private_key=load_pem_private_key(raw, password=None))
    except Exception:
        raise _error() from None


def _check_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_KEY_BYTES:
        raise _error()
    if os.name != "nt":
        # A root-owned 0400 mounted secret is useful in a root container. For
        # non-root processes the secret must be owned by their effective user.
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise _error()


def _check_directory(info: os.stat_result) -> None:
    # The descriptor directory is a real directory (never a symlink to one)
    # that only its owner can enter.
    if not stat.S_ISDIR(info.st_mode):
        raise _error()
    if os.name != "nt" and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
        raise _error()


def _read_file(path: Path, *, mounted: bool = False) -> bytes:
    # Kubernetes secret projections use symlinks. Only an explicitly selected
    # mounted path resolves those; the SDK's managed store forbids symlinks.
    if mounted:
        path = path.resolve(strict=True)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _check_file(os.fstat(fd))
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_KEY_BYTES + 1)
        if len(raw) > _MAX_KEY_BYTES:
            raise _error()
        return raw
    finally:
        os.close(fd)


@contextmanager
def _scope_lock(directory: Path, scope: str) -> Iterator[None]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_directory(directory.lstat())
    fd = os.open(
        directory / (scope + ".lock"),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _check_file(os.fstat(fd))
        if os.name == "nt":  # pragma: no cover - Windows CI
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _write_descriptor(path: Path, value: dict) -> None:
    # The scope lock serializes creators across threads AND processes. A
    # temporary 0600 inode plus rename makes a crash unable to expose a partial
    # descriptor as a valid key. No existing descriptor is ever overwritten.
    fd, name = tempfile.mkstemp(prefix=".runtime-key-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, separators=(",", ":")).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def _mounted_key() -> DPoPKey | None:
    """The explicitly mounted, read-only key, or None when none is selected."""
    mounted = os.environ.get("DEEPINTSHIELD_RUNTIME_KEY_FILE", "").strip()
    if not mounted:
        return None
    return _parse_key(_read_file(Path(mounted).expanduser(), mounted=True))


def _directory_usable(directory: Path) -> bool:
    """Whether :func:`_scope_lock` could work in ``directory``, creating nothing."""
    try:
        info = directory.lstat()
    except FileNotFoundError:
        # ``mkdir(parents=True)`` builds every missing level below the nearest
        # one that exists, which therefore has to be a directory this process
        # may write to.
        for ancestor in directory.parents:
            try:
                existing = ancestor.stat()
            except FileNotFoundError:
                continue
            return stat.S_ISDIR(existing.st_mode) and os.access(ancestor, os.W_OK | os.X_OK)
        return False
    _check_directory(info)
    return os.access(directory, os.W_OK | os.X_OK)


def runtime_key_available() -> bool:
    """Whether :func:`load_runtime_key` has what it needs in this process.

    A local answer for a caller that has to decide before any key exists. It
    sends no request, generates no key and creates or writes nothing: the
    optional dependency must import, an explicitly mounted key must be
    readable, and otherwise the descriptor directory must either exist the way
    the loader requires or be creatable. Every rule is the loader's own.

    What only loading can show is not judged here - a scope's own descriptor,
    the keyring's content, the lock file - so a True is necessary, not
    sufficient, and :func:`load_runtime_key` remains the authority.
    """
    try:
        _require_cryptography()
        if _mounted_key() is not None:
            return True
        if not _file_storage_supported() and _secure_keyring() is None:
            return False
        return _directory_usable(_store_directory())
    except Exception:
        return False


def load_runtime_key(
    gateway_origin: str, tenant_id: str, workspace_id: str, agent_subject: str,
) -> DPoPKey:
    """Load one key for this exact gateway/workspace/agent security boundary.

    DEEPINTSHIELD_RUNTIME_KEY_FILE is an explicit, read-only mounted PEM key.
    Otherwise use the native keyring, with a 0600 local file fallback on POSIX.
    DEEPINTSHIELD_RUNTIME_KEY_DIR selects the durable descriptor directory.
    """
    _require_cryptography()
    try:
        mounted = _mounted_key()
        if mounted is not None:
            return mounted
        scope = hashlib.sha256(json.dumps(
            [gateway_origin, tenant_id, workspace_id, agent_subject],
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")).hexdigest()
        directory = _store_directory()
        with _STORE_LOCK, _scope_lock(directory, scope):
            path = directory / (scope + ".json")
            exists = True
            try:
                descriptor = json.loads(_read_file(path))
            except FileNotFoundError:
                exists = False
                descriptor = None
            if exists:
                if not isinstance(descriptor, dict) or descriptor.get("version") != 1:
                    raise _error()
                backend = descriptor.get("storage")
                if backend == "keyring":
                    ring = _secure_keyring()
                    if ring is None:
                        raise _error()
                    raw = ring.get_password(_KEYRING_SERVICE, scope)
                    if not isinstance(raw, str) or not raw:
                        raise _error()
                elif backend == "file" and _file_storage_supported():
                    raw = descriptor.get("private_key_pem")
                    if not isinstance(raw, str):
                        raise _error()
                else:
                    raise _error()
                key = _parse_key(raw.encode("ascii"))
                if descriptor.get("thumbprint") != key.thumbprint:
                    raise _error()
                return key
            key = DPoPKey()
            raw = _private_bytes(key).decode("ascii")
            descriptor = {"version": 1, "thumbprint": key.thumbprint}
            ring = _secure_keyring()
            stored = False
            if ring is not None:
                try:
                    # An orphaned key can remain if a process stopped before
                    # its descriptor was committed. Reuse it, never overwrite.
                    previous = ring.get_password(_KEYRING_SERVICE, scope)
                    if previous:
                        key = _parse_key(previous.encode("ascii"))
                        descriptor["thumbprint"] = key.thumbprint
                    else:
                        ring.set_password(_KEYRING_SERVICE, scope, raw)
                    stored = True
                    descriptor["storage"] = "keyring"
                except GovernanceConfigurationError:
                    raise
                except Exception:
                    # Fallback is allowed only before an identity is enrolled.
                    pass
            if not stored:
                if not _file_storage_supported():
                    raise _error()
                descriptor.update(storage="file", private_key_pem=raw)
            _write_descriptor(path, descriptor)
            return key
    except GovernanceConfigurationError:
        raise
    except Exception:
        # Never expose PEM contents, keyring exception strings, or paths which
        # might themselves contain deployment credentials.
        raise _error() from None
