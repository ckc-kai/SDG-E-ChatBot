"""Figure object storage adapters.

Chunks store provider-independent object keys. The configured adapter resolves
those keys to a local path or S3 URI and owns writes/deletes for one document
prefix.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Protocol

from retrieval.utils import load_config


class ObjectStorage(Protocol):
    def put_png(self, key: str, image) -> str: ...

    def clear_prefix(self, prefix: str) -> None: ...

    def uri(self, key: str) -> str: ...


def _safe_key(key: str) -> str:
    path = Path(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe object key: {key!r}")
    return "/".join(path.parts)


class FileSystemObjectStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put_png(self, key: str, image) -> str:
        safe_key = _safe_key(key)
        target = self.root / safe_key
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
        return safe_key

    def clear_prefix(self, prefix: str) -> None:
        safe_prefix = _safe_key(prefix)
        target = (self.root / safe_prefix).resolve()
        root = self.root.resolve()
        if target == root or root not in target.parents:
            raise ValueError(f"Refusing to clear unsafe storage prefix: {prefix!r}")
        if target.exists():
            shutil.rmtree(target)

    def uri(self, key: str) -> str:
        return str(self.root / _safe_key(key))


class S3ObjectStorage:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        client=None,
    ):
        if not bucket:
            raise ValueError("S3 object storage requires a bucket")
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3 storage requires boto3; install the project with its AWS "
                    "dependencies"
                ) from exc
            client = boto3.client("s3", region_name=region)
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _full_key(self, key: str) -> str:
        safe_key = _safe_key(key)
        return f"{self.prefix}/{safe_key}" if self.prefix else safe_key

    def put_png(self, key: str, image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=buffer.getvalue(),
            ContentType="image/png",
        )
        return _safe_key(key)

    def clear_prefix(self, prefix: str) -> None:
        full_prefix = self._full_key(prefix).rstrip("/") + "/"
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects, "Quiet": True},
                )

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full_key(key)}"


def get_object_storage() -> ObjectStorage:
    config = load_config().get("object_storage", {})
    provider = config.get("provider", "filesystem")
    if provider == "filesystem":
        root = config.get("filesystem", {}).get(
            "root", "resources/wmp/figures"
        )
        return FileSystemObjectStorage(root)
    if provider == "s3":
        s3 = config.get("s3", {})
        return S3ObjectStorage(
            bucket=s3.get("bucket", ""),
            prefix=s3.get("prefix", ""),
            region=s3.get("region"),
        )
    raise ValueError(
        f"Unknown object storage provider {provider!r}; expected 'filesystem' or 's3'"
    )
