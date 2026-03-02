"""Cloud HPC backend abstractions for remote WarpX submission.

Provides a pluggable backend interface for submitting large WarpX scans
to cloud GPU instances. The LocalBackend runs inline on the current
machine (no cloud required). Cloud backends print SSH/API instructions
and return a job manifest that can be used for later retrieval.

Supported backends:
- ``"local"``   — run inline (default)
- ``"lambda"``  — Lambda Labs GPU cloud (stub, prints instructions)
- ``"aws"``     — AWS p4d instances (stub, prints instructions)

Usage::

    from helicon.cloud.backends import get_backend
    backend = get_backend("lambda")
    job = backend.submit(scan_config)
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CloudJob:
    """Serialisable cloud job manifest.

    Attributes
    ----------
    job_id : str
        Unique job identifier.
    backend : str
        Backend name.
    status : str
        One of ``"pending"``, ``"running"``, ``"done"``, ``"failed"``.
    config_path : str
        Path to the serialised scan config.
    output_dir : str
        Where results should be written.
    submitted_at : str
        ISO 8601 timestamp.
    metadata : dict
        Backend-specific metadata.
    """

    job_id: str
    backend: str
    status: str
    config_path: str
    output_dir: str
    submitted_at: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "status": self.status,
            "config_path": self.config_path,
            "output_dir": self.output_dir,
            "submitted_at": self.submitted_at,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> CloudJob:
        data = json.loads(Path(path).read_text())
        return cls(**data)


class CloudBackend(ABC):
    """Abstract cloud HPC backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""

    @abstractmethod
    def submit(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        *,
        n_gpus: int = 1,
        instance_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CloudJob:
        """Submit a WarpX scan job.

        Parameters
        ----------
        config_path : path-like
            Path to the YAML config file.
        output_dir : path-like
            Where output should be written.
        n_gpus : int
            Number of GPUs to request.
        instance_type : str, optional
            Cloud instance type (backend-specific).
        extra : dict, optional
            Backend-specific options.

        Returns
        -------
        CloudJob
        """

    def status(self, job_id: str) -> str:
        """Query job status. Returns ``"unknown"`` by default."""
        return "unknown"

    def retrieve(self, job: CloudJob) -> bool:
        """Retrieve results from a completed job. Returns success flag."""
        return False

    def _make_job(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        metadata: dict | None = None,
    ) -> CloudJob:
        return CloudJob(
            job_id=str(uuid.uuid4()),
            backend=self.name,
            status="pending",
            config_path=str(config_path),
            output_dir=str(output_dir),
            submitted_at=datetime.now(tz=UTC).isoformat(),
            metadata=metadata or {},
        )


class LocalBackend(CloudBackend):
    """Run scan inline on the local machine."""

    @property
    def name(self) -> str:
        return "local"

    def submit(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        *,
        n_gpus: int = 1,
        instance_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CloudJob:
        """Run the scan inline and return a completed job."""
        from helicon.config.parser import SimConfig
        from helicon.optimize.scan import ParameterRange, run_scan

        config = SimConfig.from_yaml(config_path)
        ranges_raw = (extra or {}).get("ranges", [])
        ranges = [ParameterRange(**r) for r in ranges_raw]
        method = (extra or {}).get("method", "lhc")
        dry_run = (extra or {}).get("dry_run", True)
        seed = (extra or {}).get("seed", 0)

        run_scan(
            config,
            ranges,
            output_base=str(output_dir),
            method=method,
            dry_run=dry_run,
            seed=seed,
        )

        job = self._make_job(config_path, output_dir)
        job.status = "done"
        return job

    def status(self, job_id: str) -> str:
        return "done"

    def retrieve(self, job: CloudJob) -> bool:
        return True


class LambdaLabsBackend(CloudBackend):
    """Lambda Labs GPU cloud backend (stub).

    Prints SSH/API instructions and returns a pending job manifest.
    Full implementation requires a Lambda Labs API key in the
    environment variable ``LAMBDA_API_KEY``.
    """

    @property
    def name(self) -> str:
        return "lambda"

    def submit(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        *,
        n_gpus: int = 1,
        instance_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CloudJob:
        instance = instance_type or "gpu_1x_a100_sxm4"
        job = self._make_job(
            config_path,
            output_dir,
            metadata={
                "instance_type": instance,
                "n_gpus": n_gpus,
                "provider": "lambda_labs",
            },
        )
        print(
            f"\n[Helicon Cloud] Lambda Labs submission\n"
            f"  Config:   {config_path}\n"
            f"  Output:   {output_dir}\n"
            f"  Instance: {instance}  GPUs: {n_gpus}\n"
            f"  Job ID:   {job.job_id}\n\n"
            f"  To submit manually:\n"
            f"    1. Launch a Lambda instance at https://cloud.lambda.ai\n"
            f"    2. scp {config_path} ubuntu@<ip>:~/helicon_job/config.yaml\n"
            f"    3. ssh ubuntu@<ip> 'pip install helicon && "
            f"helicon scan --config helicon_job/config.yaml "
            f"--output helicon_job/results --dry-run'\n"
            f"    4. scp -r ubuntu@<ip>:~/helicon_job/results/ {output_dir}\n"
            f"  API key: set LAMBDA_API_KEY in environment for automated submission.\n"
        )
        return job


class AWSBackend(CloudBackend):
    """AWS p4d instance backend (stub).

    Prints boto3/CLI instructions for p4d.24xlarge (8× A100) submission.
    Full implementation requires AWS credentials.
    """

    @property
    def name(self) -> str:
        return "aws"

    def submit(
        self,
        config_path: str | Path,
        output_dir: str | Path,
        *,
        n_gpus: int = 1,
        instance_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CloudJob:
        instance = instance_type or "p4d.24xlarge"
        job = self._make_job(
            config_path,
            output_dir,
            metadata={
                "instance_type": instance,
                "n_gpus": n_gpus,
                "provider": "aws",
            },
        )
        print(
            f"\n[Helicon Cloud] AWS submission\n"
            f"  Config:   {config_path}\n"
            f"  Instance: {instance}  GPUs: {n_gpus}\n"
            f"  Job ID:   {job.job_id}\n\n"
            f"  To submit via AWS CLI:\n"
            f"    aws ec2 run-instances --image-id ami-deeplearning "
            f"--instance-type {instance} "
            f"--user-data file://helicon_startup.sh\n"
            f"  Ensure AWS credentials are configured: aws configure\n"
        )
        return job


_BACKENDS: dict[str, type[CloudBackend]] = {
    "local": LocalBackend,
    "lambda": LambdaLabsBackend,
    "aws": AWSBackend,
}


def get_backend(name: str) -> CloudBackend:
    """Get a cloud backend instance by name.

    Parameters
    ----------
    name : ``"local"`` | ``"lambda"`` | ``"aws"``
        Backend identifier.

    Returns
    -------
    CloudBackend
    """
    try:
        return _BACKENDS[name]()
    except KeyError:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown backend {name!r}. Available: {available}") from None
