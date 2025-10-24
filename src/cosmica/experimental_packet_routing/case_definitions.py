from typing import Annotated, Literal

from typing_extensions import Doc

BackupCaseType = Annotated[
    Literal[
        "no-backup",
        "backup-feeder-links",
        "backup-n-hops-links",
        "backup-n-hops-links-and-feeder-links",
        "dual-backup-n-hops-links-and-feeder-links",
    ],
    Doc(
        "The type of backup case. "
        "no-backup: No backup for link failure. "
        "backup-feeder-links: Backup of feeder links failure. "
        "backup-n-hops-links: Backup of links within n hops. "
        "backup-n-hops-links-and-feeder-links: Backup of links within n hops and feeder links."
        "dual-backup-n-hops-links-and-feeder-links: Dual backup of links within n hops and feeder links."  # noqa: COM812
    ),
]
