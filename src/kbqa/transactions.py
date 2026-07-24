import json
from datetime import datetime, timezone
from pathlib import Path

from kbqa.models import StructuredContext


class TransactionFixtureError(ValueError):
    pass


class TransactionStore:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path

    def lookup(
        self,
        booking_id: str,
        *,
        as_of: datetime | None = None,
        expected_version: str | None = None,
    ) -> StructuredContext | None:
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        version = str(payload["version"])
        if expected_version is not None and version != expected_version:
            raise TransactionFixtureError(
                f"Expected fixture version {expected_version!r}, found {version!r}"
            )
        record = next((item for item in payload["bookings"] if item["id"] == booking_id), None)
        if record is None:
            return None

        fields: dict[str, str | bool | int | float | None] = {
            "status": record["status"],
            "property_id": record["property_id"],
            "check_in": record["check_in"],
            "free_cancellation_until": record["free_cancellation_until"],
        }
        if as_of is not None:
            clock = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            deadline = datetime.fromisoformat(record["free_cancellation_until"])
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            fields["free_cancellation_available"] = clock <= deadline
            fields["as_of"] = clock.isoformat()

        references = [f"booking:{booking_id}#{field}" for field in fields]
        return StructuredContext(
            record_type="booking",
            record_id=booking_id,
            fixture_version=version,
            fields=fields,
            references=references,
        )
