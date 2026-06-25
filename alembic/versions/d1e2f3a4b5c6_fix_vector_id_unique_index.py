"""Fix vector_id unique index — drop unique, recreate non-unique

Revision ID: d1e2f3a4b5c6
Revises: c64c8669ea17
Create Date: 2026-06-25 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "c64c8669ea17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The original table creation made ix_vector_metadata_vector_id a UNIQUE index.
    # Now that the composite (vector_id, embedder_model) unique constraint exists,
    # this single-column unique index incorrectly blocks multiple embedder_model
    # rows sharing the same vector_id.
    op.drop_index("ix_vector_metadata_vector_id", table_name="vector_metadata")
    op.create_index("ix_vector_metadata_vector_id", "vector_metadata", ["vector_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_vector_metadata_vector_id", table_name="vector_metadata")
    op.create_index("ix_vector_metadata_vector_id", "vector_metadata", ["vector_id"], unique=True)
