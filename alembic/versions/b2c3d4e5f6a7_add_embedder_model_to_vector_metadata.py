"""Add embedder_model to vector_metadata

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-25 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add embedder_model column (default "transformer" for existing rows)
    op.add_column(
        "vector_metadata",
        sa.Column("embedder_model", sa.String(), nullable=False, server_default="transformer"),
    )
    op.create_index("ix_vector_metadata_embedder_model", "vector_metadata", ["embedder_model"])

    # 2. Drop the old single-column unique constraint on vector_id
    op.drop_constraint("vector_metadata_pkey", "vector_metadata", type_="unique")

    # 3. Add composite unique constraint
    op.create_unique_constraint(
        "uq_vector_model", "vector_metadata", ["vector_id", "embedder_model"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_vector_model", "vector_metadata", type_="unique")
    op.create_unique_constraint("vector_metadata_pkey", "vector_metadata", ["vector_id"])
    op.drop_index("ix_vector_metadata_embedder_model", table_name="vector_metadata")
    op.drop_column("vector_metadata", "embedder_model")
