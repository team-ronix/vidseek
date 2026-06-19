"""Add vector metadata table

Revision ID: 3f7f1d2e9b10
Revises: 1c928b4ae412
Create Date: 2026-06-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3f7f1d2e9b10'
down_revision: Union[str, Sequence[str], None] = '1c928b4ae412'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'vector_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('vector_id', sa.Integer(), nullable=False),
        sa.Column('external_id', sa.String(), nullable=True),
        sa.Column('entry_type', sa.String(), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('video_path', sa.String(), nullable=True),
        sa.Column('start_time', sa.Float(), nullable=True),
        sa.Column('end_time', sa.Float(), nullable=True),
        sa.Column('payload_json', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('vector_id')
    )
    op.create_index(op.f('ix_vector_metadata_id'), 'vector_metadata', ['id'], unique=False)
    op.create_index(op.f('ix_vector_metadata_vector_id'), 'vector_metadata', ['vector_id'], unique=True)
    op.create_index(op.f('ix_vector_metadata_entry_type'), 'vector_metadata', ['entry_type'], unique=False)
    op.create_index(op.f('ix_vector_metadata_video_path'), 'vector_metadata', ['video_path'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_vector_metadata_video_path'), table_name='vector_metadata')
    op.drop_index(op.f('ix_vector_metadata_entry_type'), table_name='vector_metadata')
    op.drop_index(op.f('ix_vector_metadata_vector_id'), table_name='vector_metadata')
    op.drop_index(op.f('ix_vector_metadata_id'), table_name='vector_metadata')
    op.drop_table('vector_metadata')
