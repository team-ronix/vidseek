"""Add transcript_segments table

Revision ID: a1b2c3d4e5f6
Revises: 262e0fd8c58c
Create Date: 2026-06-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '262e0fd8c58c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'transcript_segments',
        sa.Column('id',       sa.Integer(), primary_key=True, index=True),
        sa.Column('video_id', sa.Integer(), sa.ForeignKey('videos.id'), nullable=False, index=True),
        sa.Column('text',     sa.Text(),    nullable=False),
        sa.Column('start',    sa.Float(),   nullable=False),
        sa.Column('end',      sa.Float(),   nullable=False),
        sa.Column('title',    sa.String(),  nullable=False),
    )


def downgrade() -> None:
    op.drop_table('transcript_segments')
