"""merge heads

Revision ID: c64c8669ea17
Revises: 32e20aef926f, b2c3d4e5f6a7
Create Date: 2026-06-25 02:27:11.268231

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c64c8669ea17'
down_revision: Union[str, Sequence[str], None] = ('32e20aef926f', 'b2c3d4e5f6a7')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
