from logging.config import fileConfig

import os
from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import create_engine

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base and all models so Alembic can detect them for autogenerate
from Storage.SQL.Models.Base import Base
from Storage.SQL.Models.Video import Video                  # noqa: F401
from Storage.SQL.Models.Object import Object                # noqa: F401
from Storage.SQL.Models.ObjectVideo import ObjectVideo      # noqa: F401
from Storage.SQL.Models.VRDSubject import VRDSubject        # noqa: F401
from Storage.SQL.Models.VRDPredicate import VRDPredicate    # noqa: F401
from Storage.SQL.Models.VRDObject import VRDObject          # noqa: F401
from Storage.SQL.Models.VRDVideo import VRDVideo            # noqa: F401

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    # prefer DATABASE_URL from environment (e.g. .env) if present
    load_dotenv()
    url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        connectable = create_engine(db_url, poolclass=pool.NullPool)
    else:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
