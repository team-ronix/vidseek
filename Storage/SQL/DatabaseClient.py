from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import dotenv
import os
dotenv.load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)