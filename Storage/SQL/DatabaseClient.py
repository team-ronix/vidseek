from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import dotenv
import os
dotenv.load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from Storage.SQL.Models.Base import Base
    import Storage.SQL.Models.Video    
    import Storage.SQL.Models.Object  
    import Storage.SQL.Models.ObjectVideo
    import Storage.SQL.Models.VRDVideo
    import Storage.SQL.Models.VRDObject 
    import Storage.SQL.Models.VRDSubject
    import Storage.SQL.Models.VRDPredicate
    import Storage.SQL.Models.OCRWords
    import Storage.SQL.Models.VectorMetadata
    Base.metadata.create_all(bind=engine)