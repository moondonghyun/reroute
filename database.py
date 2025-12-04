# database.py
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# DB 접속 정보 (.env)
DB_URL = os.getenv("DATABASE_URL")  # 예: mysql+pymysql://user:pass@host:3306/dbname

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# User 모델 정의 (기존 테이블과 매핑)
class User(Base):
    __tablename__ = "User"  # 실제 테이블 이름 확인 필요

    id = Column("user_id", Integer, primary_key=True, index=True)  # 내부 user_id
    sub = Column(String(255), unique=True, index=True)  # Cognito sub


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()