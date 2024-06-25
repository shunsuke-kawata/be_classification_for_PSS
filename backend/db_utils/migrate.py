from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey,Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import MYSQL_ROOT_PASSWORD,DATABASE_PORT,MYSQL_DATABASE

CONNECT_STRING = f"mysql://root:{MYSQL_ROOT_PASSWORD}@db-pss-app:{DATABASE_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False,unique=True)
    password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    authority = Column(Boolean, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False,unique=True)
    password = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    owner = relationship("User", back_populates="projects")

class ProjectMembership(Base):
    __tablename__ = "project_memberships"

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class ObjectImage(Base):
    __tablename__ = "object_images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class ObjectGroup(Base):
    __tablename__ = "object_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    to_group_id = Column(Integer, ForeignKey('object_groups.id', ondelete='CASCADE'), nullable=False)
    from_group_id = Column(Integer, ForeignKey('object_groups.id', ondelete='CASCADE'), nullable=False)
    object_image_id = Column(Integer, ForeignKey('object_images.id', ondelete='CASCADE'), nullable=False)
    method = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
#マイグレーションの実行
engine = create_engine(CONNECT_STRING)
Base.metadata.create_all(bind=engine)
