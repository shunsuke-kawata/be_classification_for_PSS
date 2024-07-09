from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config import MYSQL_ROOT_PASSWORD,DATABASE_PORT,MYSQL_DATABASE

# ここでsys.pathやconfigの読み込みは割愛します

# SQLAlchemyの接続文字列
CONNECT_STRING = f"mysql://root:{MYSQL_ROOT_PASSWORD}@db-pss-app:{DATABASE_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

# SQLAlchemyのベースクラス
Base = declarative_base()

# Userテーブルの定義
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    authority = Column(Boolean, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    projects = relationship("Project", back_populates="owner")

# Projectテーブルの定義
class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    description = Column(Text)
    images_folder_path = Column(String(255), nullable=False)
    object_images_folder_path=Column(String(255), nullable=False)
    owner_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="projects")
    memberships = relationship("ProjectMembership", back_populates="project")
    images = relationship("Image", back_populates="project")
    object_images = relationship("ObjectImage", back_populates="project")
    object_groups = relationship("ObjectGroup", back_populates="project")

# ProjectMembershipテーブルの定義
class ProjectMembership(Base):
    __tablename__ = "project_memberships"

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    project = relationship("Project", back_populates="memberships")

# Imageテーブルの定義
class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="images")

# ObjectImageテーブルの定義
class ObjectImage(Base):
    __tablename__ = "object_images"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(255), nullable=False)
    original_image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    original_image = relationship("Image")
    project = relationship("Project", back_populates="object_images")

# ObjectGroupテーブルの定義
class ObjectGroup(Base):
    __tablename__ = "object_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    is_noise_group = Column(Boolean, nullable=False)
    is_trash_group = Column(Boolean, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    user = relationship("User")
    project = relationship("Project", back_populates="object_groups")
    object_images = relationship("ObjectImage", secondary="object_group_relationships", back_populates="object_groups")

# ObjectGroupRelationshipテーブルの定義
class ObjectGroupRelationship(Base):
    __tablename__ = "object_group_relationships"

    object_image_id = Column(Integer, ForeignKey('object_images.id', ondelete='CASCADE'), primary_key=True)
    object_group_id = Column(Integer, ForeignKey('object_groups.id', ondelete='CASCADE'), primary_key=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

# Logテーブルの定義
class Log(Base):
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    to_group_id = Column(Integer, ForeignKey('object_groups.id', ondelete='CASCADE'), nullable=False)
    from_group_id = Column(Integer, ForeignKey('object_groups.id', ondelete='CASCADE'), nullable=False)
    object_image_id = Column(Integer, ForeignKey('object_images.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

# エンジンを作成してテーブルを作成
migration_engine = create_engine(CONNECT_STRING)
if __name__=='__main__':
    Base.metadata.create_all(bind=migration_engine)