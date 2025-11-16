from typing import Tuple, Any
from db_utils.commons import execute_query


def select_all_users(session) -> Tuple[Any, Any]:
    query_text = "SELECT id, name, email, authority FROM users;"
    return execute_query(session=session, query_text=query_text)


def insert_user(session, name: str, password: str, email: str, authority: int) -> Tuple[Any, Any]:
    query_text = f"INSERT INTO users(name, password, email, authority) VALUES ('{name}', '{password}', '{email}', {authority});"
    return execute_query(session=session, query_text=query_text)


def delete_user(session, user_id: int) -> Tuple[Any, Any]:
    query_text = f"DELETE FROM users WHERE id='{user_id}';"
    return execute_query(session=session, query_text=query_text)
