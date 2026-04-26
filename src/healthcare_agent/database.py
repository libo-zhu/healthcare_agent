from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from healthcare_agent.config import get_int_env, load_dotenv


DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = 3306
DEFAULT_DB_USER = "root"
DEFAULT_DB_PASSWORD = "zhu203926"
DEFAULT_DB_NAME = "healthcare_agent_app"


def get_database_config() -> dict[str, Any]:
    load_dotenv()
    import os

    return {
        "host": os.environ.get("MYSQL_HOST", DEFAULT_DB_HOST),
        "port": get_int_env("MYSQL_PORT", DEFAULT_DB_PORT),
        "user": os.environ.get("MYSQL_USER", DEFAULT_DB_USER),
        "password": os.environ.get("MYSQL_PASSWORD", DEFAULT_DB_PASSWORD),
        "database": os.environ.get("MYSQL_DATABASE", DEFAULT_DB_NAME),
        "charset": "utf8mb4",
        "autocommit": False,
        "cursorclass": DictCursor,
    }


@contextmanager
def get_connection() -> Iterator[pymysql.Connection]:
    connection = pymysql.connect(**get_database_config())
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def ensure_database_schema() -> None:
    config = get_database_config()
    database = config.pop("database")
    config["autocommit"] = True
    connection = pymysql.connect(**config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE `{database}`")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(64) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    display_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    title VARCHAR(180) NOT NULL DEFAULT '新的健康评估',
                    mode ENUM('specialist','general') NOT NULL DEFAULT 'specialist',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_conversations_user_updated (user_id, updated_at),
                    CONSTRAINT fk_conversations_user
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id BIGINT NOT NULL,
                    role ENUM('user','assistant','system') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    metadata_json JSON NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_messages_conversation_id (conversation_id, id),
                    CONSTRAINT fk_messages_conversation
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
    finally:
        connection.close()
