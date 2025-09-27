-- 使用するデータベース
USE `pss_db`;

-- ========================
-- usersテーブル
-- ========================
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    authority TINYINT(1) NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
);

-- ========================
-- projectsテーブル（root_folder_id は UUID 文字列として扱う）
-- ========================
CREATE TABLE projects (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    description TEXT,
    original_images_folder_path VARCHAR(255) NOT NULL,
    owner_id INT NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ========================
-- project_membershipsテーブル（多対多）
-- ========================
CREATE TABLE project_memberships (
    user_id INT NOT NULL,
    project_id INT NOT NULL,
    init_clustering_state TINYINT(1) NOT NULL DEFAULT 0, -- クラスタリング状態（0: 未実行, 1: 実行中, 2: 完了 3:失敗）
    mongo_result_id VARCHAR(22) NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

    PRIMARY KEY (user_id, project_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
-- ========================
-- imagesテーブル
-- ========================
CREATE TABLE images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    is_created_caption TINYINT(1) NOT NULL DEFAULT 0,
    caption TEXT,
    project_id INT NOT NULL,
    clustering_id VARCHAR(22) NOT NULL,
    chromadb_sentence_id VARCHAR(22) NOT NULL,
    chromadb_image_id VARCHAR(22) NOT NULL,
    is_deleted TINYINT(1) NOT NULL DEFAULT 0,
    deleted_at TIMESTAMP(6) NULL DEFAULT NULL,
    uploaded_user_id INT NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_user_id) REFERENCES users(id) ON DELETE CASCADE
);





-- -- ========================
-- -- インデックス（検索性能向上）
-- -- ========================
CREATE INDEX idx_images_project_id ON images(project_id);

-- ========================
-- 初期データの挿入
-- ========================
INSERT INTO users (name, password, email, authority) 
VALUES 
('system', 'system', 'sys@sys.com', 1),
('user', 'user', 'user@user.com', 0);

INSERT INTO projects (name, password, description, original_images_folder_path, owner_id) 
VALUES 
('project1', 'project1', 'project1', 'project1', 1),
('project2', 'project2', 'project2', 'project2', 2);

INSERT INTO project_memberships (user_id, project_id, init_clustering_state, mongo_result_id) 
VALUES 
(1, 1, 0, 'project1'),
(2, 2, 0, 'project2');