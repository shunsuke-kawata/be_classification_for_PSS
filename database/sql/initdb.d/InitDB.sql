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
    init_clustering_state TINYINT(1) NOT NULL DEFAULT 0, -- クラスタリング状態（0: 未実行, 1: 実行中, 2: 完了）
    root_folder_id VARCHAR(22), -- 外部キー制約なし
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
    id VARCHAR(22) PRIMARY KEY, -- ChromaDBのID
    name VARCHAR(255) NOT NULL,
    is_created_caption TINYINT(1) NOT NULL DEFAULT 0,
    caption TEXT,
    project_id INT NOT NULL,
    uploaded_user_id INT NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- -- ========================
-- -- foldersテーブル（root_folder_id を追加）
-- -- ========================
-- CREATE TABLE folders (
--     id VARCHAR(22) PRIMARY KEY, -- ChromaDBのID
--     project_id INT NOT NULL,
--     root_folder_id VARCHAR(22) NOT NULL, -- その階層構造のルートID
--     parent_folder_id VARCHAR(22), -- NULL ならルートフォルダ
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

--     FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
--     FOREIGN KEY (parent_folder_id) REFERENCES folders(id) ON DELETE CASCADE
-- );



-- -- ========================
-- -- インデックス（検索性能向上）
-- -- ========================
-- CREATE INDEX idx_folders_project_id ON folders(project_id);
-- CREATE INDEX idx_folders_parent_folder_id ON folders(parent_folder_id);
-- CREATE INDEX idx_folders_root_folder_id ON folders(root_folder_id);
-- CREATE INDEX idx_images_project_id ON images(project_id);
-- CREATE INDEX idx_images_folder_id ON images(folder_id);

-- ========================
-- 初期データの挿入
-- ========================
INSERT INTO users (name, password, email, authority) 
VALUES 
('system', 'system', 'sys@sys.com', 1),
('user', 'user', 'user@user.com', 0);