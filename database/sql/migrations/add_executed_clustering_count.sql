-- 既存のデータベースに executed_clustering_count カラムを追加するマイグレーションスクリプト
-- 使用方法: docker exec -i <mysql-container-name> mysql -u root -p<password> pss_db < add_executed_clustering_count.sql

USE `pss_db`;

-- project_memberships テーブルに executed_clustering_count カラムを追加
ALTER TABLE project_memberships
ADD COLUMN executed_clustering_count INT NOT NULL DEFAULT 0 
COMMENT '実行されたクラスタリング回数（0: 初期クラスタリング）'
AFTER continuous_clustering_state;

-- user_image_clustering_states テーブルに executed_clustering_count カラムを追加
ALTER TABLE user_image_clustering_states
ADD COLUMN executed_clustering_count INT NULL DEFAULT NULL
COMMENT 'この画像がクラスタリングされた回数（NULL: 未クラスタリング, 0: 初期クラスタリング, 1~: 継続的クラスタリング）'
AFTER is_clustered;

-- 既存のクラスタリング済み画像の executed_clustering_count を 0 に設定
UPDATE user_image_clustering_states
SET executed_clustering_count = 0
WHERE is_clustered = 1;

SELECT '✅ マイグレーション完了: executed_clustering_count カラムを追加しました' AS message;
