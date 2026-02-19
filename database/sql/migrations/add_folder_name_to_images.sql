-- imagesテーブルにfolder_nameカラムを追加するマイグレーションスクリプト
-- 使用方法: docker exec -i <mysql-container-name> mysql -u root -p<password> pss_db < add_folder_name_to_images.sql

USE `pss_db`;

-- images テーブルに folder_name カラムを追加
ALTER TABLE images
ADD COLUMN folder_name VARCHAR(255) NULL DEFAULT NULL
COMMENT 'アップロード時のフォルダ名（フォルダアップロード時のみ）'
AFTER name;

SELECT '✅ マイグレーション完了: folder_name カラムを追加しました' AS message;
