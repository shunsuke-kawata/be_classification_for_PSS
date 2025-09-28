.PHONY: build \
	up \
	down \
	up-init-data \
	build-init-data


# Dockerイメージのビルド
build:
	@ docker compose build

# コンテナを起動
up:
	@ if [ "$(filter -d,$(MAKECMDGOALS))" ]; then \
		docker compose up -d; \
	else \
		docker compose up; \
	fi

# コンテナの停止と削除
down:
	@ docker compose down

# データベースとフォルダを初期化してコンテナを起動
up-init-data:
	@ echo "データベースとフォルダを初期化しています..."
	@ docker compose down
	@ sudo rm -rf database/mysql_data database/mongo_data backend/chroma_db backend/images backend/output
	@ mkdir -p database/mysql_data database/mongo_data backend/chroma_db backend/images backend/output
	@ if [ "$(filter -d,$(MAKECMDGOALS))" ]; then \
		echo "コンテナをバックグラウンドで起動しています..."; \
		docker compose up -d; \
	else \
		echo "コンテナを起動しています..."; \
		docker compose up; \
	fi

# データベースとフォルダを初期化してビルド後にコンテナを起動
build-init-data:
	@ echo "データベースとフォルダを初期化しています..."
	@ docker compose down
	@ sudo rm -rf database/mysql_data database/mongo_data backend/chroma_db backend/images backend/output
	@ mkdir -p database/mysql_data database/mongo_data backend/chroma_db backend/images backend/output
	@ echo "Dockerイメージをビルドしています..."
	@ docker compose build
	@ if [ "$(filter -d,$(MAKECMDGOALS))" ]; then \
		echo "コンテナをバックグラウンドで起動しています..."; \
		docker compose up -d; \
	else \
		echo "コンテナを起動しています..."; \
		docker compose up; \
	fi

# オプション用のダミーターゲット
-d:
	@ :

