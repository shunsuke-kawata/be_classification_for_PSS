# classification_for_PSS
卒業研究用のリポジトリ　　ユーザごとに画像分類結果を最適化するUIおよび分類アルゴリズムを構築する

## 事前準備
- Docker Desktopのインストール
- gitにssh接続するためのキーペアの作成

## 開発環境
|  |  |
| -- | -- |
| OS | MacOS(Apple M2) | 
| フロントエンド | Next,TypeScript | 
| バックエンド | FastAPI,Python |
| データベース | MySQL,SQL |
| ストレージサービス | S3を想定 |

## 環境構築
1. ターミナルで適当なフォルダで```mkdir classification_for_PSS```(任意の名前でOK)
1. ```cd classification_for_PSS```(作成したフォルダに移動)
1. ```git clone git@github.com:shunsuke-kawata/fe_classification_for_PSS.git```
1. ```git clone git@github.com:shunsuke-kawata/be_classification_for_PSS.git```

## 環境変数ファイルの作成
- ポート番号やDBの設定などを記述する環境変数ファイルを作成する必要がある

1. ターミナルで```be_classification_for_PSS```まで移動する
1. ```touch .env```
1. 以下の例のように記述する(ポート番号などローカルで使用するものは適宜変更する)

    .env

    ```
    FRONTEND_PORT=5002
    FRONTEND_PORT_IN_CONTAINER=3000
    BACKEND_PORT=8002
    DATABASE_PORT=3306
    MYSQL_ROOT_PASSWORD=root
    MYSQL_DATABASE=pss_db
    MYSQL_USER=user
    MYSQL_PASSWORD=user
    TZ=Asia/Tokyo
    ```

## 起動方法
1. ターミナルで```be_classification_for_PSS```まで移動する
1. ```docker-compose build```(初回のみ)
1. ```docker-compose up```

## データベースのマイグレーション
- マイグレーション設定を自動で行っていないため、初回ビルドやデータ消去時にはマイグレーションが必要になる
- バックエンドコンテナかデータベースコンテナから実行できる

1. バックエンドコンテナから実行する方法
    1. バックエンドコンテナに入る
    1. ```python3 db_utils/migrate.py```

1. データベースコンテナから実行する方法
    1. データベースコンテナに入る
    1. ```mysql -u {MYSQL_USER} -p < sql/CreateTable.sql```
    1. パスワードを入力する画面になるのでパスワード(```{MYSQL_PASSWORD}```)を入力

## データベースの削除
- 一度やり直したい、データベースのスキーマ定義を変更したい時に実行する

1. ローカルの```database/data/```の中身をフォルダごと削除する(データベースを永続化
するためのvolumeディレクトリ)
1. コンテナを再度起動する
1. データベースコンテナに入る
1. ```mysql -u {MYSQL_USER} -p < sql/DropTable.sql```

## データベース構造
```database/sql/PssdbCreateTable.sql```または```backend/db_utils/migrate.py```のコードに記載

## バックエンドAPIのdocsを更新する
1. ```http://localhost:{BACTEND_PORT}/system/docs/update```を叩く
1. ```backend/index.html```が作成されるので```docs```フォルダに移動する
1. pushする

## dockerコマンド
- イメージビルド ```docker-compose build```
- キャッシュを使用せずにイメージビルド　```docker-compose build --no-cache```
- コンテナ作成 ```docker-compose up```
- コンテナ作成(バックグラウンド実行) ```docker-compose up -d```
- コンテナ情報一覧表示 ```docker ps```
- コンテナ内に入る ```docker exec -it {コンテナID} bash```
- コンテナに対してコンテナ環境からコードを実行する　```docker exec -it {コンテナID} {実行するコマンド}```
- 起動するコンテナを選択して実行 ```docker-compose up {コンテナ名}```
