# classification_for_PSS
卒業研究用のリポジトリ　　ユーザごとに画像分類結果を最適化するUIおよび分類アルゴリズムを構築する

## 事前準備
- Docker Desktop
- gitにssh接続するためのキーペアの作成

## 使用技術
|  |  |
| -- | -- |
| フロントエンド | Next,TypeScript | 
| バックエンド | FastAPI,Python |
| データベース | MySQL,SQL |

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
    FRONTEND_PORT=5000
    BACKEND_PORT=8000
    DB_PORT=5432
    ```

## 起動方法
1. ターミナルで```be_classification_for_PSS```まで移動する
1. ```docker-compose build```(初回のみ)
1. ```docker-compose up```
