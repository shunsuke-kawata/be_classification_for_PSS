import os
from dotenv import load_dotenv
from enum import IntEnum

# .envファイルの内容を読み込む
load_dotenv()

# ポート設定
FRONTEND_PORT = os.environ['FRONTEND_PORT']
FRONTEND_PORT_IN_CONTAINER = os.environ['FRONTEND_PORT_IN_CONTAINER']
BACKEND_PORT = os.environ['BACKEND_PORT']

# MySQL 設定
DATABASE_PORT = os.environ['DATABASE_PORT']
DATABASE_PORT_IN_CONTAINER = os.environ['DATABASE_PORT_IN_CONTAINER']
MYSQL_ROOT_PASSWORD = os.environ['MYSQL_ROOT_PASSWORD']
MYSQL_DATABASE = os.environ['MYSQL_DATABASE']
MYSQL_USER = os.environ['MYSQL_USER']
MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
MYSQL_HOST = os.environ['MYSQL_HOST']

# MongoDB 設定
MONGO_USER = os.environ['MONGO_USER']
MONGO_PASSWORD = os.environ['MONGO_PASSWORD']
MONGO_HOST = os.environ['MONGO_HOST']
MONGO_PORT = os.environ['MONGO_PORT']
MONGO_DB = os.environ['MONGO_DB']
MONGO_AUTH_DB = os.environ['MONGO_AUTH_DB']
MONGO_INITDB_ROOT_USERNAME = os.environ['MONGO_INITDB_ROOT_USERNAME']
MONGO_INITDB_ROOT_PASSWORD = os.environ['MONGO_INITDB_ROOT_PASSWORD']

# 管理者コード
ADMINISTRATOR_CODE = os.environ['ADMINISTRATOR_CODE']

# OpenAI API キー
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# パス設定
DEFAULT_IMAGE_PATH = os.environ.get('DEFAULT_IMAGE_PATH', 'images')
DEFAULT_OUTPUT_PATH = os.environ.get('DEFAULT_OUTPUT_PATH', 'output')
NEXT_PUBLIC_DEFAULT_IMAGE_PATH = os.environ.get('NEXT_PUBLIC_DEFAULT_IMAGE_PATH', '/images')

# クラスタリングステータス定義
class INIT_CLUSTERING_STATUS(IntEnum):
    NOT_EXECUTED = 0
    EXECUTING = 1
    FINISHED = 2
    FAILED = 3

class CONTINUOUS_CLUSTERING_STATUS(IntEnum):
    NOT_EXECUTABLE = 0
    EXECUTING = 1
    EXECUTABLE = 2

MAJOR_COLORS = colors = [
    "red", "blue", "yellow", "green", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "cyan", "magenta", "beige", "ivory",
    "turquoise", "teal", "lime", "olive", "navy", "maroon", "coral",
    "salmon", "khaki", "violet", "indigo", "gold", "silver", "bronze",
    "crimson", "plum", "orchid", "lavender", "mint", "aqua", "azure",
    "chocolate", "tan", "peach", "apricot", "amber", "burgundy",
    "mustard", "emerald", "jade", "rose", "ruby", "sapphire",
    "skyblue", "aquamarine", "chartreuse", "fuchsia", "periwinkle",
    "slate", "charcoal", "sand", "seashell", "honey", "cream",
    "snow", "wheat", "moccasin", "tomato", "firebrick", "orchid",
    "lavenderblush", "midnightblue", "royalblue", "steelblue",
    "dodgerblue", "deepskyblue", "lightblue", "powderblue",
    "forestgreen", "seagreen", "lightgreen", "darkgreen", "springgreen",
    "palegreen", "chartreuse", "greenyellow", "lawngreen",
    "darkolivegreen", "darkslategray", "slategray", "lightgray",
    "antiquewhite", "bisque", "blanchedalmond", "burlywood",
    "cornsilk", "linen", "oldlace", "papayawhip", "peachpuff",
    "tan", "sienna", "peru", "rosybrown"
]

MAJOR_SHAPES = [
    "circle", "square", "triangle", "rectangle", "oval", "ellipse", "diamond", "star", "heart",
    "sphere", "cube", "cone", "cylinder", "pyramid", "prism", "torus",
    "polygon", "hexagon", "pentagon", "octagon", "line", "curve", "arc", "angle", "edge", "surface",
    "round", "flat", "straight", "curved", "bent", "twisted", "spiral", "wavy", "irregular",
    "smooth", "rough", "sharp", "pointed", "blunt",
    "shape", "form", "figure", "outline", "contour", "pattern", "structure", "silhouette",
    "geometry", "dimension", "frame", "profile"
]