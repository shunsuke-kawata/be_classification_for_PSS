from sentence_transformers import SentenceTransformer

class SentenceEmbeddingsManager:
    _model = SentenceTransformer("all-MiniLM-L6-v2")  # クラス変数として1度だけ初期化

    @classmethod
    def sentence_to_embedding(cls, sentence: str) -> list[float]:
        return cls._model.encode(sentence)
    