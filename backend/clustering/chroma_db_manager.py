import base64
import json
import os
import re
import uuid
import chromadb
from embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from utils import Utils
class ChromaDBManager:
    
    class ChromaDocument:
        def __init__(self,document:str):
            self._document = document
        
        @property
        def document(self)->str:
            return self._document
        
        def get_split_document(self)->list[str]:
            return re.findall(r'.+?\.(?=\s+[A-Z]|$)', self._document)
    class ChromaMetaData:
        def __init__(self, path: str,document:str,is_success: bool,id:str | None=None):
            #新規作成の場合UUIDを設定,そうでない時、引数から継承
            if id is None:
                self._id = Utils.generate_uuid()
            else:
                self._id = id
            
            self._path = path
            self._is_success = is_success
            self._document = document

        @property
        def id(self) -> str:
            return self._id

        @property
        def path(self) -> str:
            return self._path

        @property
        def document(self)->"ChromaDBManager.ChromaDocument":
            return ChromaDBManager.ChromaDocument(self._document)

        @property
        def is_success(self) -> bool:
            return self._is_success

        def to_dict(self) -> dict:
            return {
                "id": self.id,
                "path": self.path,
                "is_success": self.is_success,
                "document":self.document
            }
    
        
    def __init__(self, colection_name:str,path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=colection_name)

    def add(self,documents:list[str], metadatas:list[ChromaMetaData],embeddings:list[list[float]] = None)->None:
        existing_paths = {meta.path for meta in self.get_all_metadata()}

        filtered_indices = [i for i, meta in enumerate(metadatas) if meta.path not in existing_paths]

        if not filtered_indices:
            return

        # フィルター後のデータを抽出
        filtered_documents = [documents[i] for i in filtered_indices]
        filtered_metadatas = [metadatas[i] for i in filtered_indices]

        kwargs = {
            "ids": [meta.id for meta in filtered_metadatas],
            "documents": filtered_documents,
            "metadatas": [meta.to_dict() for meta in filtered_metadatas],
        }
        filtered_embeddings = [embeddings[i] for i in filtered_indices] if embeddings else None
        if filtered_embeddings:
            kwargs["embeddings"] = filtered_embeddings

        self.collection.upsert(**kwargs)

    def get_all(self)->dict[str,list]:
        results = self.collection.get(include=["documents", "metadatas","embeddings"],limit=None)
        return results

    def get_all_metadata(self) -> list[ChromaMetaData]:
        all_data = self.get_all()
        return [self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in all_data["metadatas"]]
    
    def get_all_documents(self) -> list[ChromaDocument]:
        all_data = self.get_all()
        return [self.ChromaDocument(document=document) for document in all_data["documents"]]
    
    def get_all_embeddings(self)->list[list[float]]:
        all_data = self.get_all()
        return all_data
    
    def update(self, ids:list[str], documents:list[str], metadatas:list[ChromaMetaData],embeddings:list[list[float]] = None)->None:
        
        metadata_dict = [meta.to_dict() for meta in metadatas]

        kwargs = {"ids": ids, "documents": documents, "metadatas": metadata_dict}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)
    
    def delete(self, ids:list[str])->None:
        self.collection.delete(ids=ids)

    def query_by_ids(
        self,
        query_ids: list[str]
    ) -> dict[str, list]:
        results = self.collection.get(
            ids=query_ids,
            include=["documents", "metadatas", "embeddings"]
        )
        return {
            'ids': results['ids'],
            'metadatas': [self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in results['metadatas']],
            'documents':[self.ChromaDocument(document=document) for document in results['documents']],
            'embeddings': results['embeddings'],
        }
    
    def query_by_embeddings(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        distance_threshold: float | None = None
    ) -> dict[str, list]:
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        result_dict = {
            'ids':results['ids'][0],
            'metadatas':[self.ChromaMetaData(id=metadata['id'],path=metadata['path'],document=metadata['document'],is_success=metadata['is_success']) for metadata in results['metadatas'][0]],
            'documents':[self.ChromaDocument(document=document)for document in results['documents'][0]],
            "embeddings":results['embeddings'][0],
            'distances':results['distances'][0],
        }
        

        # distance_threshold が指定されていない場合はそのまま返す
        if distance_threshold is not None:
            filtered_result = {
                'ids': [],
                'metadatas': [],
                'documents': [],
                'embeddings': [],
                'distances': [],
            }

            for _id, meta, doc, emb, dist in zip(results['ids'][0],results['metadatas'][0],results['documents'][0],results['embeddings'][0],results['distances'][0]):
                meta_obj = self.ChromaMetaData(id=meta['id'],path=meta['path'],document=meta['document'],is_success=meta['is_success'])
                if dist <= distance_threshold:
                    filtered_result['ids'].append(_id)
                    filtered_result['metadatas'].append(meta_obj)
                    filtered_result['documents'].append(doc)
                    filtered_result['embeddings'].append(emb)
                    filtered_result['distances'].append(dist)
            return filtered_result
                
        return result_dict
 

if __name__ == "__main__":

    with open('captions_20250522_013210.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    succeed_data = [item for item in data if item['is_success']]

    # ChromaMetaDataインスタンスを作成
    metadatas = [
        ChromaDBManager.ChromaMetaData(
            path=item["path"],
            document=item["caption"],
            is_success=item["is_success"]
        )
        for item in succeed_data
    ]

    # Sentence Embedding（テキストベース）
    sentence_db_manager = ChromaDBManager("sentence_embeddings")
    embeddings = SentenceEmbeddingsManager.sentence_to_embedding("The main object is a red and white rectangular USB flash drive. It's used for data storage and transfer. Its hypernym is a data storage device.")
    
    res = sentence_db_manager.query_by_ids([
      "rZZikyqfQJyLd6tlJvdY_Q",
      "NgKreNJZSRyVqkRJy-jJkw",
      "BwjUo1ZOTv6PuR0M7lJcMA",
      "LfO_q4JQQ2ihbogWc1cUUg",
      "-yVtFwuqSHek6VUOqNBiQA",
      "4r-vbY17RwW_s2tj0pmImw",
      "kNubX9xbTj2oFkUhO-WJ7g",
      "ZWB-W6pASwaJAwyTCd5Qsw",
      "nq4BlJUJRkKTT2s8NFX3VQ",
      "Jt1jUpEATg6TiWePk7RGZg",
      "oJZbeyCQSKOlxmvhq6LG-w",
      "oLncXMb1Q4GdTjfFpArFkQ",
      "vJKDVreqSVaaXH9Nl-WosA",
      "SzxRZzZtS9OCvoMM-y12SA",
      "Dh-bhjHeQH2RB_4o65kwKQ",
      "6tdo_uvNQ8GTONHlolIDvw",
      "NiiZP7YSSj-EkvfRXN_skA",
      "Jd6rzqfNSnqtvGEksu5ixg",
      "USIlVlbgR9mKj8gBjuHs3A",
      "VLRfit8ETryvB8Gm_b09jA",
      "J40milS_QCigx5EGhjjAiQ",
      "xkLaUza8Qoy0XjtIMxa70g",
      "c-Yg9TQyThqwecxJNz8Isw",
      "qc6gLAhFRbGCj-Ij76vCDQ",
      "oDgWZ887TDOc3sL7r2sz6w",
      "P472aaQpSKet8f5xbiKAPg",
      "oLPSWb_NTsKAPzauLzOmsg",
      "EtdHsZiAQIKO5HMt3tIQ0A",
      "NpJ1s8wrTrGPFsIWJf3eWA",
      "i6_8W1oER7SS5UrO-xaOcg",
      "Y2YvFh0XT3m0qbByRATIAA",
      "mFIrf_OkSDKGA8ylbpJWpw",
      "3XEyLGcMRneinVvgDWXQKQ",
      "PJYQYyG7Rm-tbH9XHRPFwQ",
      "bQCtopOXQ3alEdCw6c5J2w",
      "jLlzRkz2TA-o2kgQgGdjYA",
      "sT7FSc3xR9ux6x--Zdw7gQ",
      "ziO58IUJS8iDqFdolOmPPw",
      "gYynSGMQTT-27f681RSK7Q",
      "_RxP5YQDRZeaLA7Z8ChEow",
      "rSjgdT4gSBqeZ4MezaQo1A",
      "HBCvl1ISTfyPILbVuUm-SQ",
      "3KdbiP9KR9ap2URfZFWpPg",
      "YiiLf6DGQfWzfXfknhKHqg",
      "vfe25gQERz2Ft1w_OSHTvA",
      "bNQlOxjNSpaxF67liYBIRQ",
      "FerftUfzSV2HW2ROT_rq8w",
      "SKckYhfJRj2n11-HEHDPeQ",
      "iG2_AOljSsiRS0VZw2UT8Q",
      "FeLZhDuNRL2d4PBp80jRaw",
      "yhotqbFeQ7eyG7B_EkqWjw",
      "QNH4d9oKT6GVNACoJ-S27A",
      "EbjlqJ-6RlekltAllgi8eg",
      "ivpMTvklQWWWJ5toPrpj2Q",
      "5T6xVEjXTuW2nfZpfR6Mhg",
      "bWyXF6B9TxCdSIPyQ-rctw",
      "hwfyJ00sTDSYBAqFkh1oBQ",
      "5Bu-nk3_RgqTbBaUQfEAOQ",
      "NOKuCJ8mQAu43EI1PXNb8w",
      "nWDfvL7JT2-JO7aKpvBvjg",
      "KTl2hpK7RiW1IrnvF1yOMg",
      "eU2d9YsGRjW651mKT3PL5Q",
      "ry8Fv6ZlQGWFN21sWcfq-Q",
      "1s2EhKuIQmiAhwpcUn9pBw",
      "LL1z1gXpQ-CxV9AfkuwVZQ",
      "hBY6mnX7QmuzgPWhBWYpMQ",
      "CoMudIZnTT-yKqmK-beqfg",
      "FbifEFxsSz68nOYTrEvKqA",
      "xBy85zL-RrqriGU4WtwlRQ",
      "kkx_8VHMSlq8rDh_QShL6A",
      "ofzbeI3nTWKs-BnwtiwZAg",
      "oxawppGjRT6lkdrvgfe__Q",
      "gdiUWJk5T76qu77-euDNUg",
      "ZbkUlDPJTHyFldQlPqGFhQ",
      "fhubyXIMSzOViE86_09BHg",
      "onAV9bi4QtqwalrWEWAQaA",
      "u5o_uGCXT82NCSfPMda3Uw",
      "VQHipL4XSDqYoASfJCDHTw",
      "syiqipuiSDaw84XZLqGm3g",
      "4x2flAgSRNWN9GQdlJsVCQ",
      "tgv9EWBGTYuulx-x5Mcw3Q",
      "hdsV9skVSReNH-2dxmFzxg",
      "bdj-7eZbQwqb9JALmI8P0A",
      "kuleDAYZSrizZ7UpN4VHoQ",
      "LerVVnqwRki-souNWIsqag",
      "Yb28qc_hTJCDZ_Y7FLahdw",
      "UP5x1u_cSw2Vq91AEQqJpQ",
      "yARSTceJT8OY-imI1NodfA",
      "LDiw06sKSXqdqqe6MTYbrg",
      "aQc8KlpZSxCWi2SHK2Li_Q",
      "g4OLQXFaQ6KH6bynlUfrFg",
      "B0Iyk0vjRfyMwZcTURIn-Q",
      "p9oaAqnxSlWU59G_J19f9Q",
      "5YsBTs0fRRaEcAkOovU17w",
      "8lpTFnbRSYquaypDLr6xIg",
      "orGMp3LsSsmMya7czryDsw",
      "NWrV0QyESeaOE_GTTpyzdA",
      "OYx4JgfFSHmk7sVYsl_Wtg",
      "SpVsdgXdRymm02mm79fECA",
      "3l3zRB3mTjqwCoFnNVqyFQ",
      "bjgbK48PQ2eZDGzGjkJpLA",
      "-dhvy7ZuT1-pwp4-RzAqvg",
      "uoBmbRMzRDeHM5xpPXdkfQ",
      "qNHXvmy2TVWeuqmvuDv6sg",
      "iwCd3YmFQkanO3YEsXTRLQ",
      "L4gIeiDPSjmrhiujSfTw5Q",
      "sSPlAMWzQu2a4QFNlQuqug",
      "AqAab6fySpKMd1_BpNWqqA",
      "4WuAZ91QQFWERVvDJVtR0g",
      "JXif97-2Qie0PppoQfbhqg",
      "V10CX6E6QCu-xJ42JaoA8w",
      "Ozd6li4YSziW6HeNvhIkyw",
      "DA1WJtlTRPG3dMG8X1kGOg",
      "V3MsuW4ESgix3MQZBx3H5Q",
      "6qsuH_ScSO6ZHrx1y4cj5w",
      "ZIgsxX0hRzePSNKZ14ehNw",
      "fyWS12FARe2s-B8ZrugZbQ",
      "dIpLXH84Qa6CbLAVNpv0EQ",
      "IveAJezgSmaDY5u4graeuw",
      "W8QOrASpQPGhYzaCijLkcw",
      "1aSDQrfwQOGNRh1tb3A_XQ",
      "s3fc-VwyTVWljBKoZaXG3w",
      "_-gp67OTQta13z3pU8u_ow",
      "Ji9OoWrgRiWriFdl-ve5tQ",
      "Wd1x5OiTRPug0divNboLgA",
      "H7ipmlhCSOiQc3lCIMPrZg",
      "HH9RPhk-TNG2ZELqFIM1uw",
      "qpDmALD9TmmlrjpoehrBkQ",
      "e6YgNL7kQv-TybIK6YgF-w",
      "WOf8whvuQMy37kZ0zvwrjQ",
      "WVRXpRA4SSS17PRQnTDliQ",
      "y94cTiTFQeuFimrG7sh3eg",
      "Fi3BBfgHSb6WAObzeGxbhQ",
      "hAxtFDhBTumV2VssbESSCA",
      "hKZMy-jySOaecl22q9EAkQ",
      "ur6YbsK_SwaQJkfxGWTvuA",
      "-Pmi0D_2TOup0RvKDj1xWg",
      "LrWN19-DRCSF8qt_uMpvMQ",
      "pl0QTu8zTYuKqYOoxb-LCA",
      "o3xz585hTiyOAZRfon2ZJg",
      "pB9bdIFlTdqUnpyL2V9qAw",
      "vF2TbCrpQbu_KnDLGo5Ovg",
      "iJhiWnIVTVCdOvDTINssew",
      "VP9EJQUMSwKR8yViL6sdXg",
      "md2-KA2MQLuDIhCAUBZcMQ",
      "S4fXy62YSSCKUaNWGuK1lQ",
      "ixnnbAY3REOxa96n0zM5LQ",
      "mXKrgVtTTcKc6KUBmUAQgg",
      "CeaSdSpvT0SVDqetTl_6Ag",
      "bSDsgmJLRp6YLO53L20zAw",
      "XfoycdxuT26p8LuH7lPTyw",
      "-_xNrrMrTgq3SnofTjhDeg",
      "4Eeasyo0RS2us4DVUAuKRw",
      "FP88QzFKSAuvG7UDbCrmDA",
      "aUrGO1enTYO-84K4DuEX8g",
      "SqSbyIMmSGO3TC1zhm-TQw",
      "wqY3QgHlS1igmMTcx86mtg",
      "F_WR1SWJTPKxIvauDHH6_w",
      "bGuJhjaBRGuA8bH27J-Mvw",
      "yxUrGuzwSs-4qUyYyecwiQ",
      "yJv8I3gDQA-4J9T2c2zl2w",
      "Ks8faIm_Q4iLKcnxpTEx7Q",
      "BS9ubErlQs6yuCpRP3WU-g",
      "-sDTfeETQTigy7VtoDpv9g",
      "AZon3FECQTG0WBoezlge1w",
      "Bug66FemQdekNN0v9XLvag",
      "XY5j_iY-QJ-ZR41SgtCZXQ",
      "h76_aSEoQAmiTdGnZOPtSA",
      "JgZvFX5xSJ2qQtpQRHn4EA",
      "73sc7MPHSS-dLBu0jBbZ_A",
      "4COSYp5SRp-DGFXVkcr3iw",
      "6j0FWQWlTP6rCjnJgtlvvA",
      "n55PUpLqQCK0AFhW6ZSG_g",
      "oLol0cB3QByL6AbOguersA",
      "TdPDEBSeRTKqx87vPmuB7Q",
      "EDx-7NcVSDe4h1768xnH1Q",
      "YnXqa3stS5i0zAqLzia1nQ",
      "T0bZGJWCTCClhU6S-qVGIg",
      "qbQlMIDURoeRpq0ZMJBtUA",
      "yZJm1eJmR8iZwk30U0TCuA",
      "O3rreBFcTJSBoAoupxQOIg",
      "w33iIpsMRpaxiEzsiStmbQ",
      "QMr8uqxYTg2R99LmTyX58A",
      "twMZ_ZsfR8eTUjuv2mMlBg",
      "ag8ITBO1Q-iocbfQ9RLr8g",
      "qdHs4OCkQRu3m-jgaVMfDw",
      "SRrXXhqHRaOqzGu_Yhyv2Q",
      "wOfJMS-1QGGmZ_oNsBUiFw",
      "SGBAgYr5RmyROKOG7p324g",
      "t4vTHR6sSjeEtzTQTLCqZQ",
      "a_yFccEbRT624VJaZtSbSg"
    ])
    for item in res['documents']:
        print(item.get_split_document())


    # sentence_db_manager.add(
    #     documents=[meta.document for meta in metadatas],  # または meta.caption にしたければ ChromaMetaData にフィールド追加が必要
    #     metadatas=metadatas,
    #     embeddings=[SentenceEmbeddingsManager.sentence_to_embedding(item["caption"]) for item in succeed_data]
    # )
     
    # # Image Embedding（画像ベース）
    # image_db_manager = ChromaDBManager("image_embeddings")
    # image_db_manager.add(
    #     documents=[meta.path for meta in metadatas],
    #     embeddings=[ImageEmbeddingsManager.image_to_embedding(f"./imgs/{item['path']}") for item in succeed_data],
    #     metadatas=metadatas
    # )
    