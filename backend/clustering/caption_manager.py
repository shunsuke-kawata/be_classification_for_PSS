import base64
import json
from glob import glob
import os
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

import sys
sys.path.append('../')
from config import OPENAI_API_KEY

# .envファイルの内容を読み込見込む
load_dotenv()

class CaptionManager:
    """
    画像からOpenAIでキャプション生成し、厳密な7フィールドフォーマットを検証する。

    フォーマット（1行のみ）:
    The main object is {color} {object}. Its size is {size}. Its weight is {weight}.
    It's used for {usage}. Its material is {material}. Its safety is {safety}. Its category is {category}.

    ルール:
    - 生成文は1行のみ（改行禁止）
    - 固定フレーズは言い換え禁止（case-sensitive）
    - 各文末は '.'、文末の直後はスペース1つ（最後の文末はスペースなし）
    - フィールド値の中に '.' を含めない
    - 禁止文字: {} [] <>
    - unknown 禁止（不確かな場合も最善推定で埋める）
    - size: tiny, small, medium, large, very large
    - weight: very light, light, medium, heavy, very heavy
    - safety: safe, slightly dangerous, dangerous
    """

    # 固定ラベル
    _SIZE_ALLOWED_PATTERN = r"tiny|small|medium|large|very large"
    _WEIGHT_ALLOWED_PATTERN = r"very light|light|medium|heavy|very heavy"
    _SAFETY_ALLOWED_PATTERN = r"safe|slightly dangerous|dangerous"

    # 厳密フォーマット正規表現
    _FORMAT_PATTERN = re.compile(
        rf"^The main object is (?P<main>[^.]+)\. "
        rf"Its size is (?P<size>{_SIZE_ALLOWED_PATTERN})\. "
        rf"Its weight is (?P<weight>{_WEIGHT_ALLOWED_PATTERN})\. "
        rf"It's used for (?P<usage>[^.]+)\. "
        rf"Its material is (?P<material>[^.]+)\. "
        rf"Its safety is (?P<safety>{_SAFETY_ALLOWED_PATTERN})\. "
        rf"Its category is (?P<category>[^.]+)\.$"
    )

    PROMPT = (
        "Explain the object in the image in following sentence format. "
        "Describe the MAIN OBJECT in the image. "
        "Output must be a single line only. Do NOT add any extra text. Do NOT use line breaks. "
        "Use EXACTLY these fixed phrases (case-sensitive) and do NOT paraphrase them: "
        "\"The main object is \", \"Its size is \", \"Its weight is \", \"It's used for \", "
        "\"Its material is \", \"Its safety is \", \"Its category is \". "
        "Use exactly one space after each fixed phrase. "
        "Each sentence must end with a period '.', and there must be exactly one space after each period (except the last one). "
        "Do NOT include any curly braces {}, square brackets [], or angle brackets <> in your output. "
        "Do NOT use any additional periods '.' inside any field value (use commas or semicolons instead). "
        "Do NOT output the word 'unknown'. If you are not sure, make the best possible guess rather than using placeholders. "
        
        "Size must be chosen from EXACTLY one of: tiny, small, medium, large, very large. "
        "IMPORTANT: Choose size by comparing to everyday objects in the real world overall, NOT within the same object type. "
        "For example, even if a marker is large among pens, it is generally 'small' as a real-world object. "
        
        "Weight must be chosen from EXACTLY one of: very light, light, medium, heavy, very heavy. "
        "IMPORTANT: Assume the object is handheld. Choose weight by comparing within handheld objects in general "
        "(not within the same object type). For example, a phone is heavier than a pen; a stapler is heavier than a marker. "
        
        "Safety must be chosen from EXACTLY one of: safe, slightly dangerous, dangerous. "
        "Choose 'dangerous' ONLY if the main object is clearly dangerous and can easily cause injury "
        "(e.g., scissors, cutter/box cutter, knife, blade, razor, needle, broken glass). "
        "Choose 'slightly dangerous' for mild risk items that require some care (e.g., pointed tip, hard edges, small parts). "
        "Otherwise choose 'safe'. "
        "Do NOT add explanations for safety. "
        
        "Output exactly this structure: "
        "The main object is [base color + object name]. "
        "Its size is [one allowed label]. "
        "Its weight is [one allowed label]. "
        "It's used for [free text]. "
        "Its material is [free text]. "
        "Its safety is [one allowed label]. "
        "Its category is [free text]."
    )
    
    @classmethod
    def encode_image(cls, image_path: Path) -> str | None:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    @classmethod
    def _sanitize_single_line(cls, text: str) -> str:
        """生成テキストを単一行に正規化（改行除去・連続空白を1つに）"""
        if text is None:
            return ""
        s = text.replace('"', "").strip()
        s = s.replace("\r", " ").replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _check_format(cls, sentence: str) -> bool:
        """厳密チェック（改行なし・禁止文字なし・unknown禁止・size/weight/safety固定）"""
        s = (sentence or "").strip()

        # 改行禁止
        if "\n" in s or "\r" in s:
            return False

        # 禁止文字
        if any(c in s for c in "{}[]<>"):
            return False

        # unknown 禁止
        if re.search(r"\bunknown\b", s, flags=re.IGNORECASE):
            return False

        # フォーマット厳密一致
        m = cls._FORMAT_PATTERN.fullmatch(s)
        if not m:
            return False

        # 空欄を弾く
        for key in ("main", "size", "weight", "usage", "material", "safety", "category"):
            if not m.group(key).strip():
                return False

        return True

    @classmethod
    def generate_caption(cls, encoded_image: str, openai_api_key: str, max_retries: int = 3) -> tuple[bool, str | None]:
        """
        Base64エンコード画像からキャプション生成

        Args:
            encoded_image: Base64エンコードされた画像
            openai_api_key: OpenAI APIキー
            max_retries: 最大リトライ回数

        Returns:
            tuple[bool, str|None]: (成功フラグ, キャプション)
        """
        client = OpenAI(api_key=openai_api_key)
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 OpenAIでキャプション生成中 (試行 {attempt + 1}/{max_retries})")
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": cls.PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=220
                )
                
                raw_caption = response.choices[0].message.content or ""
                caption = cls._sanitize_single_line(raw_caption)

                if cls._check_format(caption):
                    print(f"  ✅ キャプション生成成功: {caption}")
                    return True, caption
                else:
                    print(f"  ⚠️ 試行 {attempt + 1} 失敗: 形式が不正")
                    print(f"     ↳ 生成結果: {caption}")

            except Exception as e:
                print(f"  ❌ 試行 {attempt + 1} 失敗: {str(e)}")

        return False, None

    
# メイン関数
def main():
    json_output_path = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = []

    image_paths = glob("imgs/*.jpg")
    
    for idx, image_path in enumerate(image_paths):
        encoded_image = CaptionManager.encode_image(image_path)
        if encoded_image is None:
            continue

        success, caption = CaptionManager.generate_caption(encoded_image,openai_api_key=OPENAI_API_KEY)

        result = {
            "index": idx,
            "path": os.path.basename(image_path),
            "is_success": success,
            "caption": caption if caption else "Failed to generate caption"
        }
        results.append(result)

        # 毎ループごとにJSON保存
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write JSON at index {idx}: {e}")

if __name__ == "__main__":
    main()