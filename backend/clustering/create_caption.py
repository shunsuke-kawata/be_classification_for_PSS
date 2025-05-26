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

client = OpenAI(api_key=OPENAI_API_KEY)

# base64エンコードする関数
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# フォーマットチェック関数
def check_format_sentence(caption):
    pattern = r'The main object is .+? .+?\.\s*It\'s used for .+?\.\s*Its hypernym is .+?\.'
    return re.match(pattern, caption) is not None

# キャプション生成関数
def generate_caption(encoded_image):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Explain the image in following sentence format. The main object is {color and shape} {object name}. It's used for {usage of the main object}. Its hypernym is {hypernym of the main object}."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            caption = response.choices[0].message.content.strip()
            caption = caption.replace('"', '')

            if check_format_sentence(caption):
                return True, caption
            else:
                print(f"Attempt {attempt + 1} failed: Invalid format")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return False, None

class CaptionManager:
    
    def _encode_image(image_path:Path)->str | None:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None 
    
    def _check_format(sentence:str)->bool:
        pattern = r'The main object is .+? .+?\.\s*It\'s used for .+?\.\s*Its hypernym is .+?\.'
        return re.match(pattern, sentence) is not None
    
# メイン関数
def main():
    json_output_path = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = []

    image_paths = glob("imgs/*.jpg")
    
    for idx, image_path in enumerate(image_paths):
        encoded_image = encode_image(image_path)
        if encoded_image is None:
            continue

        success, caption = generate_caption(encoded_image)

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