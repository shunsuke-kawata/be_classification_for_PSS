from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

class ImageEmbeddingsManager:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 学習済みモデルを初期化して特徴抽出器に変更（最終層を除去）
    _model = resnet18(pretrained=True)
    _model = torch.nn.Sequential(*list(_model.children())[:-1])
    _model.eval()
    _model.to(_device)

    # 前処理もクラス変数として定義
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    @classmethod
    def image_to_embedding(cls, image_path: str) -> list[float] | None:
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = cls._transform(image).unsqueeze(0).to(cls._device)
            with torch.no_grad():
                embedding = cls._model(image_tensor).squeeze().cpu().numpy()
            return embedding.tolist()
        except Exception as e:
            return None