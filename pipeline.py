from typing import Dict, List, Any, Union
from PIL import Image
import requests
import torch
import base64
import os
from io import BytesIO
from models.blip_feature_extractor import blip_feature_extractor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PreTrainedPipeline():
    def __init__(self, path=""):
        # load the optimized model
        self.model_path = os.path.join(path, 'model_large_retrieval_coco.pth')
        self.model = blip_feature_extractor(
            pretrained=self.model_path,
            image_size=384,
            vit='large',
            med_config=os.path.join(path, 'configs/med_config.json')
        )
        self.model.eval()
        self.model = self.model.to(device)

        image_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __call__(self, inputs: Union[str, "Image.Image"]) -> List[float]:
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`dict`:. The object returned should be a dict like {"feature_vector": [0.6331314444541931,0.8802216053009033,...,-0.7866355180740356,]} containing :
                - "feature_vector": A list of floats corresponding to the image embedding.
        """
        parameters = {"mode": "image"}
        if isinstance(inputs, str):
            # decode base64 image to PIL
            image = Image.open(
                BytesIO(base64.b64decode(inputs))).convert("RGB")
        elif isinstance(inputs, "Image.Image"):
            image = Image.open(inputs).convert("RGB")

        image = self.transform(image).unsqueeze(0).to(device)

        text = ""
        with torch.no_grad():
            feature_vector = self.model(image, text, mode=parameters["mode"])[
                0, 0].tolist()
        # postprocess the prediction
        return feature_vector
