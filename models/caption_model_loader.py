"""
Caption Model Loader for GIT (Generative Image-to-Text)

Supports two modes:
1. Real-time mode: Loads GIT model and extracts features during training
2. Precomputed mode: Loads pre-extracted features from CSV files (faster training)
"""

import json
import sys
import os
import pandas as pd
import torch
import PIL
import random
import numpy as np
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

# Add generativeimage2text to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'generativeimage2text'))

from common import qd_tqdm as tqdm
from tsv_io import load_from_yaml_file
from torch_common import torch_load, load_state_dict
from process_image import load_image_by_pil
from model import get_git_model
from azfuse import File


class CaptionModelLoader:
    """
    GIT Caption Model Loader with dual-mode support

    Modes:
    - use_precomputed=False: Load GIT model and extract features in real-time (slower)
    - use_precomputed=True: Load pre-extracted features from CSV (faster)
    """

    def __init__(self, model_name='GIT_LARGE_R_TEXTCAPS', model_path=None,
                 use_precomputed=False, precomputed_path=None):
        """
        Args:
            model_name: Name of GIT model to use
            model_path: Path to GIT model checkpoint
            use_precomputed: Whether to use pre-extracted features from CSV
            precomputed_path: Path to precomputed features directory
        """
        self.set_all_seeds()
        self.model_name = model_name
        self.use_precomputed = use_precomputed
        self.precomputed_path = precomputed_path

        # Use relative paths
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, 'output')

        self.model_path = model_path

        # Only load model if not using precomputed features
        if not use_precomputed:
            self.model, self.tokenizer = self.load_model_and_tokenizer()
            print(f"✓ GIT model loaded: {model_name} (real-time mode)")
        else:
            self.model = None
            self.tokenizer = None
            if precomputed_path is None:
                # Default path for precomputed features
                self.precomputed_path = os.path.join(project_root, 'precomputed_features',
                                                     model_name, 'normalized_result')
            print(f"✓ Using precomputed features from: {self.precomputed_path}")

    def set_all_seeds(self, seed=42):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_model_and_tokenizer(self):
        """Load GIT model and tokenizer for real-time inference"""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parameter_yaml_path = os.path.join(project_root, 'aux_data', 'models',
                                            self.model_name, 'parameter.yaml')

        if os.path.isfile(parameter_yaml_path):
            param = load_from_yaml_file(parameter_yaml_path)
        else:
            param = {}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = get_git_model(tokenizer, param)
        checkpoint_path = os.path.join(self.model_path, self.model_name, 'snapshot', 'model.pt')

        checkpoint = torch_load(checkpoint_path)['model']
        load_state_dict(model, checkpoint)
        model.cuda().eval()
        return model, tokenizer

    def get_image_transform(self, param):
        """Get image transformation pipeline"""
        crop_size = param.get('test_crop_size', 224)
        trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        return Compose(trans)

    def load_precomputed_features(self, image_id):
        """
        Load pre-extracted caption features from CSV

        Args:
            image_id: Image filename (with or without extension)

        Returns:
            torch.Tensor: Caption features (1, 1024)
        """
        # Remove extension if present
        base_name = os.path.basename(image_id)
        file_name_without_extension = os.path.splitext(base_name)[0]

        csv_path = os.path.join(self.precomputed_path,
                               f'{file_name_without_extension}_predictions.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Precomputed features not found: {csv_path}\n"
                f"Please run feature extraction first or use real-time mode (use_precomputed=False)"
            )

        df = pd.read_csv(csv_path, header=None)
        features = torch.tensor(df.values, dtype=torch.float32)
        return features

    def test_git_inference_single_image(self, image_tensor, image_ids=None):
        """
        Extract caption features from images

        Args:
            image_tensor: Batch of images (B, C, H, W)
            image_ids: List of image IDs (only needed for precomputed mode)

        Returns:
            predictions: Caption features (B, 1024)
            all_caps: Generated captions (list of strings)
        """
        # Mode 1: Use precomputed features
        if self.use_precomputed:
            if image_ids is None:
                raise ValueError("image_ids required when using precomputed features")

            # Load features for each image in batch
            batch_features = []
            for image_id in image_ids:
                features = self.load_precomputed_features(image_id)
                batch_features.append(features)

            predictions = torch.cat(batch_features, dim=0)

            # No captions in precomputed mode
            all_caps = ["[precomputed features]"] * len(image_ids)

            return predictions, all_caps

        # Mode 2: Real-time inference
        else:
            with torch.no_grad():
                result = self.model({'image': image_tensor})

            # Decode captions
            all_caps = []
            for pred in result['predictions']:
                cap = self.tokenizer.decode(pred.tolist(), skip_special_tokens=True)
                all_caps.append(cap)

            return result['predictions'], all_caps


class MinMaxResizeForTest(object):
    """Resize image while maintaining aspect ratio"""

    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self):
        return f'MinMaxResizeForTest({self.min_size}, {self.max_size})'

    def __call__(self, img):
        size = self.get_size(img.size)
        import torchvision.transforms.functional as F
        image = F.resize(img, size, interpolation=PIL.Image.BICUBIC)
        return image
