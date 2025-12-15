# Precomputed Caption Features

This directory contains pre-extracted caption features from the GIT (Generative Image-to-Text) model.

## Why Use Precomputed Features?

Using precomputed features significantly speeds up training:
- **Real-time mode**: ~2-3 hours per epoch
- **Precomputed mode**: ~30-40 minutes per epoch

## Directory Structure

```
precomputed_features/
└── GIT_LARGE_R_TEXTCAPS/
    └── normalized_result/
        ├── image_001_predictions.csv
        ├── image_002_predictions.csv
        └── ...
```

## Feature Format

Each CSV file contains:
- **Dimensions**: (1, 1024) - one row with 1024 features
- **Normalization**: Features are normalized using global min/max values
  - Min: 101
  - Max: 29561
- **Format**: Comma-separated values, no header

## How to Use

### Option 1: Use Provided Features (if available)

If precomputed features are provided separately:

```bash
# Extract the features archive
tar -xzf precomputed_features.tar.gz -C precomputed_features/

# Or copy from another location
cp -r /path/to/normalized_result precomputed_features/GIT_LARGE_R_TEXTCAPS/
```

### Option 2: Extract Your Own Features

If you need to extract features for a new dataset:

1. **Ensure GIT model is downloaded** (see main README.md)

2. **Create extraction script** (example):
```python
import os
import pandas as pd
import torch
from models.caption_model_loader import CaptionModelLoader
from dataset import ICAA17KDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = ICAA17KDataset('dataset/ICAA17K/1train.csv',
                          'dataset/ICAA17K/images',
                          if_train=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load GIT model (real-time mode)
caption_loader = CaptionModelLoader(use_precomputed=False)

# Extract features
output_dir = 'precomputed_features/GIT_LARGE_R_TEXTCAPS/normalized_result'
os.makedirs(output_dir, exist_ok=True)

for x, image_id, y in loader:
    x = x.cuda()
    features, _ = caption_loader.test_git_inference_single_image(x)

    # Normalize features
    global_min_val = 101
    global_max_val = 29561
    features = (features - global_min_val) / (global_max_val - global_min_val)

    # Save to CSV
    image_name = os.path.splitext(image_id[0])[0]
    csv_path = os.path.join(output_dir, f'{image_name}_predictions.csv')
    pd.DataFrame(features.cpu().numpy()).to_csv(csv_path,
                                                  index=False,
                                                  header=False)
    print(f'Saved: {csv_path}')
```

3. **Run extraction**:
```bash
python extract_features.py
```

## Enabling Precomputed Mode

Once features are ready, enable precomputed mode in `models/dat.py`:

```python
self.caption_model_loader = CaptionModelLoader(
    use_precomputed=True,
    precomputed_path='precomputed_features/GIT_LARGE_R_TEXTCAPS/normalized_result'
)
```

## Sharing Features

If you want to share your precomputed features:

```bash
# Create archive
cd precomputed_features
tar -czf ../precomputed_features_ICAA17K.tar.gz GIT_LARGE_R_TEXTCAPS/

# The archive can be shared separately from the code repository
```

## Notes

- Precomputed features are **dataset-specific** - each dataset needs its own features
- Features should match the training/test split exactly
- File naming must match: `{image_id}_predictions.csv`
- The `.gitignore` excludes these files from version control due to size
