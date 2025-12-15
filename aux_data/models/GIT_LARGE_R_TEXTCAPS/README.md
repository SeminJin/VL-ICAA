# GIT Model Files

## Auto-Download

The GIT (Generative Image-to-Text) model will be automatically downloaded via `azfuse` when you first run the training script.

The model files will be placed in:
```
output/GIT_LARGE_R_TEXTCAPS/snapshot/model.pt
```

## Manual Download (if auto-download fails)

If automatic download doesn't work, you can manually download:

1. Install azfuse:
```bash
pip install git+https://github.com/microsoft/azfuse.git
```

2. Set environment variable:
```bash
export AZFUSE_TSV_USE_FUSE=1
```

3. Run a test inference to trigger download:
```bash
cd generativeimage2text
python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', 'image_path': 'aux_data/images/test.jpg', 'model_name': 'GIT_LARGE_R_TEXTCAPS', 'prefix': ''}"
```

## Model Files

After download, you should have:
```
GIT_LARGE_R_TEXTCAPS/
├── parameter.yaml       # Model configuration (already included)
└── (model checkpoint will be in output/GIT_LARGE_R_TEXTCAPS/snapshot/)
```

## Alternative Models

You can also use other GIT models by changing the model name in `models/caption_model_loader.py`:

- `GIT_BASE_TEXTCAPS` (smaller, faster)
- `GIT_LARGE_COCO`
- `GIT_BASE_COCO`

## Reference

- [GenerativeImage2Text Repository](https://github.com/microsoft/GenerativeImage2Text)
- [GIT Paper](https://arxiv.org/abs/2205.14100)
