# ICAA17K Dataset

## Directory Structure

Place your dataset files in this directory:

```
ICAA17K/
├── images/           # Image files
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── 1train.csv       # Training split
└── 1test.csv        # Test split
```

## CSV Format

The CSV files should contain the following columns:

- **ID**: Image filename (e.g., `image_001.jpg`)
- **MOS**: Mean Opinion Score (0-10 scale)
- **color**: Color harmony score (0-10 scale)

### Example CSV:

```csv
ID,MOS,color
image_001.jpg,7.5,8.2
image_002.jpg,6.3,7.1
image_003.jpg,8.9,9.0
```

## Notes

- All scores should be normalized to a 0-10 scale
- Image files should be in common formats (JPG, PNG, etc.)
- Ensure image filenames in CSV match actual files in the `images/` folder
