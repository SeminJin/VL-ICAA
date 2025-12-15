# Pretrained Weights

## Download DAT Pretrained Weights

Download the DAT base model pretrained on ImageNet-1K:

```bash
# DAT Base (recommended)
wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_base_in1k_224.pth \
     -O dat_base_checkpoint.pth
```

## Alternative DAT Models

### DAT-Tiny (Faster, less accurate)
```bash
wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_tiny_in1k_224.pth \
     -O dat_tiny_checkpoint.pth
```

### DAT-Small (Balanced)
```bash
wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_small_in1k_224.pth \
     -O dat_small_checkpoint.pth
```

### DAT-Base 384 (Higher resolution)
```bash
wget https://github.com/LeapLabTHU/DAT/releases/download/v1.0/dat_base_in1k_384.pth \
     -O dat_base_384_checkpoint.pth
```

## After Download

Update `option.py` if you use a different model:

```python
parser.add_argument('--path_to_model_weight', type=str,
                    default='pretrained_weights/your_model_name.pth')
```

## Reference

- [DAT GitHub Repository](https://github.com/LeapLabTHU/DAT)
- [DAT Paper](https://arxiv.org/abs/2201.00520)
