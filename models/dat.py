import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pandas as pd
from timm.models.layers import DropPath, to_2tuple
from models.SoftHistogram import SoftHistogram
from .dat_blocks import *
from models.caption_model_loader import CaptionModelLoader

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        # stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = [] 
        references = []
        attn_weight = []
        for d in range(self.depths):
            x0 = x
            # print("x.shape: ", x.shape)
            """
            torch.Size([16, 96, 56, 56])
            torch.Size([16, 192, 28, 28])
            torch.Size([16, 384, 14, 14])
            torch.Size([16, 768, 7, 7])
            """
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))  
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references
    
# 10번 
class FeatureFusionModule(nn.Module):
    def __init__(self, hist_feature_dim, caption_feature_dim, output_dim):
        super(FeatureFusionModule, self).__init__()
        self.fc1 = nn.Linear(hist_feature_dim + caption_feature_dim, output_dim)
        self.gelu1 = nn.GELU()
        # Layer Normalization 추가 위치 변경
        self.ln1 = nn.LayerNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.gelu2 = nn.GELU()
        # Layer Normalization 추가 위치 변경
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, hist_feature, caption_feature):
        combined_features = torch.cat([hist_feature, caption_feature], dim=1)
        x = self.fc1(combined_features)
        x = self.gelu1(x)
        x = self.ln1(x)  # 활성화 함수 후 Layer Normalization 적용
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.ln2(x)  # 활성화 함수 후 Layer Normalization 적용
        return x

class CaptionFeatureReduce(nn.Module):
    def __init__(self):
        super(CaptionFeatureReduce, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.gelu1 = nn.GELU()
        # Layer Normalization 추가 위치 변경
        self.ln1 = nn.LayerNorm(512)
        self.linear2 = nn.Linear(512, 256)
        self.gelu2 = nn.GELU()
        # Layer Normalization 추가 위치 변경
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu1(x)
        x = self.ln1(x)  # 활성화 함수 후 Layer Normalization 적용
        x = self.linear2(x)
        x = self.gelu2(x)
        x = self.ln2(x)  # 활성화 함수 후 Layer Normalization 적용
        return x
    
class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], 
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs):
        super().__init__()

        # DAT 
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.reset_parameters()

        self.hst_head = nn.Linear(1024, 36)
        self.hist_feature = SoftHistogram(n_features=36, n_examples=6, num_bins=6, quantiles=False)
        # 10 
        self.caption_feature_reduce = CaptionFeatureReduce()
        self.feature_fusion_module = FeatureFusionModule(hist_feature_dim=216, caption_feature_dim=256, output_dim=256)
        self.class_head = nn.Linear(256, num_classes)
        self.class_head2 = nn.Linear(256, num_classes)

        self.sg1 = nn.Sigmoid()
        self.sg2 = nn.Sigmoid()

        # Initialize Caption Model Loader
        # Set use_precomputed=True and provide precomputed_path for faster training
        # Otherwise, it will use real-time GIT inference (slower but no preprocessing needed)
        self.caption_model_loader = CaptionModelLoader(
            use_precomputed=False,  # Set to True to use precomputed features
            # precomputed_path='path/to/precomputed/features'  # Uncomment and set path if using precomputed
        )

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    
    # 01, 02 ,03
    def forward(self, x, image_ids):

        # # 직접 모델 불러오는 버전
        caption_features, all_caps = self.caption_model_loader.test_git_inference_single_image(x)
        print("caption_features", caption_features.shape) # [batch, 1024]

        # LARGE R MODEL 정규화 로직에 사용된 Min, Max 값
        global_min_val = 101
        global_max_val = 29561
        # Normalize caption features (using pre-calculated min/max from GIT_LARGE_R_TEXTCAPS)
        caption_features = (caption_features - global_min_val) / (global_max_val - global_min_val)

        # Reduce caption feature dimensions: 1024 -> 256
        caption_features_reduced = self.caption_feature_reduce(caption_features)


        x = self.patch_proj(x)
        positions = []
        references = []

        for i in range(4):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)

        x = self.cls_norm(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)

        # histogram
        x = self.hst_head(x) 
        x = self.hist_feature(x) 
        x = einops.rearrange(x, 'b p w -> w p b') #(w: input이미지 수)
        x = x.squeeze(dim = 2)

        # x 와 caption_feature 결합
        combined_features = self.feature_fusion_module(x, caption_features_reduced)

        x1 = self.class_head(combined_features)
        x1 = self.sg1(x1)
        x2 = self.class_head2(combined_features)
        x2 = self.sg2(x2)
        x = torch.cat([x1, x2], 1)
        print("all_caps", all_caps)
        return x, positions, references, all_caps
        # return x, positions, references
    
    def get_last_shared_layer(self):
        return self.hst_head 

