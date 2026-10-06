/content/shoes_vto
================================================================================
ARShoe M1 Training
================================================================================

📁 Loading datasets...
Loaded 643 images from /content/dataset/shuffled_v3/train/images
Loaded 79 images from /content/dataset/shuffled_v3/valid/images
  Train: 643 images
  Val: 79 images

🏗️  Creating model...
  Total parameters: 197,584
  Encoder: 24,096
  Heatmap Head: 74,896
  PAF Head: 75,806
  Class Head: 22,786
  Budget: 15.2% of 1.3M

🎯 Initializing trainer...
  [GPU] Enabled TF32 & cuDNN benchmark for A100 maximum throughput!
  [GPU] Model compiled with torch.compile() for A100 maximum speed!
/content/shoes_vto/src/training/trainer_m1.py:145: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

🚀 Starting training...

================================================================================
Training ARShoe M1 for 200 epochs
Device: cuda
Mixed Precision (AMP): Enabled
Train samples: 643
Val samples: 79
Batch size: 16
Learning rate: 0.001
Loss weights: {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
================================================================================
Epoch 1:   0% 0/41 [00:00<?, ?it/s]/content/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1:   2% 1/41 [00:21<14:09, 21.24s/it, loss=11.6991, hm=2.5070, paf=0.3119, cls=0.6983]/content/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1: 100% 41/41 [00:45<00:00,  1.10s/it, loss=4.7597, hm=0.7804, paf=0.2998, cls=0.6924]

Epoch 1 - Train: 7.3508 (HM: 1.4428, PAF: 0.2650, Cls: 0.6997)  LR: 0.001000
Epoch 2: 100% 41/41 [00:06<00:00,  6.37it/s, loss=3.3718, hm=0.4594, paf=0.2351, cls=0.7094]
                                             
Epoch 2 Summary:
  Train - Total: 3.7251, HM: 0.5498, PAF: 0.2421, Cls: 0.6945
  Val   - Total: 3.4014, HM: 0.4633, PAF: 0.2368, Cls: 0.7165
  Metrics - Class Acc: 39.68%, PCK@0.2: 31.02%
  LR: 0.001000
  💾 Saved best model (val_loss: 3.4014)
Epoch 3: 100% 41/41 [00:06<00:00,  6.53it/s, loss=3.2825, hm=0.4580, paf=0.2026, cls=0.6969]

Epoch 3 - Train: 3.1949 (HM: 0.4219, PAF: 0.2325, Cls: 0.6947)  LR: 0.000999
Epoch 4: 100% 41/41 [00:06<00:00,  6.56it/s, loss=2.8703, hm=0.3381, paf=0.2294, cls=0.7061]
                                             
Epoch 4 Summary:
  Train - Total: 3.0795, HM: 0.3952, PAF: 0.2286, Cls: 0.6942
  Val   - Total: 3.1641, HM: 0.4130, PAF: 0.2320, Cls: 0.6988
  Metrics - Class Acc: 42.66%, PCK@0.2: 32.39%
  LR: 0.000999
  💾 Saved best model (val_loss: 3.1641)
Epoch 5: 100% 41/41 [00:06<00:00,  6.67it/s, loss=2.9127, hm=0.3618, paf=0.2086, cls=0.6990]

Epoch 5 - Train: 3.0069 (HM: 0.3794, PAF: 0.2250, Cls: 0.6929)  LR: 0.000998
Epoch 6: 100% 41/41 [00:06<00:00,  6.75it/s, loss=2.5914, hm=0.2907, paf=0.1939, cls=0.6938]
                                             
Epoch 6 Summary:
  Train - Total: 2.9625, HM: 0.3696, PAF: 0.2216, Cls: 0.6938
  Val   - Total: 3.0208, HM: 0.3932, PAF: 0.2102, Cls: 0.6851
  Metrics - Class Acc: 57.71%, PCK@0.2: 34.53%
  LR: 0.000998
  💾 Saved best model (val_loss: 3.0208)
Epoch 7: 100% 41/41 [00:06<00:00,  6.78it/s, loss=3.4002, hm=0.4739, paf=0.2384, cls=0.6852]

Epoch 7 - Train: 2.9368 (HM: 0.3660, PAF: 0.2173, Cls: 0.6921)  LR: 0.000997
Epoch 8: 100% 41/41 [00:06<00:00,  6.67it/s, loss=3.2228, hm=0.4234, paf=0.2495, cls=0.6868]
                                             
Epoch 8 Summary:
  Train - Total: 2.9198, HM: 0.3616, PAF: 0.2164, Cls: 0.6939
  Val   - Total: 2.8950, HM: 0.3639, PAF: 0.2017, Cls: 0.6906
  Metrics - Class Acc: 54.10%, PCK@0.2: 38.66%
  LR: 0.000996
  💾 Saved best model (val_loss: 2.8950)
Epoch 9: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.9123, hm=0.3568, paf=0.2270, cls=0.6873]

Epoch 9 - Train: 2.8904 (HM: 0.3548, PAF: 0.2144, Cls: 0.6948)  LR: 0.000995
Epoch 10: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.9235, hm=0.3554, paf=0.2228, cls=0.7043]
                                             
Epoch 10 Summary:
  Train - Total: 2.8667, HM: 0.3518, PAF: 0.2110, Cls: 0.6916
  Val   - Total: 2.9730, HM: 0.3713, PAF: 0.2165, Cls: 0.7030
  Metrics - Class Acc: 41.83%, PCK@0.2: 38.35%
  LR: 0.000994
Epoch 11: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.9220, hm=0.3496, paf=0.2449, cls=0.6891]

Epoch 11 - Train: 2.8438 (HM: 0.3464, PAF: 0.2096, Cls: 0.6927)  LR: 0.000993
Epoch 12: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.7037, hm=0.2837, paf=0.2667, cls=0.6903]
                                             
Epoch 12 Summary:
  Train - Total: 2.8360, HM: 0.3445, PAF: 0.2086, Cls: 0.6940
  Val   - Total: 3.0718, HM: 0.3984, PAF: 0.2178, Cls: 0.6950
  Metrics - Class Acc: 50.73%, PCK@0.2: 45.45%
  LR: 0.000991
Epoch 13: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.8807, hm=0.3544, paf=0.2155, cls=0.6882]

Epoch 13 - Train: 2.8072 (HM: 0.3396, PAF: 0.2052, Cls: 0.6922)  LR: 0.000990
Epoch 14: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.8254, hm=0.3313, paf=0.2360, cls=0.6854]
                                             
Epoch 14 Summary:
  Train - Total: 2.7964, HM: 0.3385, PAF: 0.2027, Cls: 0.6912
  Val   - Total: 2.8244, HM: 0.3446, PAF: 0.2036, Cls: 0.6926
  Metrics - Class Acc: 49.74%, PCK@0.2: 42.70%
  LR: 0.000988
  💾 Saved best model (val_loss: 2.8244)
Epoch 15: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.7647, hm=0.3324, paf=0.1968, cls=0.6943]

Epoch 15 - Train: 2.7829 (HM: 0.3357, PAF: 0.2012, Cls: 0.6917)  LR: 0.000986
Epoch 16: 100% 41/41 [00:06<00:00,  6.70it/s, loss=2.9647, hm=0.3690, paf=0.2210, cls=0.6978]
                                             
Epoch 16 Summary:
  Train - Total: 2.7888, HM: 0.3376, PAF: 0.2013, Cls: 0.6905
  Val   - Total: 2.9867, HM: 0.3898, PAF: 0.1980, Cls: 0.6877
  Metrics - Class Acc: 54.20%, PCK@0.2: 47.44%
  LR: 0.000984
Epoch 17: 100% 41/41 [00:05<00:00,  6.87it/s, loss=2.7460, hm=0.3379, paf=0.1663, cls=0.7077]

Epoch 17 - Train: 2.7720 (HM: 0.3348, PAF: 0.1966, Cls: 0.6930)  LR: 0.000982
Epoch 18: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.7326, hm=0.3322, paf=0.1902, cls=0.6823]
                                             
Epoch 18 Summary:
  Train - Total: 2.7489, HM: 0.3315, PAF: 0.1929, Cls: 0.6916
  Val   - Total: 2.7843, HM: 0.3425, PAF: 0.1842, Cls: 0.6974
  Metrics - Class Acc: 42.48%, PCK@0.2: 45.45%
  LR: 0.000980
  💾 Saved best model (val_loss: 2.7843)
Epoch 19: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.7252, hm=0.3385, paf=0.1897, cls=0.6614]

Epoch 19 - Train: 2.7375 (HM: 0.3287, PAF: 0.1935, Cls: 0.6903)  LR: 0.000978
Epoch 20: 100% 41/41 [00:06<00:00,  6.74it/s, loss=3.0174, hm=0.3440, paf=0.2609, cls=0.7464]
                                             
Epoch 20 Summary:
  Train - Total: 2.7214, HM: 0.3260, PAF: 0.1898, Cls: 0.6920
  Val   - Total: 2.9055, HM: 0.3771, PAF: 0.1793, Cls: 0.6922
  Metrics - Class Acc: 50.70%, PCK@0.2: 49.81%
  LR: 0.000976
Epoch 21: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.9109, hm=0.3725, paf=0.1823, cls=0.7043]

Epoch 21 - Train: 2.7206 (HM: 0.3264, PAF: 0.1898, Cls: 0.6903)  LR: 0.000973
Epoch 22: 100% 41/41 [00:06<00:00,  6.72it/s, loss=2.6628, hm=0.3047, paf=0.1982, cls=0.6984]
                                             
Epoch 22 Summary:
  Train - Total: 2.7110, HM: 0.3251, PAF: 0.1892, Cls: 0.6882
  Val   - Total: 2.7794, HM: 0.3438, PAF: 0.1854, Cls: 0.6888
  Metrics - Class Acc: 52.37%, PCK@0.2: 47.59%
  LR: 0.000971
  💾 Saved best model (val_loss: 2.7794)
Epoch 23: 100% 41/41 [00:05<00:00,  6.91it/s, loss=2.8640, hm=0.3441, paf=0.2566, cls=0.6496]

Epoch 23 - Train: 2.7007 (HM: 0.3230, PAF: 0.1891, Cls: 0.6869)  LR: 0.000968
Epoch 24: 100% 41/41 [00:06<00:00,  6.75it/s, loss=2.7526, hm=0.3313, paf=0.2032, cls=0.6806]
                                             
Epoch 24 Summary:
  Train - Total: 2.6850, HM: 0.3209, PAF: 0.1867, Cls: 0.6853
  Val   - Total: 2.7839, HM: 0.3311, PAF: 0.1974, Cls: 0.7098
  Metrics - Class Acc: 44.31%, PCK@0.2: 51.87%
  LR: 0.000965
Epoch 25: 100% 41/41 [00:05<00:00,  6.90it/s, loss=2.8093, hm=0.3594, paf=0.1870, cls=0.6652]

Epoch 25 - Train: 2.6765 (HM: 0.3216, PAF: 0.1829, Cls: 0.6829)  LR: 0.000962
Epoch 26: 100% 41/41 [00:06<00:00,  6.81it/s, loss=3.0704, hm=0.3495, paf=0.2988, cls=0.7164]
                                             
Epoch 26 Summary:
  Train - Total: 2.6715, HM: 0.3179, PAF: 0.1848, Cls: 0.6869
  Val   - Total: 2.7676, HM: 0.3347, PAF: 0.1956, Cls: 0.6916
  Metrics - Class Acc: 47.26%, PCK@0.2: 48.43%
  LR: 0.000959
  💾 Saved best model (val_loss: 2.7676)
Epoch 27: 100% 41/41 [00:05<00:00,  6.93it/s, loss=2.7332, hm=0.3428, paf=0.1407, cls=0.7203]

Epoch 27 - Train: 2.6658 (HM: 0.3183, PAF: 0.1832, Cls: 0.6842)  LR: 0.000956
Epoch 28: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.3528, hm=0.2568, paf=0.1571, cls=0.6743]
                                             
Epoch 28 Summary:
  Train - Total: 2.6567, HM: 0.3170, PAF: 0.1829, Cls: 0.6820
  Val   - Total: 2.7616, HM: 0.3385, PAF: 0.1835, Cls: 0.6937
  Metrics - Class Acc: 49.54%, PCK@0.2: 53.09%
  LR: 0.000953
  💾 Saved best model (val_loss: 2.7616)
Epoch 29: 100% 41/41 [00:05<00:00,  6.87it/s, loss=2.6224, hm=0.3261, paf=0.1646, cls=0.6592]

Epoch 29 - Train: 2.6423 (HM: 0.3148, PAF: 0.1810, Cls: 0.6808)  LR: 0.000950
Epoch 30: 100% 41/41 [00:06<00:00,  6.72it/s, loss=2.9958, hm=0.3924, paf=0.1987, cls=0.6859]
                                             
Epoch 30 Summary:
  Train - Total: 2.6449, HM: 0.3156, PAF: 0.1796, Cls: 0.6823
  Val   - Total: 2.6928, HM: 0.3253, PAF: 0.1773, Cls: 0.6914
  Metrics - Class Acc: 51.96%, PCK@0.2: 55.16%
  LR: 0.000946
  💾 Saved best model (val_loss: 2.6928)
Epoch 31: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.8223, hm=0.3515, paf=0.2049, cls=0.6711]

Epoch 31 - Train: 2.6193 (HM: 0.3127, PAF: 0.1768, Cls: 0.6765)  LR: 0.000942
Epoch 32: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.6753, hm=0.2932, paf=0.2003, cls=0.7346]
                                             
Epoch 32 Summary:
  Train - Total: 2.6149, HM: 0.3120, PAF: 0.1758, Cls: 0.6769
  Val   - Total: 2.7625, HM: 0.3374, PAF: 0.1791, Cls: 0.7031
  Metrics - Class Acc: 48.81%, PCK@0.2: 52.86%
  LR: 0.000939
Epoch 33: 100% 41/41 [00:05<00:00,  6.90it/s, loss=2.6575, hm=0.3232, paf=0.1766, cls=0.6745]

Epoch 33 - Train: 2.6212 (HM: 0.3132, PAF: 0.1766, Cls: 0.6769)  LR: 0.000935
Epoch 34: 100% 41/41 [00:06<00:00,  6.72it/s, loss=2.8142, hm=0.3375, paf=0.2337, cls=0.6647]
                                             
Epoch 34 Summary:
  Train - Total: 2.6074, HM: 0.3110, PAF: 0.1766, Cls: 0.6733
  Val   - Total: 2.7048, HM: 0.3294, PAF: 0.1835, Cls: 0.6800
  Metrics - Class Acc: 54.90%, PCK@0.2: 51.03%
  LR: 0.000931
Epoch 35: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.2906, hm=0.2766, paf=0.1572, cls=0.5800]

Epoch 35 - Train: 2.5797 (HM: 0.3074, PAF: 0.1721, Cls: 0.6707)  LR: 0.000927
Epoch 36: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.6203, hm=0.3231, paf=0.1443, cls=0.6929]
                                             
Epoch 36 Summary:
  Train - Total: 2.5777, HM: 0.3069, PAF: 0.1736, Cls: 0.6687
  Val   - Total: 2.8030, HM: 0.3456, PAF: 0.1863, Cls: 0.6987
  Metrics - Class Acc: 52.95%, PCK@0.2: 51.64%
  LR: 0.000923
Epoch 37: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.5095, hm=0.2676, paf=0.2087, cls=0.6812]

Epoch 37 - Train: 2.5823 (HM: 0.3077, PAF: 0.1743, Cls: 0.6688)  LR: 0.000919
Epoch 38: 100% 41/41 [00:06<00:00,  6.69it/s, loss=2.5221, hm=0.2898, paf=0.1570, cls=0.6993]
                                             
Epoch 38 Summary:
  Train - Total: 2.5749, HM: 0.3077, PAF: 0.1715, Cls: 0.6675
  Val   - Total: 2.5555, HM: 0.3120, PAF: 0.1619, Cls: 0.6556
  Metrics - Class Acc: 61.59%, PCK@0.2: 58.21%
  LR: 0.000914
  💾 Saved best model (val_loss: 2.5555)
Epoch 39: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.6987, hm=0.3323, paf=0.1688, cls=0.6880]

Epoch 39 - Train: 2.5712 (HM: 0.3082, PAF: 0.1689, Cls: 0.6670)  LR: 0.000910
Epoch 40: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.5185, hm=0.2710, paf=0.1940, cls=0.6977]
                                             
Epoch 40 Summary:
  Train - Total: 2.5617, HM: 0.3067, PAF: 0.1709, Cls: 0.6622
  Val   - Total: 2.6020, HM: 0.3110, PAF: 0.1726, Cls: 0.6753
  Metrics - Class Acc: 56.55%, PCK@0.2: 56.07%
  LR: 0.000905
Epoch 41: 100% 41/41 [00:05<00:00,  6.88it/s, loss=2.6781, hm=0.3185, paf=0.1760, cls=0.7015]

Epoch 41 - Train: 2.5471 (HM: 0.3040, PAF: 0.1687, Cls: 0.6624)  LR: 0.000901
Epoch 42: 100% 41/41 [00:06<00:00,  6.70it/s, loss=2.2846, hm=0.2886, paf=0.1273, cls=0.5838]
                                             
Epoch 42 Summary:
  Train - Total: 2.5357, HM: 0.3030, PAF: 0.1679, Cls: 0.6586
  Val   - Total: 2.5857, HM: 0.3184, PAF: 0.1610, Cls: 0.6602
  Metrics - Class Acc: 58.03%, PCK@0.2: 60.28%
  LR: 0.000896
Epoch 43: 100% 41/41 [00:05<00:00,  6.87it/s, loss=2.6795, hm=0.3239, paf=0.1704, cls=0.6954]

Epoch 43 - Train: 2.5430 (HM: 0.3043, PAF: 0.1688, Cls: 0.6587)  LR: 0.000891
Epoch 44: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.8207, hm=0.3954, paf=0.1659, cls=0.6049]
                                             
Epoch 44 Summary:
  Train - Total: 2.5282, HM: 0.3043, PAF: 0.1651, Cls: 0.6539
  Val   - Total: 2.6646, HM: 0.3198, PAF: 0.1780, Cls: 0.6863
  Metrics - Class Acc: 53.92%, PCK@0.2: 55.31%
  LR: 0.000886
Epoch 45: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.3679, hm=0.2467, paf=0.1915, cls=0.6653]

Epoch 45 - Train: 2.5042 (HM: 0.2980, PAF: 0.1662, Cls: 0.6530)  LR: 0.000881
Epoch 46: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.3285, hm=0.2912, paf=0.1753, cls=0.5420]
                                             
Epoch 46 Summary:
  Train - Total: 2.4985, HM: 0.2988, PAF: 0.1643, Cls: 0.6499
  Val   - Total: 2.6940, HM: 0.3215, PAF: 0.1864, Cls: 0.6901
  Metrics - Class Acc: 53.53%, PCK@0.2: 55.69%
  LR: 0.000876
Epoch 47: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.4540, hm=0.2639, paf=0.1853, cls=0.6851]

Epoch 47 - Train: 2.5038 (HM: 0.2986, PAF: 0.1652, Cls: 0.6528)  LR: 0.000871
Epoch 48: 100% 41/41 [00:06<00:00,  6.70it/s, loss=2.3646, hm=0.2705, paf=0.1503, cls=0.6547]
                                             
Epoch 48 Summary:
  Train - Total: 2.5061, HM: 0.3016, PAF: 0.1637, Cls: 0.6482
  Val   - Total: 2.7229, HM: 0.3339, PAF: 0.1992, Cls: 0.6594
  Metrics - Class Acc: 59.98%, PCK@0.2: 60.96%
  LR: 0.000866
Epoch 49: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.2752, hm=0.2131, paf=0.1671, cls=0.7256]

Epoch 49 - Train: 2.4879 (HM: 0.2978, PAF: 0.1636, Cls: 0.6464)  LR: 0.000860
Epoch 50: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.6135, hm=0.2692, paf=0.1547, cls=0.8182]
                                             
Epoch 50 Summary:
  Train - Total: 2.4900, HM: 0.2980, PAF: 0.1633, Cls: 0.6475
  Val   - Total: 2.5942, HM: 0.3064, PAF: 0.1731, Cls: 0.6816
  Metrics - Class Acc: 54.79%, PCK@0.2: 58.75%
  LR: 0.000855
Epoch 51: 100% 41/41 [00:05<00:00,  6.87it/s, loss=2.6952, hm=0.3012, paf=0.1921, cls=0.7373]

Epoch 51 - Train: 2.4792 (HM: 0.2984, PAF: 0.1616, Cls: 0.6416)  LR: 0.000849
Epoch 52: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.4222, hm=0.3187, paf=0.1423, cls=0.5753]
                                             
Epoch 52 Summary:
  Train - Total: 2.4770, HM: 0.2975, PAF: 0.1630, Cls: 0.6406
  Val   - Total: 2.5239, HM: 0.3132, PAF: 0.1542, Cls: 0.6419
  Metrics - Class Acc: 63.05%, PCK@0.2: 59.97%
  LR: 0.000844
  💾 Saved best model (val_loss: 2.5239)
Epoch 53: 100% 41/41 [00:05<00:00,  6.94it/s, loss=2.7346, hm=0.3095, paf=0.2027, cls=0.7273]

Epoch 53 - Train: 2.4659 (HM: 0.2977, PAF: 0.1606, Cls: 0.6358)  LR: 0.000838
Epoch 54: 100% 41/41 [00:05<00:00,  6.89it/s, loss=2.6445, hm=0.3565, paf=0.1288, cls=0.6407]
                                             
Epoch 54 Summary:
  Train - Total: 2.4745, HM: 0.2970, PAF: 0.1608, Cls: 0.6431
  Val   - Total: 2.6462, HM: 0.3120, PAF: 0.1734, Cls: 0.7009
  Metrics - Class Acc: 53.49%, PCK@0.2: 60.12%
  LR: 0.000832
Epoch 55: 100% 41/41 [00:05<00:00,  6.89it/s, loss=2.7332, hm=0.3581, paf=0.1447, cls=0.6743]

Epoch 55 - Train: 2.4503 (HM: 0.2954, PAF: 0.1586, Cls: 0.6343)  LR: 0.000826
Epoch 56: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.6731, hm=0.3236, paf=0.1888, cls=0.6674]
                                             
Epoch 56 Summary:
  Train - Total: 2.4455, HM: 0.2951, PAF: 0.1578, Cls: 0.6330
  Val   - Total: 2.6156, HM: 0.3074, PAF: 0.1709, Cls: 0.6961
  Metrics - Class Acc: 54.00%, PCK@0.2: 57.75%
  LR: 0.000821
Epoch 57: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.7371, hm=0.3325, paf=0.2045, cls=0.6654]

Epoch 57 - Train: 2.4505 (HM: 0.2959, PAF: 0.1595, Cls: 0.6320)  LR: 0.000814
Epoch 58: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.8566, hm=0.3660, paf=0.1955, cls=0.6676]
                                             
Epoch 58 Summary:
  Train - Total: 2.4381, HM: 0.2939, PAF: 0.1580, Cls: 0.6310
  Val   - Total: 2.7232, HM: 0.3332, PAF: 0.1818, Cls: 0.6844
  Metrics - Class Acc: 57.82%, PCK@0.2: 58.67%
  LR: 0.000808
Epoch 59: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.5596, hm=0.2968, paf=0.2321, cls=0.6054]

Epoch 59 - Train: 2.4229 (HM: 0.2929, PAF: 0.1595, Cls: 0.6215)  LR: 0.000802
Epoch 60: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.6167, hm=0.3287, paf=0.1606, cls=0.6539]
                                             
Epoch 60 Summary:
  Train - Total: 2.4221, HM: 0.2922, PAF: 0.1592, Cls: 0.6234
  Val   - Total: 2.5528, HM: 0.3040, PAF: 0.1760, Cls: 0.6565
  Metrics - Class Acc: 60.56%, PCK@0.2: 62.49%
  LR: 0.000796
Epoch 61: 100% 41/41 [00:05<00:00,  6.90it/s, loss=2.5166, hm=0.2733, paf=0.2069, cls=0.6731]

Epoch 61 - Train: 2.4317 (HM: 0.2942, PAF: 0.1588, Cls: 0.6249)  LR: 0.000790
Epoch 62: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.7256, hm=0.3306, paf=0.1651, cls=0.7153]
                                             
Epoch 62 Summary:
  Train - Total: 2.4263, HM: 0.2931, PAF: 0.1584, Cls: 0.6249
  Val   - Total: 2.5666, HM: 0.3180, PAF: 0.1664, Cls: 0.6412
  Metrics - Class Acc: 63.30%, PCK@0.2: 61.50%
  LR: 0.000783
Epoch 63: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.5658, hm=0.3035, paf=0.1530, cls=0.6973]

Epoch 63 - Train: 2.4092 (HM: 0.2928, PAF: 0.1573, Cls: 0.6157)  LR: 0.000777
Epoch 64: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.7669, hm=0.3532, paf=0.1569, cls=0.6937]
                                             
Epoch 64 Summary:
  Train - Total: 2.4295, HM: 0.2950, PAF: 0.1572, Cls: 0.6235
  Val   - Total: 2.7927, HM: 0.3433, PAF: 0.1852, Cls: 0.6993
  Metrics - Class Acc: 54.71%, PCK@0.2: 52.02%
  LR: 0.000770
Epoch 65: 100% 41/41 [00:05<00:00,  6.91it/s, loss=2.4226, hm=0.2812, paf=0.2071, cls=0.5892]

Epoch 65 - Train: 2.4147 (HM: 0.2921, PAF: 0.1577, Cls: 0.6207)  LR: 0.000764
Epoch 66: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.3601, hm=0.2555, paf=0.1481, cls=0.6946]
                                             
Epoch 66 Summary:
  Train - Total: 2.3794, HM: 0.2880, PAF: 0.1545, Cls: 0.6122
  Val   - Total: 2.6006, HM: 0.3108, PAF: 0.1788, Cls: 0.6665
  Metrics - Class Acc: 59.16%, PCK@0.2: 58.82%
  LR: 0.000757
Epoch 67: 100% 41/41 [00:05<00:00,  6.87it/s, loss=2.5475, hm=0.2716, paf=0.1935, cls=0.7162]

Epoch 67 - Train: 2.3916 (HM: 0.2903, PAF: 0.1558, Cls: 0.6126)  LR: 0.000750
Epoch 68: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.5242, hm=0.2672, paf=0.2151, cls=0.6835]
                                             
Epoch 68 Summary:
  Train - Total: 2.4072, HM: 0.2918, PAF: 0.1559, Cls: 0.6189
  Val   - Total: 2.5304, HM: 0.3050, PAF: 0.1695, Cls: 0.6475
  Metrics - Class Acc: 61.32%, PCK@0.2: 59.59%
  LR: 0.000743
Epoch 69: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.5320, hm=0.3085, paf=0.1834, cls=0.6208]

Epoch 69 - Train: 2.3969 (HM: 0.2904, PAF: 0.1563, Cls: 0.6152)  LR: 0.000737
Epoch 70: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.1976, hm=0.2682, paf=0.1714, cls=0.5213]
                                             
Epoch 70 Summary:
  Train - Total: 2.3940, HM: 0.2931, PAF: 0.1557, Cls: 0.6067
  Val   - Total: 2.5493, HM: 0.3171, PAF: 0.1633, Cls: 0.6362
  Metrics - Class Acc: 62.64%, PCK@0.2: 56.91%
  LR: 0.000730
Epoch 71: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.4615, hm=0.2374, paf=0.1509, cls=0.8067]

Epoch 71 - Train: 2.3759 (HM: 0.2880, PAF: 0.1538, Cls: 0.6109)  LR: 0.000723
Epoch 72: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.7713, hm=0.3471, paf=0.1763, cls=0.6869]
                                             
Epoch 72 Summary:
  Train - Total: 2.3649, HM: 0.2879, PAF: 0.1527, Cls: 0.6053
  Val   - Total: 2.5365, HM: 0.3007, PAF: 0.1633, Cls: 0.6714
  Metrics - Class Acc: 58.84%, PCK@0.2: 61.65%
  LR: 0.000716
Epoch 73: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.2846, hm=0.2572, paf=0.1559, cls=0.6293]

Epoch 73 - Train: 2.3658 (HM: 0.2890, PAF: 0.1507, Cls: 0.6054)  LR: 0.000709
Epoch 74: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.4369, hm=0.2506, paf=0.2033, cls=0.6851]
                                             
Epoch 74 Summary:
  Train - Total: 2.3688, HM: 0.2889, PAF: 0.1531, Cls: 0.6047
  Val   - Total: 2.5227, HM: 0.3016, PAF: 0.1604, Cls: 0.6636
  Metrics - Class Acc: 59.75%, PCK@0.2: 62.57%
  LR: 0.000702
  💾 Saved best model (val_loss: 2.5227)
Epoch 75: 100% 41/41 [00:05<00:00,  6.91it/s, loss=2.3783, hm=0.2891, paf=0.1482, cls=0.6170]

Epoch 75 - Train: 2.3661 (HM: 0.2884, PAF: 0.1523, Cls: 0.6053)  LR: 0.000694
Epoch 76: 100% 41/41 [00:06<00:00,  6.60it/s, loss=2.2132, hm=0.2694, paf=0.1504, cls=0.5566]
                                             
Epoch 76 Summary:
  Train - Total: 2.3466, HM: 0.2867, PAF: 0.1501, Cls: 0.5998
  Val   - Total: 2.4105, HM: 0.2954, PAF: 0.1494, Cls: 0.6202
  Metrics - Class Acc: 65.20%, PCK@0.2: 62.18%
  LR: 0.000687
  💾 Saved best model (val_loss: 2.4105)
Epoch 77: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.0926, hm=0.2139, paf=0.1622, cls=0.6085]

Epoch 77 - Train: 2.3415 (HM: 0.2842, PAF: 0.1517, Cls: 0.6007)  LR: 0.000680
Epoch 78: 100% 41/41 [00:06<00:00,  6.82it/s, loss=3.1467, hm=0.4022, paf=0.2542, cls=0.6863]
                                             
Epoch 78 Summary:
  Train - Total: 2.3636, HM: 0.2893, PAF: 0.1524, Cls: 0.6011
  Val   - Total: 2.5262, HM: 0.3103, PAF: 0.1596, Cls: 0.6439
  Metrics - Class Acc: 63.00%, PCK@0.2: 65.24%
  LR: 0.000673
Epoch 79: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.3082, hm=0.2870, paf=0.1361, cls=0.5919]

Epoch 79 - Train: 2.3416 (HM: 0.2858, PAF: 0.1491, Cls: 0.6001)  LR: 0.000665
Epoch 80: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.6946, hm=0.3200, paf=0.2097, cls=0.6633]
                                             
Epoch 80 Summary:
  Train - Total: 2.3209, HM: 0.2854, PAF: 0.1477, Cls: 0.5892
  Val   - Total: 2.4370, HM: 0.2907, PAF: 0.1632, Cls: 0.6320
  Metrics - Class Acc: 64.40%, PCK@0.2: 61.73%
  LR: 0.000658
Epoch 81: 100% 41/41 [00:05<00:00,  6.90it/s, loss=3.2284, hm=0.4622, paf=0.1768, cls=0.6840]

Epoch 81 - Train: 2.3478 (HM: 0.2878, PAF: 0.1498, Cls: 0.5980)  LR: 0.000651
Epoch 82: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.5498, hm=0.3343, paf=0.1357, cls=0.6274]
                                             
Epoch 82 Summary:
  Train - Total: 2.3277, HM: 0.2870, PAF: 0.1482, Cls: 0.5889
  Val   - Total: 2.5271, HM: 0.3130, PAF: 0.1620, Cls: 0.6339
  Metrics - Class Acc: 64.12%, PCK@0.2: 60.73%
  LR: 0.000643
Epoch 83: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.4127, hm=0.2834, paf=0.1336, cls=0.6747]

Epoch 83 - Train: 2.3290 (HM: 0.2856, PAF: 0.1475, Cls: 0.5944)  LR: 0.000636
Epoch 84: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.6471, hm=0.3180, paf=0.1833, cls=0.6722]
                                             
Epoch 84 Summary:
  Train - Total: 2.3375, HM: 0.2864, PAF: 0.1503, Cls: 0.5942
  Val   - Total: 2.4901, HM: 0.3046, PAF: 0.1637, Cls: 0.6295
  Metrics - Class Acc: 65.03%, PCK@0.2: 58.37%
  LR: 0.000628
Epoch 85: 100% 41/41 [00:05<00:00,  6.91it/s, loss=2.7733, hm=0.3032, paf=0.1889, cls=0.7885]

Epoch 85 - Train: 2.3418 (HM: 0.2868, PAF: 0.1493, Cls: 0.5974)  LR: 0.000621
Epoch 86: 100% 41/41 [00:06<00:00,  6.69it/s, loss=2.0295, hm=0.2419, paf=0.1373, cls=0.5248]
                                             
Epoch 86 Summary:
  Train - Total: 2.3066, HM: 0.2833, PAF: 0.1475, Cls: 0.5856
  Val   - Total: 2.4521, HM: 0.2917, PAF: 0.1674, Cls: 0.6338
  Metrics - Class Acc: 64.03%, PCK@0.2: 62.95%
  LR: 0.000613
Epoch 87: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.4167, hm=0.2821, paf=0.1728, cls=0.6284]

Epoch 87 - Train: 2.3271 (HM: 0.2852, PAF: 0.1500, Cls: 0.5909)  LR: 0.000605
Epoch 88: 100% 41/41 [00:06<00:00,  6.71it/s, loss=3.3291, hm=0.4091, paf=0.2735, cls=0.7638]
                                             
Epoch 88 Summary:
  Train - Total: 2.3186, HM: 0.2852, PAF: 0.1475, Cls: 0.5886
  Val   - Total: 2.4484, HM: 0.2980, PAF: 0.1507, Cls: 0.6368
  Metrics - Class Acc: 63.61%, PCK@0.2: 62.95%
  LR: 0.000598
Epoch 89: 100% 41/41 [00:06<00:00,  6.74it/s, loss=3.1286, hm=0.4097, paf=0.1996, cls=0.7269]

Epoch 89 - Train: 2.3265 (HM: 0.2875, PAF: 0.1474, Cls: 0.5878)  LR: 0.000590
Epoch 90: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.6279, hm=0.3727, paf=0.1512, cls=0.5566]
                                             
Epoch 90 Summary:
  Train - Total: 2.3020, HM: 0.2848, PAF: 0.1446, Cls: 0.5825
  Val   - Total: 2.4483, HM: 0.2927, PAF: 0.1699, Cls: 0.6252
  Metrics - Class Acc: 65.48%, PCK@0.2: 63.03%
  LR: 0.000582
Epoch 91: 100% 41/41 [00:05<00:00,  6.87it/s, loss=3.1226, hm=0.4427, paf=0.1821, cls=0.6583]

Epoch 91 - Train: 2.3115 (HM: 0.2853, PAF: 0.1464, Cls: 0.5850)  LR: 0.000575
Epoch 92: 100% 41/41 [00:05<00:00,  6.83it/s, loss=2.5534, hm=0.2985, paf=0.1948, cls=0.6466]
                                             
Epoch 92 Summary:
  Train - Total: 2.3058, HM: 0.2846, PAF: 0.1471, Cls: 0.5821
  Val   - Total: 2.4353, HM: 0.2956, PAF: 0.1640, Cls: 0.6167
  Metrics - Class Acc: 66.90%, PCK@0.2: 64.09%
  LR: 0.000567
Epoch 93: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.7027, hm=0.3119, paf=0.1711, cls=0.7420]

Epoch 93 - Train: 2.2964 (HM: 0.2840, PAF: 0.1450, Cls: 0.5804)  LR: 0.000559
Epoch 94: 100% 41/41 [00:06<00:00,  6.72it/s, loss=1.9849, hm=0.2208, paf=0.1382, cls=0.5502]
                                             
Epoch 94 Summary:
  Train - Total: 2.2870, HM: 0.2813, PAF: 0.1439, Cls: 0.5825
  Val   - Total: 2.4112, HM: 0.2941, PAF: 0.1525, Cls: 0.6199
  Metrics - Class Acc: 65.90%, PCK@0.2: 66.08%
  LR: 0.000552
Epoch 95: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.8970, hm=0.3687, paf=0.2022, cls=0.6786]

Epoch 95 - Train: 2.3051 (HM: 0.2847, PAF: 0.1470, Cls: 0.5816)  LR: 0.000544
Epoch 96: 100% 41/41 [00:06<00:00,  6.73it/s, loss=2.4134, hm=0.2985, paf=0.1717, cls=0.5839]
                                             
Epoch 96 Summary:
  Train - Total: 2.2856, HM: 0.2835, PAF: 0.1437, Cls: 0.5762
  Val   - Total: 2.3638, HM: 0.2854, PAF: 0.1487, Cls: 0.6166
  Metrics - Class Acc: 66.57%, PCK@0.2: 65.24%
  LR: 0.000536
  💾 Saved best model (val_loss: 2.3638)
Epoch 97: 100% 41/41 [00:06<00:00,  6.61it/s, loss=1.7802, hm=0.1954, paf=0.1683, cls=0.4414]

Epoch 97 - Train: 2.2717 (HM: 0.2810, PAF: 0.1453, Cls: 0.5714)  LR: 0.000528
Epoch 98: 100% 41/41 [00:06<00:00,  6.68it/s, loss=2.0704, hm=0.2358, paf=0.1458, cls=0.5571]
                                             
Epoch 98 Summary:
  Train - Total: 2.2503, HM: 0.2789, PAF: 0.1412, Cls: 0.5684
  Val   - Total: 2.4260, HM: 0.2956, PAF: 0.1557, Cls: 0.6215
  Metrics - Class Acc: 66.08%, PCK@0.2: 64.86%
  LR: 0.000521
Epoch 99: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.4132, hm=0.3321, paf=0.1462, cls=0.5283]

Epoch 99 - Train: 2.2702 (HM: 0.2819, PAF: 0.1436, Cls: 0.5704)  LR: 0.000513
Epoch 100: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.2984, hm=0.2880, paf=0.1251, cls=0.5975]
                                             
Epoch 100 Summary:
  Train - Total: 2.2631, HM: 0.2806, PAF: 0.1415, Cls: 0.5719
  Val   - Total: 2.3555, HM: 0.2845, PAF: 0.1464, Cls: 0.6166
  Metrics - Class Acc: 66.97%, PCK@0.2: 66.08%
  LR: 0.000505
  💾 Saved best model (val_loss: 2.3555)
Epoch 101: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.2719, hm=0.2636, paf=0.1664, cls=0.5899]

Epoch 101 - Train: 2.2604 (HM: 0.2800, PAF: 0.1438, Cls: 0.5686)  LR: 0.000497
Epoch 102: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.5840, hm=0.3006, paf=0.1857, cls=0.6734]
                                             
Epoch 102 Summary:
  Train - Total: 2.2594, HM: 0.2794, PAF: 0.1416, Cls: 0.5724
  Val   - Total: 2.3527, HM: 0.2846, PAF: 0.1413, Cls: 0.6210
  Metrics - Class Acc: 66.14%, PCK@0.2: 65.32%
  LR: 0.000489
  💾 Saved best model (val_loss: 2.3527)
Epoch 103: 100% 41/41 [00:05<00:00,  6.92it/s, loss=2.4735, hm=0.3054, paf=0.1334, cls=0.6566]

Epoch 103 - Train: 2.2503 (HM: 0.2794, PAF: 0.1427, Cls: 0.5649)  LR: 0.000482
Epoch 104: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.8449, hm=0.4130, paf=0.1616, cls=0.5797]
                                             
Epoch 104 Summary:
  Train - Total: 2.2485, HM: 0.2812, PAF: 0.1392, Cls: 0.5634
  Val   - Total: 2.3462, HM: 0.2823, PAF: 0.1501, Cls: 0.6112
  Metrics - Class Acc: 67.21%, PCK@0.2: 65.01%
  LR: 0.000474
  💾 Saved best model (val_loss: 2.3462)
Epoch 105: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.6621, hm=0.3212, paf=0.1497, cls=0.7187]

Epoch 105 - Train: 2.2386 (HM: 0.2781, PAF: 0.1411, Cls: 0.5627)  LR: 0.000466
Epoch 106: 100% 41/41 [00:05<00:00,  6.83it/s, loss=2.3046, hm=0.2616, paf=0.1707, cls=0.6112]
                                             
Epoch 106 Summary:
  Train - Total: 2.2213, HM: 0.2759, PAF: 0.1406, Cls: 0.5576
  Val   - Total: 2.4272, HM: 0.2863, PAF: 0.1594, Cls: 0.6423
  Metrics - Class Acc: 63.71%, PCK@0.2: 65.32%
  LR: 0.000458
Epoch 107: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.7449, hm=0.3615, paf=0.1456, cls=0.6718]

Epoch 107 - Train: 2.2618 (HM: 0.2821, PAF: 0.1417, Cls: 0.5667)  LR: 0.000451
Epoch 108: 100% 41/41 [00:06<00:00,  6.80it/s, loss=1.9483, hm=0.2604, paf=0.0994, cls=0.4720]

Epoch 108 Summary:
  Train - Total: 2.2185, HM: 0.2767, PAF: 0.1392, Cls: 0.5555
  Val   - Total: 2.4157, HM: 0.2847, PAF: 0.1562, Cls: 0.6428
  Metrics - Class Acc: 63.22%, PCK@0.2: 63.33%
  LR: 0.000443
Epoch 109: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.6028, hm=0.3473, paf=0.1669, cls=0.5866]

Epoch 109 - Train: 2.2386 (HM: 0.2787, PAF: 0.1404, Cls: 0.5621)  LR: 0.000435
Epoch 110: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.7709, hm=0.3333, paf=0.1706, cls=0.7311]
                                             
Epoch 110 Summary:
  Train - Total: 2.2265, HM: 0.2775, PAF: 0.1372, Cls: 0.5613
  Val   - Total: 2.3378, HM: 0.2836, PAF: 0.1493, Cls: 0.6030
  Metrics - Class Acc: 68.39%, PCK@0.2: 65.01%
  LR: 0.000428
  💾 Saved best model (val_loss: 2.3378)
Epoch 111: 100% 41/41 [00:05<00:00,  6.92it/s, loss=2.6636, hm=0.3383, paf=0.1665, cls=0.6516]

Epoch 111 - Train: 2.2402 (HM: 0.2796, PAF: 0.1390, Cls: 0.5624)  LR: 0.000420
Epoch 112: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.2760, hm=0.2885, paf=0.1503, cls=0.5477]
                                             
Epoch 112 Summary:
  Train - Total: 2.2176, HM: 0.2767, PAF: 0.1393, Cls: 0.5548
  Val   - Total: 2.3493, HM: 0.2822, PAF: 0.1499, Cls: 0.6137
  Metrics - Class Acc: 67.02%, PCK@0.2: 66.00%
  LR: 0.000412
Epoch 113: 100% 41/41 [00:06<00:00,  6.82it/s, loss=1.9155, hm=0.2252, paf=0.1655, cls=0.4557]

Epoch 113 - Train: 2.2120 (HM: 0.2756, PAF: 0.1399, Cls: 0.5532)  LR: 0.000405
Epoch 114: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.4767, hm=0.2841, paf=0.1716, cls=0.6646]
                                             
Epoch 114 Summary:
  Train - Total: 2.2236, HM: 0.2779, PAF: 0.1399, Cls: 0.5548
  Val   - Total: 2.3655, HM: 0.2847, PAF: 0.1560, Cls: 0.6096
  Metrics - Class Acc: 67.66%, PCK@0.2: 64.94%
  LR: 0.000397
Epoch 115: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.6999, hm=0.3897, paf=0.1154, cls=0.6070]

Epoch 115 - Train: 2.2112 (HM: 0.2783, PAF: 0.1390, Cls: 0.5467)  LR: 0.000389
Epoch 116: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.1342, hm=0.2184, paf=0.1668, cls=0.6179]
                                             
Epoch 116 Summary:
  Train - Total: 2.2088, HM: 0.2756, PAF: 0.1380, Cls: 0.5535
  Val   - Total: 2.4098, HM: 0.2836, PAF: 0.1535, Cls: 0.6455
  Metrics - Class Acc: 63.28%, PCK@0.2: 67.15%
  LR: 0.000382
Epoch 117: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.4862, hm=0.2545, paf=0.2086, cls=0.7006]

Epoch 117 - Train: 2.2201 (HM: 0.2775, PAF: 0.1386, Cls: 0.5553)  LR: 0.000374
Epoch 118: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.7069, hm=0.3318, paf=0.1566, cls=0.7109]
                                             
Epoch 118 Summary:
  Train - Total: 2.2219, HM: 0.2767, PAF: 0.1382, Cls: 0.5592
  Val   - Total: 2.3112, HM: 0.2813, PAF: 0.1478, Cls: 0.5936
  Metrics - Class Acc: 69.52%, PCK@0.2: 64.17%
  LR: 0.000367
  💾 Saved best model (val_loss: 2.3112)
Epoch 119: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.6580, hm=0.2720, paf=0.1811, cls=0.8051]

Epoch 119 - Train: 2.2140 (HM: 0.2761, PAF: 0.1379, Cls: 0.5558)  LR: 0.000359
Epoch 120: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.6083, hm=0.3458, paf=0.1584, cls=0.6055]
                                             
Epoch 120 Summary:
  Train - Total: 2.2179, HM: 0.2791, PAF: 0.1373, Cls: 0.5513
  Val   - Total: 2.3549, HM: 0.2815, PAF: 0.1525, Cls: 0.6158
  Metrics - Class Acc: 66.80%, PCK@0.2: 65.16%
  LR: 0.000352
Epoch 121: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.1164, hm=0.2584, paf=0.1263, cls=0.5535]

Epoch 121 - Train: 2.2100 (HM: 0.2759, PAF: 0.1377, Cls: 0.5539)  LR: 0.000345
Epoch 122: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.7526, hm=0.3389, paf=0.1960, cls=0.6701]
                                             
Epoch 122 Summary:
  Train - Total: 2.2139, HM: 0.2783, PAF: 0.1387, Cls: 0.5489
  Val   - Total: 2.3203, HM: 0.2818, PAF: 0.1498, Cls: 0.5956
  Metrics - Class Acc: 69.24%, PCK@0.2: 64.09%
  LR: 0.000337
Epoch 123: 100% 41/41 [00:05<00:00,  6.90it/s, loss=2.8114, hm=0.3705, paf=0.1609, cls=0.6717]

Epoch 123 - Train: 2.2040 (HM: 0.2763, PAF: 0.1373, Cls: 0.5495)  LR: 0.000330
Epoch 124: 100% 41/41 [00:06<00:00,  6.70it/s, loss=2.1711, hm=0.2787, paf=0.1115, cls=0.5554]
                                             
Epoch 124 Summary:
  Train - Total: 2.1747, HM: 0.2739, PAF: 0.1346, Cls: 0.5399
  Val   - Total: 2.3008, HM: 0.2794, PAF: 0.1442, Cls: 0.5967
  Metrics - Class Acc: 69.33%, PCK@0.2: 65.70%
  LR: 0.000323
  💾 Saved best model (val_loss: 2.3008)
Epoch 125: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.0926, hm=0.2749, paf=0.0768, cls=0.5596]

Epoch 125 - Train: 2.1863 (HM: 0.2752, PAF: 0.1338, Cls: 0.5453)  LR: 0.000316
Epoch 126: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.3996, hm=0.3098, paf=0.1553, cls=0.5666]
                                             
Epoch 126 Summary:
  Train - Total: 2.1843, HM: 0.2742, PAF: 0.1358, Cls: 0.5438
  Val   - Total: 2.3439, HM: 0.2801, PAF: 0.1470, Cls: 0.6196
  Metrics - Class Acc: 66.60%, PCK@0.2: 67.00%
  LR: 0.000308
Epoch 127: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.6138, hm=0.3015, paf=0.2244, cls=0.6393]

Epoch 127 - Train: 2.2018 (HM: 0.2755, PAF: 0.1377, Cls: 0.5496)  LR: 0.000301
Epoch 128: 100% 41/41 [00:06<00:00,  6.65it/s, loss=2.8198, hm=0.3336, paf=0.1757, cls=0.7560]
                                             
Epoch 128 Summary:
  Train - Total: 2.2108, HM: 0.2756, PAF: 0.1376, Cls: 0.5553
  Val   - Total: 2.3537, HM: 0.2794, PAF: 0.1447, Cls: 0.6312
  Metrics - Class Acc: 65.38%, PCK@0.2: 66.31%
  LR: 0.000294
Epoch 129: 100% 41/41 [00:05<00:00,  6.89it/s, loss=2.3397, hm=0.3274, paf=0.1394, cls=0.5009]

Epoch 129 - Train: 2.1806 (HM: 0.2746, PAF: 0.1344, Cls: 0.5421)  LR: 0.000287
Epoch 130: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.2203, hm=0.2729, paf=0.1622, cls=0.5362]
                                             
Epoch 130 Summary:
  Train - Total: 2.1700, HM: 0.2733, PAF: 0.1336, Cls: 0.5399
  Val   - Total: 2.3281, HM: 0.2784, PAF: 0.1442, Cls: 0.6174
  Metrics - Class Acc: 66.89%, PCK@0.2: 66.23%
  LR: 0.000280
Epoch 131: 100% 41/41 [00:06<00:00,  6.71it/s, loss=2.9724, hm=0.3890, paf=0.1803, cls=0.7038]

Epoch 131 - Train: 2.2032 (HM: 0.2763, PAF: 0.1367, Cls: 0.5498)  LR: 0.000273
Epoch 132: 100% 41/41 [00:06<00:00,  6.75it/s, loss=2.5368, hm=0.2816, paf=0.1871, cls=0.6909]
                                             
Epoch 132 Summary:
  Train - Total: 2.2095, HM: 0.2776, PAF: 0.1384, Cls: 0.5482
  Val   - Total: 2.3258, HM: 0.2809, PAF: 0.1502, Cls: 0.6013
  Metrics - Class Acc: 68.85%, PCK@0.2: 65.85%
  LR: 0.000267
Epoch 133: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.1953, hm=0.2581, paf=0.1672, cls=0.5524]

Epoch 133 - Train: 2.1614 (HM: 0.2719, PAF: 0.1344, Cls: 0.5368)  LR: 0.000260
Epoch 134: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.2826, hm=0.2686, paf=0.1597, cls=0.5925]
                                             
Epoch 134 Summary:
  Train - Total: 2.1901, HM: 0.2748, PAF: 0.1360, Cls: 0.5460
  Val   - Total: 2.3212, HM: 0.2802, PAF: 0.1489, Cls: 0.6017
  Metrics - Class Acc: 68.79%, PCK@0.2: 67.53%
  LR: 0.000253
Epoch 135: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.2689, hm=0.3118, paf=0.1557, cls=0.4736]

Epoch 135 - Train: 2.1799 (HM: 0.2732, PAF: 0.1370, Cls: 0.5421)  LR: 0.000246
Epoch 136: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.7975, hm=0.3188, paf=0.2054, cls=0.7410]
                                             
Epoch 136 Summary:
  Train - Total: 2.1929, HM: 0.2757, PAF: 0.1366, Cls: 0.5444
  Val   - Total: 2.3673, HM: 0.2870, PAF: 0.1536, Cls: 0.6081
  Metrics - Class Acc: 68.03%, PCK@0.2: 65.78%
  LR: 0.000240
Epoch 137: 100% 41/41 [00:05<00:00,  6.84it/s, loss=2.3751, hm=0.2923, paf=0.1615, cls=0.5886]

Epoch 137 - Train: 2.1701 (HM: 0.2739, PAF: 0.1345, Cls: 0.5369)  LR: 0.000233
Epoch 138: 100% 41/41 [00:06<00:00,  6.67it/s, loss=2.4485, hm=0.3189, paf=0.1473, cls=0.5855]
                                             
Epoch 138 Summary:
  Train - Total: 2.1692, HM: 0.2719, PAF: 0.1344, Cls: 0.5418
  Val   - Total: 2.3437, HM: 0.2823, PAF: 0.1478, Cls: 0.6125
  Metrics - Class Acc: 67.28%, PCK@0.2: 66.16%
  LR: 0.000227
Epoch 139: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.7498, hm=0.2953, paf=0.2419, cls=0.7232]

Epoch 139 - Train: 2.1785 (HM: 0.2726, PAF: 0.1361, Cls: 0.5440)  LR: 0.000220
Epoch 140: 100% 41/41 [00:06<00:00,  6.71it/s, loss=2.6688, hm=0.3890, paf=0.0938, cls=0.6168]
                                             
Epoch 140 Summary:
  Train - Total: 2.1759, HM: 0.2755, PAF: 0.1319, Cls: 0.5402
  Val   - Total: 2.3260, HM: 0.2778, PAF: 0.1433, Cls: 0.6188
  Metrics - Class Acc: 66.69%, PCK@0.2: 66.92%
  LR: 0.000214
Epoch 141: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.4807, hm=0.2781, paf=0.1944, cls=0.6529]

Epoch 141 - Train: 2.1625 (HM: 0.2717, PAF: 0.1332, Cls: 0.5394)  LR: 0.000208
Epoch 142: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.5147, hm=0.3069, paf=0.1698, cls=0.6316]
                                             
Epoch 142 Summary:
  Train - Total: 2.1667, HM: 0.2733, PAF: 0.1329, Cls: 0.5385
  Val   - Total: 2.3165, HM: 0.2773, PAF: 0.1471, Cls: 0.6088
  Metrics - Class Acc: 68.03%, PCK@0.2: 67.07%
  LR: 0.000202
Epoch 143: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.8573, hm=0.3879, paf=0.2340, cls=0.5585]

Epoch 143 - Train: 2.1719 (HM: 0.2747, PAF: 0.1352, Cls: 0.5351)  LR: 0.000196
Epoch 144: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.6910, hm=0.2899, paf=0.2424, cls=0.6976]
                                             
Epoch 144 Summary:
  Train - Total: 2.1889, HM: 0.2755, PAF: 0.1367, Cls: 0.5424
  Val   - Total: 2.3288, HM: 0.2812, PAF: 0.1484, Cls: 0.6049
  Metrics - Class Acc: 68.36%, PCK@0.2: 66.39%
  LR: 0.000189
Epoch 145: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.8474, hm=0.3931, paf=0.1623, cls=0.6335]

Epoch 145 - Train: 2.1808 (HM: 0.2756, PAF: 0.1345, Cls: 0.5395)  LR: 0.000184
Epoch 146: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.2699, hm=0.2756, paf=0.1641, cls=0.5596]
                                             
Epoch 146 Summary:
  Train - Total: 2.1566, HM: 0.2723, PAF: 0.1333, Cls: 0.5338
  Val   - Total: 2.3067, HM: 0.2771, PAF: 0.1406, Cls: 0.6113
  Metrics - Class Acc: 67.57%, PCK@0.2: 67.30%
  LR: 0.000178
Epoch 147: 100% 41/41 [00:06<00:00,  6.73it/s, loss=2.2531, hm=0.3095, paf=0.1350, cls=0.4969]

Epoch 147 - Train: 2.1556 (HM: 0.2730, PAF: 0.1326, Cls: 0.5322)  LR: 0.000172
Epoch 148: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.8239, hm=0.4039, paf=0.1628, cls=0.5884]
                                             
Epoch 148 Summary:
  Train - Total: 2.1630, HM: 0.2741, PAF: 0.1348, Cls: 0.5314
  Val   - Total: 2.2859, HM: 0.2760, PAF: 0.1432, Cls: 0.5971
  Metrics - Class Acc: 69.43%, PCK@0.2: 68.30%
  LR: 0.000166
  💾 Saved best model (val_loss: 2.2859)
Epoch 149: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.2460, hm=0.2745, paf=0.1000, cls=0.6321]

Epoch 149 - Train: 2.1713 (HM: 0.2738, PAF: 0.1331, Cls: 0.5400)  LR: 0.000161
Epoch 150: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.4611, hm=0.2974, paf=0.2041, cls=0.5757]
                                             
Epoch 150 Summary:
  Train - Total: 2.1435, HM: 0.2710, PAF: 0.1327, Cls: 0.5294
  Val   - Total: 2.2997, HM: 0.2754, PAF: 0.1469, Cls: 0.6028
  Metrics - Class Acc: 68.59%, PCK@0.2: 67.69%
  LR: 0.000155
Epoch 151: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.7288, hm=0.3558, paf=0.1701, cls=0.6436]

Epoch 151 - Train: 2.1613 (HM: 0.2739, PAF: 0.1334, Cls: 0.5326)  LR: 0.000150
Epoch 152: 100% 41/41 [00:06<00:00,  6.69it/s, loss=2.2238, hm=0.2523, paf=0.1773, cls=0.5734]
                                             
Epoch 152 Summary:
  Train - Total: 2.1545, HM: 0.2717, PAF: 0.1330, Cls: 0.5344
  Val   - Total: 2.3157, HM: 0.2773, PAF: 0.1474, Cls: 0.6078
  Metrics - Class Acc: 68.07%, PCK@0.2: 68.22%
  LR: 0.000144
Epoch 153: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.5455, hm=0.2835, paf=0.1518, cls=0.7386]

Epoch 153 - Train: 2.1561 (HM: 0.2713, PAF: 0.1315, Cls: 0.5386)  LR: 0.000139
Epoch 154: 100% 41/41 [00:06<00:00,  6.70it/s, loss=2.7136, hm=0.3344, paf=0.1849, cls=0.6709]
                                             
Epoch 154 Summary:
  Train - Total: 2.1525, HM: 0.2724, PAF: 0.1328, Cls: 0.5316
  Val   - Total: 2.2911, HM: 0.2765, PAF: 0.1470, Cls: 0.5939
  Metrics - Class Acc: 69.52%, PCK@0.2: 66.92%
  LR: 0.000134
Epoch 155: 100% 41/41 [00:06<00:00,  6.65it/s, loss=3.1176, hm=0.4468, paf=0.1807, cls=0.6460]

Epoch 155 - Train: 2.1591 (HM: 0.2752, PAF: 0.1321, Cls: 0.5293)  LR: 0.000129
Epoch 156: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.4954, hm=0.3393, paf=0.1359, cls=0.5777]
                                             
Epoch 156 Summary:
  Train - Total: 2.1606, HM: 0.2743, PAF: 0.1337, Cls: 0.5306
  Val   - Total: 2.2764, HM: 0.2752, PAF: 0.1446, Cls: 0.5909
  Metrics - Class Acc: 70.11%, PCK@0.2: 67.38%
  LR: 0.000124
  💾 Saved best model (val_loss: 2.2764)
Epoch 157: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.3217, hm=0.2851, paf=0.1613, cls=0.5725]

Epoch 157 - Train: 2.1399 (HM: 0.2713, PAF: 0.1314, Cls: 0.5278)  LR: 0.000119
Epoch 158: 100% 41/41 [00:05<00:00,  6.83it/s, loss=2.7725, hm=0.3876, paf=0.1549, cls=0.6081]
                                             
Epoch 158 Summary:
  Train - Total: 2.1458, HM: 0.2726, PAF: 0.1320, Cls: 0.5275
  Val   - Total: 2.3202, HM: 0.2769, PAF: 0.1467, Cls: 0.6129
  Metrics - Class Acc: 67.54%, PCK@0.2: 68.30%
  LR: 0.000114
Epoch 159: 100% 41/41 [00:06<00:00,  6.75it/s, loss=2.6336, hm=0.3713, paf=0.1598, cls=0.5524]

Epoch 159 - Train: 2.1411 (HM: 0.2729, PAF: 0.1312, Cls: 0.5247)  LR: 0.000109
Epoch 160: 100% 41/41 [00:06<00:00,  6.73it/s, loss=2.7340, hm=0.3463, paf=0.1690, cls=0.6740]
                                             
Epoch 160 Summary:
  Train - Total: 2.1490, HM: 0.2717, PAF: 0.1308, Cls: 0.5338
  Val   - Total: 2.2837, HM: 0.2761, PAF: 0.1403, Cls: 0.5991
  Metrics - Class Acc: 69.13%, PCK@0.2: 67.23%
  LR: 0.000105
Epoch 161: 100% 41/41 [00:06<00:00,  6.71it/s, loss=2.0128, hm=0.2622, paf=0.1401, cls=0.4559]

Epoch 161 - Train: 2.1331 (HM: 0.2701, PAF: 0.1310, Cls: 0.5273)  LR: 0.000100
Epoch 162: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.5134, hm=0.3501, paf=0.1543, cls=0.5363]
                                             
Epoch 162 Summary:
  Train - Total: 2.1376, HM: 0.2722, PAF: 0.1317, Cls: 0.5237
  Val   - Total: 2.2999, HM: 0.2752, PAF: 0.1440, Cls: 0.6075
  Metrics - Class Acc: 68.13%, PCK@0.2: 67.23%
  LR: 0.000096
Epoch 163: 100% 41/41 [00:05<00:00,  6.87it/s, loss=1.9948, hm=0.2502, paf=0.0944, cls=0.5368]

Epoch 163 - Train: 2.1296 (HM: 0.2694, PAF: 0.1303, Cls: 0.5276)  LR: 0.000091
Epoch 164: 100% 41/41 [00:06<00:00,  6.81it/s, loss=2.5051, hm=0.3217, paf=0.1441, cls=0.6200]
                                             
Epoch 164 Summary:
  Train - Total: 2.1354, HM: 0.2705, PAF: 0.1306, Cls: 0.5280
  Val   - Total: 2.2780, HM: 0.2747, PAF: 0.1425, Cls: 0.5962
  Metrics - Class Acc: 69.44%, PCK@0.2: 68.75%
  LR: 0.000087
Epoch 165: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.4412, hm=0.3229, paf=0.1635, cls=0.5485]

Epoch 165 - Train: 2.1399 (HM: 0.2718, PAF: 0.1302, Cls: 0.5282)  LR: 0.000083
Epoch 166: 100% 41/41 [00:06<00:00,  6.79it/s, loss=2.2490, hm=0.3060, paf=0.1185, cls=0.5253]
                                             
Epoch 166 Summary:
  Train - Total: 2.1206, HM: 0.2699, PAF: 0.1283, Cls: 0.5230
  Val   - Total: 2.2675, HM: 0.2741, PAF: 0.1401, Cls: 0.5940
  Metrics - Class Acc: 69.54%, PCK@0.2: 68.30%
  LR: 0.000079
  💾 Saved best model (val_loss: 2.2675)
Epoch 167: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.3101, hm=0.2706, paf=0.1535, cls=0.6138]

Epoch 167 - Train: 2.1396 (HM: 0.2707, PAF: 0.1314, Cls: 0.5294)  LR: 0.000075
Epoch 168: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.8200, hm=0.3784, paf=0.1937, cls=0.6129]
                                             
Epoch 168 Summary:
  Train - Total: 2.1529, HM: 0.2740, PAF: 0.1327, Cls: 0.5276
  Val   - Total: 2.2772, HM: 0.2746, PAF: 0.1423, Cls: 0.5963
  Metrics - Class Acc: 69.43%, PCK@0.2: 67.99%
  LR: 0.000071
Epoch 169: 100% 41/41 [00:06<00:00,  6.79it/s, loss=3.1192, hm=0.4408, paf=0.1567, cls=0.6950]

Epoch 169 - Train: 2.1484 (HM: 0.2736, PAF: 0.1310, Cls: 0.5280)  LR: 0.000068
Epoch 170: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.2934, hm=0.2773, paf=0.1335, cls=0.6116]
                                             
Epoch 170 Summary:
  Train - Total: 2.1326, HM: 0.2725, PAF: 0.1300, Cls: 0.5216
  Val   - Total: 2.2800, HM: 0.2750, PAF: 0.1429, Cls: 0.5961
  Metrics - Class Acc: 69.34%, PCK@0.2: 69.29%
  LR: 0.000064
Epoch 171: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.3694, hm=0.2655, paf=0.1895, cls=0.6191]

Epoch 171 - Train: 2.1270 (HM: 0.2682, PAF: 0.1314, Cls: 0.5275)  LR: 0.000060
Epoch 172: 100% 41/41 [00:05<00:00,  6.85it/s, loss=2.1821, hm=0.2833, paf=0.1237, cls=0.5343]
                                             
Epoch 172 Summary:
  Train - Total: 2.1135, HM: 0.2685, PAF: 0.1306, Cls: 0.5190
  Val   - Total: 2.2766, HM: 0.2741, PAF: 0.1435, Cls: 0.5954
  Metrics - Class Acc: 69.45%, PCK@0.2: 68.68%
  LR: 0.000057
Epoch 173: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.4347, hm=0.2748, paf=0.2060, cls=0.6157]

Epoch 173 - Train: 2.1246 (HM: 0.2704, PAF: 0.1316, Cls: 0.5198)  LR: 0.000054
Epoch 174: 100% 41/41 [00:06<00:00,  6.71it/s, loss=2.0320, hm=0.2592, paf=0.1020, cls=0.5274]
                                             
Epoch 174 Summary:
  Train - Total: 2.1145, HM: 0.2681, PAF: 0.1293, Cls: 0.5223
  Val   - Total: 2.2741, HM: 0.2740, PAF: 0.1439, Cls: 0.5935
  Metrics - Class Acc: 69.59%, PCK@0.2: 68.37%
  LR: 0.000051
Epoch 175: 100% 41/41 [00:05<00:00,  6.92it/s, loss=2.6308, hm=0.3593, paf=0.1649, cls=0.5760]

Epoch 175 - Train: 2.1363 (HM: 0.2717, PAF: 0.1311, Cls: 0.5248)  LR: 0.000048
Epoch 176: 100% 41/41 [00:06<00:00,  6.82it/s, loss=1.8650, hm=0.1994, paf=0.1524, cls=0.5084]
                                             
Epoch 176 Summary:
  Train - Total: 2.1051, HM: 0.2673, PAF: 0.1296, Cls: 0.5178
  Val   - Total: 2.2770, HM: 0.2739, PAF: 0.1411, Cls: 0.5994
  Metrics - Class Acc: 69.09%, PCK@0.2: 68.53%
  LR: 0.000045
Epoch 177: 100% 41/41 [00:06<00:00,  6.83it/s, loss=2.3527, hm=0.2883, paf=0.1524, cls=0.5964]

Epoch 177 - Train: 2.1050 (HM: 0.2678, PAF: 0.1293, Cls: 0.5168)  LR: 0.000042
Epoch 178: 100% 41/41 [00:06<00:00,  6.67it/s, loss=2.5715, hm=0.3173, paf=0.1758, cls=0.6339]
                                             
Epoch 178 Summary:
  Train - Total: 2.1320, HM: 0.2713, PAF: 0.1309, Cls: 0.5234
  Val   - Total: 2.2930, HM: 0.2737, PAF: 0.1462, Cls: 0.6039
  Metrics - Class Acc: 68.61%, PCK@0.2: 68.91%
  LR: 0.000039
Epoch 179: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.5407, hm=0.3493, paf=0.1134, cls=0.6111]

Epoch 179 - Train: 2.1359 (HM: 0.2712, PAF: 0.1295, Cls: 0.5280)  LR: 0.000037
Epoch 180: 100% 41/41 [00:06<00:00,  6.73it/s, loss=2.0366, hm=0.2496, paf=0.1069, cls=0.5497]
                                             
Epoch 180 Summary:
  Train - Total: 2.1291, HM: 0.2699, PAF: 0.1292, Cls: 0.5275
  Val   - Total: 2.2878, HM: 0.2739, PAF: 0.1448, Cls: 0.6018
  Metrics - Class Acc: 68.93%, PCK@0.2: 68.98%
  LR: 0.000034
Epoch 181: 100% 41/41 [00:06<00:00,  6.78it/s, loss=2.3261, hm=0.2863, paf=0.1352, cls=0.6069]

Epoch 181 - Train: 2.1171 (HM: 0.2697, PAF: 0.1287, Cls: 0.5206)  LR: 0.000032
Epoch 182: 100% 41/41 [00:05<00:00,  6.84it/s, loss=1.6265, hm=0.1942, paf=0.0623, cls=0.4834]
                                             
Epoch 182 Summary:
  Train - Total: 2.0986, HM: 0.2669, PAF: 0.1286, Cls: 0.5160
  Val   - Total: 2.2765, HM: 0.2729, PAF: 0.1419, Cls: 0.6008
  Metrics - Class Acc: 68.90%, PCK@0.2: 69.21%
  LR: 0.000030
Epoch 183: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.1471, hm=0.2881, paf=0.1385, cls=0.4784]

Epoch 183 - Train: 2.0971 (HM: 0.2665, PAF: 0.1282, Cls: 0.5165)  LR: 0.000028
Epoch 184: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.5343, hm=0.3520, paf=0.1393, cls=0.5652]
                                             
Epoch 184 Summary:
  Train - Total: 2.1109, HM: 0.2697, PAF: 0.1277, Cls: 0.5179
  Val   - Total: 2.2746, HM: 0.2738, PAF: 0.1416, Cls: 0.5974
  Metrics - Class Acc: 69.22%, PCK@0.2: 68.14%
  LR: 0.000026
Epoch 185: 100% 41/41 [00:06<00:00,  6.73it/s, loss=2.5927, hm=0.3446, paf=0.1850, cls=0.5630]

Epoch 185 - Train: 2.1281 (HM: 0.2708, PAF: 0.1301, Cls: 0.5232)  LR: 0.000024
Epoch 186: 100% 41/41 [00:06<00:00,  6.78it/s, loss=1.9091, hm=0.2204, paf=0.1223, cls=0.5219]
                                             
Epoch 186 Summary:
  Train - Total: 2.0975, HM: 0.2662, PAF: 0.1288, Cls: 0.5167
  Val   - Total: 2.2790, HM: 0.2739, PAF: 0.1426, Cls: 0.5987
  Metrics - Class Acc: 69.27%, PCK@0.2: 68.75%
  LR: 0.000022
Epoch 187: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.8924, hm=0.4289, paf=0.1631, cls=0.5672]

Epoch 187 - Train: 2.1384 (HM: 0.2731, PAF: 0.1310, Cls: 0.5226)  LR: 0.000020
Epoch 188: 100% 41/41 [00:06<00:00,  6.74it/s, loss=2.2901, hm=0.2989, paf=0.1640, cls=0.5110]
                                             
Epoch 188 Summary:
  Train - Total: 2.1138, HM: 0.2692, PAF: 0.1284, Cls: 0.5202
  Val   - Total: 2.2792, HM: 0.2735, PAF: 0.1439, Cls: 0.5983
  Metrics - Class Acc: 69.31%, PCK@0.2: 68.22%
  LR: 0.000019
Epoch 189: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.4719, hm=0.3357, paf=0.1454, cls=0.5589]

Epoch 189 - Train: 2.1318 (HM: 0.2702, PAF: 0.1307, Cls: 0.5264)  LR: 0.000017
Epoch 190: 100% 41/41 [00:06<00:00,  6.67it/s, loss=2.2940, hm=0.3128, paf=0.1414, cls=0.5066]
                                             
Epoch 190 Summary:
  Train - Total: 2.1066, HM: 0.2684, PAF: 0.1287, Cls: 0.5172
  Val   - Total: 2.2775, HM: 0.2734, PAF: 0.1442, Cls: 0.5970
  Metrics - Class Acc: 69.29%, PCK@0.2: 68.60%
  LR: 0.000016
Epoch 191: 100% 41/41 [00:05<00:00,  6.86it/s, loss=2.4691, hm=0.2830, paf=0.2374, cls=0.5748]

Epoch 191 - Train: 2.1027 (HM: 0.2675, PAF: 0.1311, Cls: 0.5136)  LR: 0.000015
Epoch 192: 100% 41/41 [00:06<00:00,  6.80it/s, loss=2.4676, hm=0.3157, paf=0.1527, cls=0.5995]
                                             
Epoch 192 Summary:
  Train - Total: 2.1286, HM: 0.2712, PAF: 0.1292, Cls: 0.5235
  Val   - Total: 2.2790, HM: 0.2736, PAF: 0.1431, Cls: 0.5990
  Metrics - Class Acc: 69.03%, PCK@0.2: 68.75%
  LR: 0.000014
Epoch 193: 100% 41/41 [00:05<00:00,  6.83it/s, loss=2.2328, hm=0.2636, paf=0.1528, cls=0.5818]

Epoch 193 - Train: 2.1192 (HM: 0.2686, PAF: 0.1289, Cls: 0.5247)  LR: 0.000013
Epoch 194: 100% 41/41 [00:06<00:00,  6.75it/s, loss=2.3858, hm=0.3097, paf=0.1167, cls=0.6089]
                                             
Epoch 194 Summary:
  Train - Total: 2.1268, HM: 0.2699, PAF: 0.1290, Cls: 0.5260
  Val   - Total: 2.2753, HM: 0.2742, PAF: 0.1434, Cls: 0.5944
  Metrics - Class Acc: 69.72%, PCK@0.2: 68.68%
  LR: 0.000012
Epoch 195: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.7728, hm=0.3848, paf=0.1533, cls=0.6179]

Epoch 195 - Train: 2.1228 (HM: 0.2716, PAF: 0.1285, Cls: 0.5195)  LR: 0.000012
Epoch 196: 100% 41/41 [00:06<00:00,  6.76it/s, loss=2.1164, hm=0.2579, paf=0.1376, cls=0.5399]
                                             
Epoch 196 Summary:
  Train - Total: 2.1154, HM: 0.2695, PAF: 0.1287, Cls: 0.5200
  Val   - Total: 2.2694, HM: 0.2731, PAF: 0.1427, Cls: 0.5945
  Metrics - Class Acc: 69.61%, PCK@0.2: 68.83%
  LR: 0.000011
Epoch 197: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.6512, hm=0.3393, paf=0.1647, cls=0.6432]

Epoch 197 - Train: 2.1290 (HM: 0.2710, PAF: 0.1295, Cls: 0.5239)  LR: 0.000011
Epoch 198: 100% 41/41 [00:06<00:00,  6.82it/s, loss=2.3950, hm=0.2939, paf=0.1584, cls=0.6018]
                                             
Epoch 198 Summary:
  Train - Total: 2.1170, HM: 0.2696, PAF: 0.1294, Cls: 0.5198
  Val   - Total: 2.2746, HM: 0.2737, PAF: 0.1427, Cls: 0.5963
  Metrics - Class Acc: 69.50%, PCK@0.2: 68.60%
  LR: 0.000010
Epoch 199: 100% 41/41 [00:06<00:00,  6.79it/s, loss=3.1621, hm=0.4019, paf=0.2102, cls=0.7561]

Epoch 199 - Train: 2.1360 (HM: 0.2720, PAF: 0.1308, Cls: 0.5241)  LR: 0.000010
Epoch 200: 100% 41/41 [00:06<00:00,  6.77it/s, loss=2.4341, hm=0.3407, paf=0.1127, cls=0.5640]
                                             
Epoch 200 Summary:
  Train - Total: 2.1120, HM: 0.2700, PAF: 0.1282, Cls: 0.5170
  Val   - Total: 2.2761, HM: 0.2728, PAF: 0.1426, Cls: 0.5999
  Metrics - Class Acc: 69.02%, PCK@0.2: 68.68%
  LR: 0.000010

================================================================================
✅ Training complete! Best val loss: 2.2675
================================================================================

✅ Training complete!
📁 Checkpoints saved to: outputs/m1_training/checkpoints