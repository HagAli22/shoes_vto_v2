================================================================================
👟 ARShoe M1 Fine-Tuning Pipeline
================================================================================
  Pretrained Checkpoint : /content/shoes_vto_v2/shoes_vto/outputs/m1_training/checkpoints/best.pth
  Target Dataset Root   : /content/shoes_vto_v2/shoes_vto/dataset/shuffled_v4
  Output Directory      : /content/shoes_vto_v2/shoes_vto/outputs/m1_finetune_v4
  Epochs                : 150
  Learning Rate         : 0.0002 (Cosine decay -> 1e-06)
  Batch Size            : 16
  Workers               : 16
  Mixed Precision (AMP) : True
================================================================================

📁 Loading datasets...
Loaded 1156 images from /content/shoes_vto_v2/shoes_vto/dataset/shuffled_v4/train/images
Loaded 143 images from /content/shoes_vto_v2/shoes_vto/dataset/shuffled_v4/valid/images
  Train: 1156 images
  Val  : 143 images

🏗️ Initializing ARShoe M1 architecture...
  Total parameters: 197,584
  Encoder: 24,096 | Heatmap: 74,896 | PAF: 75,806 | Class: 22,786
📦 Loading pretrained checkpoint: /content/shoes_vto_v2/shoes_vto/outputs/m1_training/checkpoints/best.pth
   Source checkpoint info: Epoch=166, Best Val Loss=2.2675190448760985
✅ Pretrained weights loaded strictly and verified successfully!

🎯 Initializing Trainer with fresh fine-tuning optimizer & scheduler...
  [GPU] Enabled TF32 & cuDNN benchmark for A100 maximum throughput!
  [GPU] Model compiled with torch.compile() for A100 maximum speed!
/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:145: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

🚀 Launching Fine-Tuning for 150 epochs...

================================================================================
Training ARShoe M1 for 150 epochs
Device: cuda
Mixed Precision (AMP): Enabled
Train samples: 1156
Val samples: 143
Batch size: 16
Learning rate: 0.0002
Loss weights: {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
================================================================================
Epoch 1:   0% 0/73 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1:   1% 1/73 [00:03<04:03,  3.38s/it, loss=3.6442, hm=0.4927, paf=0.2555, cls=0.7749]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1: 100% 73/73 [00:09<00:00,  7.94it/s, loss=3.4861, hm=0.4255, paf=0.2671, cls=0.8334]

Epoch 1 - Train: 3.4628 (HM: 0.4522, PAF: 0.2607, Cls: 0.7550)  LR: 0.000200
Epoch 2: 100% 73/73 [00:04<00:00, 17.10it/s, loss=3.1893, hm=0.4286, paf=0.2494, cls=0.6506]
                                             
Epoch 2 Summary:
  Train - Total: 3.2644, HM: 0.4276, PAF: 0.2381, Cls: 0.7185
  Val   - Total: 3.2222, HM: 0.4200, PAF: 0.2309, Cls: 0.7202
  Metrics - Class Acc: 52.80%, PCK@0.2: 42.41%
  LR: 0.000200
  💾 Saved best model (val_loss: 3.2222)
Epoch 3: 100% 73/73 [00:04<00:00, 17.64it/s, loss=2.7111, hm=0.3017, paf=0.2097, cls=0.7232]

Epoch 3 - Train: 3.1677 (HM: 0.4148, PAF: 0.2290, Cls: 0.7003)  LR: 0.000200
Epoch 4: 100% 73/73 [00:04<00:00, 17.61it/s, loss=2.9124, hm=0.3765, paf=0.1951, cls=0.6775]
                                             
Epoch 4 Summary:
  Train - Total: 3.1231, HM: 0.4104, PAF: 0.2239, Cls: 0.6891
  Val   - Total: 3.1049, HM: 0.4066, PAF: 0.2239, Cls: 0.6871
  Metrics - Class Acc: 55.62%, PCK@0.2: 44.95%
  LR: 0.000200
  💾 Saved best model (val_loss: 3.1049)
Epoch 5: 100% 73/73 [00:04<00:00, 17.83it/s, loss=3.2648, hm=0.4473, paf=0.2205, cls=0.6897]

Epoch 5 - Train: 3.0916 (HM: 0.4057, PAF: 0.2219, Cls: 0.6834)  LR: 0.000199
Epoch 6: 100% 73/73 [00:04<00:00, 17.82it/s, loss=3.2144, hm=0.4350, paf=0.2188, cls=0.6912]
                                             
Epoch 6 Summary:
  Train - Total: 3.0627, HM: 0.4007, PAF: 0.2185, Cls: 0.6819
  Val   - Total: 3.0769, HM: 0.4021, PAF: 0.2184, Cls: 0.6878
  Metrics - Class Acc: 55.15%, PCK@0.2: 47.00%
  LR: 0.000199
  💾 Saved best model (val_loss: 3.0769)
Epoch 7: 100% 73/73 [00:04<00:00, 17.77it/s, loss=3.0636, hm=0.4198, paf=0.2010, cls=0.6549]

Epoch 7 - Train: 3.0391 (HM: 0.3980, PAF: 0.2147, Cls: 0.6785)  LR: 0.000199
Epoch 8: 100% 73/73 [00:04<00:00, 17.72it/s, loss=2.9467, hm=0.3825, paf=0.2285, cls=0.6398]
                                             
Epoch 8 Summary:
  Train - Total: 3.0243, HM: 0.3951, PAF: 0.2142, Cls: 0.6771
  Val   - Total: 3.0350, HM: 0.3946, PAF: 0.2140, Cls: 0.6858
  Metrics - Class Acc: 56.40%, PCK@0.2: 47.28%
  LR: 0.000199
  💾 Saved best model (val_loss: 3.0350)
Epoch 9: 100% 73/73 [00:04<00:00, 17.86it/s, loss=2.9225, hm=0.3649, paf=0.2173, cls=0.6854]

Epoch 9 - Train: 2.9986 (HM: 0.3911, PAF: 0.2132, Cls: 0.6720)  LR: 0.000198
Epoch 10: 100% 73/73 [00:04<00:00, 17.38it/s, loss=3.1312, hm=0.4324, paf=0.2154, cls=0.6472]
                                             
Epoch 10 Summary:
  Train - Total: 2.9902, HM: 0.3909, PAF: 0.2091, Cls: 0.6725
  Val   - Total: 2.9993, HM: 0.3910, PAF: 0.2141, Cls: 0.6714
  Metrics - Class Acc: 58.42%, PCK@0.2: 47.85%
  LR: 0.000198
  💾 Saved best model (val_loss: 2.9993)
Epoch 11: 100% 73/73 [00:04<00:00, 18.06it/s, loss=3.1953, hm=0.4134, paf=0.2510, cls=0.6931]

Epoch 11 - Train: 2.9831 (HM: 0.3895, PAF: 0.2092, Cls: 0.6711)  LR: 0.000197
Epoch 12: 100% 73/73 [00:04<00:00, 17.92it/s, loss=3.1816, hm=0.4175, paf=0.2630, cls=0.6572]
                                             
Epoch 12 Summary:
  Train - Total: 2.9678, HM: 0.3876, PAF: 0.2074, Cls: 0.6683
  Val   - Total: 2.9830, HM: 0.3880, PAF: 0.2086, Cls: 0.6760
  Metrics - Class Acc: 57.97%, PCK@0.2: 49.29%
  LR: 0.000197
  💾 Saved best model (val_loss: 2.9830)
Epoch 13: 100% 73/73 [00:04<00:00, 17.76it/s, loss=2.9424, hm=0.3686, paf=0.2374, cls=0.6621]

Epoch 13 - Train: 2.9495 (HM: 0.3838, PAF: 0.2064, Cls: 0.6678)  LR: 0.000196
Epoch 14: 100% 73/73 [00:04<00:00, 17.90it/s, loss=3.0151, hm=0.3927, paf=0.2018, cls=0.6940]
                                             
Epoch 14 Summary:
  Train - Total: 2.9466, HM: 0.3844, PAF: 0.2055, Cls: 0.6654
  Val   - Total: 2.9620, HM: 0.3853, PAF: 0.2090, Cls: 0.6685
  Metrics - Class Acc: 58.97%, PCK@0.2: 48.45%
  LR: 0.000196
  💾 Saved best model (val_loss: 2.9620)
Epoch 15: 100% 73/73 [00:04<00:00, 17.71it/s, loss=2.8513, hm=0.3650, paf=0.1956, cls=0.6667]

Epoch 15 - Train: 2.9308 (HM: 0.3813, PAF: 0.2041, Cls: 0.6649)  LR: 0.000195
Epoch 16: 100% 73/73 [00:04<00:00, 17.95it/s, loss=3.2940, hm=0.4868, paf=0.1777, cls=0.6609]
                                             
Epoch 16 Summary:
  Train - Total: 2.9245, HM: 0.3822, PAF: 0.2027, Cls: 0.6602
  Val   - Total: 2.9488, HM: 0.3817, PAF: 0.2044, Cls: 0.6755
  Metrics - Class Acc: 57.52%, PCK@0.2: 50.67%
  LR: 0.000194
  💾 Saved best model (val_loss: 2.9488)
Epoch 17: 100% 73/73 [00:04<00:00, 17.89it/s, loss=2.7303, hm=0.3760, paf=0.1713, cls=0.5891]

Epoch 17 - Train: 2.9103 (HM: 0.3799, PAF: 0.2014, Cls: 0.6587)  LR: 0.000194
Epoch 18: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.9455, hm=0.3850, paf=0.2109, cls=0.6558]
                                             
Epoch 18 Summary:
  Train - Total: 2.8947, HM: 0.3779, PAF: 0.1989, Cls: 0.6569
  Val   - Total: 2.9356, HM: 0.3792, PAF: 0.2061, Cls: 0.6711
  Metrics - Class Acc: 58.63%, PCK@0.2: 50.85%
  LR: 0.000193
  💾 Saved best model (val_loss: 2.9356)
Epoch 19: 100% 73/73 [00:04<00:00, 18.08it/s, loss=3.2362, hm=0.4351, paf=0.2476, cls=0.6672]

Epoch 19 - Train: 2.9044 (HM: 0.3789, PAF: 0.2013, Cls: 0.6575)  LR: 0.000192
Epoch 20: 100% 73/73 [00:04<00:00, 17.15it/s, loss=2.9714, hm=0.3894, paf=0.1988, cls=0.6775]
                                             
Epoch 20 Summary:
  Train - Total: 2.8847, HM: 0.3756, PAF: 0.1996, Cls: 0.6556
  Val   - Total: 2.9343, HM: 0.3814, PAF: 0.2038, Cls: 0.6674
  Metrics - Class Acc: 58.74%, PCK@0.2: 51.31%
  LR: 0.000191
  💾 Saved best model (val_loss: 2.9343)
Epoch 21: 100% 73/73 [00:04<00:00, 17.87it/s, loss=3.2051, hm=0.4067, paf=0.2184, cls=0.7610]

Epoch 21 - Train: 2.8916 (HM: 0.3768, PAF: 0.1995, Cls: 0.6568)  LR: 0.000191
Epoch 22: 100% 73/73 [00:04<00:00, 17.93it/s, loss=2.7863, hm=0.3308, paf=0.2257, cls=0.6746]
                                             
Epoch 22 Summary:
  Train - Total: 2.8760, HM: 0.3747, PAF: 0.1996, Cls: 0.6519
  Val   - Total: 2.9079, HM: 0.3763, PAF: 0.2045, Cls: 0.6625
  Metrics - Class Acc: 59.49%, PCK@0.2: 50.49%
  LR: 0.000190
  💾 Saved best model (val_loss: 2.9079)
Epoch 23: 100% 73/73 [00:04<00:00, 18.08it/s, loss=2.7804, hm=0.3416, paf=0.2487, cls=0.6110]

Epoch 23 - Train: 2.8586 (HM: 0.3731, PAF: 0.1964, Cls: 0.6489)  LR: 0.000189
Epoch 24: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.5005, hm=0.3196, paf=0.1488, cls=0.6164]
                                             
Epoch 24 Summary:
  Train - Total: 2.8580, HM: 0.3734, PAF: 0.1959, Cls: 0.6485
  Val   - Total: 2.8931, HM: 0.3745, PAF: 0.2010, Cls: 0.6621
  Metrics - Class Acc: 59.80%, PCK@0.2: 51.27%
  LR: 0.000188
  💾 Saved best model (val_loss: 2.8931)
Epoch 25: 100% 73/73 [00:04<00:00, 17.93it/s, loss=2.7156, hm=0.3401, paf=0.2035, cls=0.6320]

Epoch 25 - Train: 2.8522 (HM: 0.3732, PAF: 0.1949, Cls: 0.6463)  LR: 0.000187
Epoch 26: 100% 73/73 [00:04<00:00, 18.03it/s, loss=2.6867, hm=0.3419, paf=0.1979, cls=0.6156]
                                             
Epoch 26 Summary:
  Train - Total: 2.8415, HM: 0.3710, PAF: 0.1938, Cls: 0.6465
  Val   - Total: 2.8864, HM: 0.3737, PAF: 0.2005, Cls: 0.6605
  Metrics - Class Acc: 59.43%, PCK@0.2: 51.02%
  LR: 0.000186
  💾 Saved best model (val_loss: 2.8864)
Epoch 27: 100% 73/73 [00:04<00:00, 18.03it/s, loss=3.1192, hm=0.4284, paf=0.2336, cls=0.6256]

Epoch 27 - Train: 2.8351 (HM: 0.3701, PAF: 0.1935, Cls: 0.6451)  LR: 0.000185
Epoch 28: 100% 73/73 [00:04<00:00, 18.17it/s, loss=2.6945, hm=0.3249, paf=0.2093, cls=0.6508]
                                             
Epoch 28 Summary:
  Train - Total: 2.8414, HM: 0.3709, PAF: 0.1951, Cls: 0.6451
  Val   - Total: 2.8802, HM: 0.3717, PAF: 0.1996, Cls: 0.6629
  Metrics - Class Acc: 59.50%, PCK@0.2: 50.85%
  LR: 0.000183
  💾 Saved best model (val_loss: 2.8802)
Epoch 29: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.9265, hm=0.4164, paf=0.1592, cls=0.6284]

Epoch 29 - Train: 2.8347 (HM: 0.3712, PAF: 0.1934, Cls: 0.6421)  LR: 0.000182
Epoch 30: 100% 73/73 [00:04<00:00, 18.10it/s, loss=2.9286, hm=0.3668, paf=0.2108, cls=0.6933]
                                             
Epoch 30 Summary:
  Train - Total: 2.8276, HM: 0.3698, PAF: 0.1932, Cls: 0.6414
  Val   - Total: 2.8678, HM: 0.3707, PAF: 0.1985, Cls: 0.6588
  Metrics - Class Acc: 59.64%, PCK@0.2: 51.80%
  LR: 0.000181
  💾 Saved best model (val_loss: 2.8678)
Epoch 31: 100% 73/73 [00:04<00:00, 17.92it/s, loss=3.0712, hm=0.4114, paf=0.2091, cls=0.6717]

Epoch 31 - Train: 2.8190 (HM: 0.3693, PAF: 0.1915, Cls: 0.6391)  LR: 0.000180
Epoch 32: 100% 73/73 [00:04<00:00, 17.93it/s, loss=3.0894, hm=0.4244, paf=0.1733, cls=0.6969]
                                             
Epoch 32 Summary:
  Train - Total: 2.8166, HM: 0.3690, PAF: 0.1912, Cls: 0.6387
  Val   - Total: 2.8704, HM: 0.3738, PAF: 0.1989, Cls: 0.6516
  Metrics - Class Acc: 61.00%, PCK@0.2: 50.32%
  LR: 0.000178
Epoch 33: 100% 73/73 [00:04<00:00, 17.59it/s, loss=2.7066, hm=0.3516, paf=0.1658, cls=0.6458]

Epoch 33 - Train: 2.8081 (HM: 0.3678, PAF: 0.1914, Cls: 0.6361)  LR: 0.000177
Epoch 34: 100% 73/73 [00:03<00:00, 18.28it/s, loss=3.3281, hm=0.4602, paf=0.2274, cls=0.6883]
                                             
Epoch 34 Summary:
  Train - Total: 2.8227, HM: 0.3699, PAF: 0.1920, Cls: 0.6394
  Val   - Total: 2.8512, HM: 0.3681, PAF: 0.1995, Cls: 0.6533
  Metrics - Class Acc: 61.12%, PCK@0.2: 52.12%
  LR: 0.000176
  💾 Saved best model (val_loss: 2.8512)
Epoch 35: 100% 73/73 [00:04<00:00, 18.22it/s, loss=3.2513, hm=0.4361, paf=0.2050, cls=0.7313]

Epoch 35 - Train: 2.8016 (HM: 0.3675, PAF: 0.1882, Cls: 0.6368)  LR: 0.000174
Epoch 36: 100% 73/73 [00:04<00:00, 18.06it/s, loss=3.3151, hm=0.4607, paf=0.2427, cls=0.6580]
                                             
Epoch 36 Summary:
  Train - Total: 2.7988, HM: 0.3672, PAF: 0.1897, Cls: 0.6337
  Val   - Total: 2.8318, HM: 0.3678, PAF: 0.1957, Cls: 0.6460
  Metrics - Class Acc: 62.03%, PCK@0.2: 52.43%
  LR: 0.000173
  💾 Saved best model (val_loss: 2.8318)
Epoch 37: 100% 73/73 [00:03<00:00, 18.26it/s, loss=2.9826, hm=0.4064, paf=0.1939, cls=0.6460]

Epoch 37 - Train: 2.7950 (HM: 0.3664, PAF: 0.1887, Cls: 0.6346)  LR: 0.000172
Epoch 38: 100% 73/73 [00:03<00:00, 18.31it/s, loss=3.2376, hm=0.4170, paf=0.2320, cls=0.7370]
                                             
Epoch 38 Summary:
  Train - Total: 2.7977, HM: 0.3662, PAF: 0.1894, Cls: 0.6362
  Val   - Total: 2.8384, HM: 0.3683, PAF: 0.1971, Cls: 0.6473
  Metrics - Class Acc: 62.04%, PCK@0.2: 51.41%
  LR: 0.000170
Epoch 39: 100% 73/73 [00:03<00:00, 18.30it/s, loss=2.5916, hm=0.3126, paf=0.1774, cls=0.6577]

Epoch 39 - Train: 2.7940 (HM: 0.3655, PAF: 0.1893, Cls: 0.6356)  LR: 0.000169
Epoch 40: 100% 73/73 [00:03<00:00, 18.30it/s, loss=2.6520, hm=0.3346, paf=0.1762, cls=0.6408]
                                             
Epoch 40 Summary:
  Train - Total: 2.7800, HM: 0.3638, PAF: 0.1869, Cls: 0.6340
  Val   - Total: 2.8263, HM: 0.3665, PAF: 0.1942, Cls: 0.6479
  Metrics - Class Acc: 61.94%, PCK@0.2: 52.26%
  LR: 0.000167
  💾 Saved best model (val_loss: 2.8263)
Epoch 41: 100% 73/73 [00:04<00:00, 18.22it/s, loss=2.7678, hm=0.3690, paf=0.1876, cls=0.6110]

Epoch 41 - Train: 2.7744 (HM: 0.3637, PAF: 0.1867, Cls: 0.6308)  LR: 0.000166
Epoch 42: 100% 73/73 [00:03<00:00, 18.32it/s, loss=3.1430, hm=0.4277, paf=0.2157, cls=0.6671]
                                             
Epoch 42 Summary:
  Train - Total: 2.7691, HM: 0.3628, PAF: 0.1862, Cls: 0.6303
  Val   - Total: 2.8389, HM: 0.3662, PAF: 0.1988, Cls: 0.6511
  Metrics - Class Acc: 61.96%, PCK@0.2: 51.94%
  LR: 0.000164
Epoch 43: 100% 73/73 [00:04<00:00, 18.14it/s, loss=3.2234, hm=0.4399, paf=0.2234, cls=0.6780]

Epoch 43 - Train: 2.7744 (HM: 0.3641, PAF: 0.1871, Cls: 0.6291)  LR: 0.000162
Epoch 44: 100% 73/73 [00:04<00:00, 17.59it/s, loss=2.7978, hm=0.3878, paf=0.1766, cls=0.5956]
                                             
Epoch 44 Summary:
  Train - Total: 2.7676, HM: 0.3641, PAF: 0.1861, Cls: 0.6262
  Val   - Total: 2.8111, HM: 0.3649, PAF: 0.1907, Cls: 0.6467
  Metrics - Class Acc: 61.82%, PCK@0.2: 51.09%
  LR: 0.000161
  💾 Saved best model (val_loss: 2.8111)
Epoch 45: 100% 73/73 [00:04<00:00, 18.19it/s, loss=2.5449, hm=0.2844, paf=0.1825, cls=0.6948]

Epoch 45 - Train: 2.7589 (HM: 0.3613, PAF: 0.1862, Cls: 0.6276)  LR: 0.000159
Epoch 46: 100% 73/73 [00:04<00:00, 18.24it/s, loss=2.6984, hm=0.3396, paf=0.2163, cls=0.6049]
                                             
Epoch 46 Summary:
  Train - Total: 2.7538, HM: 0.3622, PAF: 0.1855, Cls: 0.6228
  Val   - Total: 2.8165, HM: 0.3649, PAF: 0.1927, Cls: 0.6477
  Metrics - Class Acc: 61.67%, PCK@0.2: 53.07%
  LR: 0.000157
Epoch 47: 100% 73/73 [00:03<00:00, 18.25it/s, loss=2.5453, hm=0.3152, paf=0.1876, cls=0.6063]

Epoch 47 - Train: 2.7569 (HM: 0.3613, PAF: 0.1862, Cls: 0.6264)  LR: 0.000156
Epoch 48: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.9886, hm=0.4428, paf=0.1533, cls=0.6071]
                                             
Epoch 48 Summary:
  Train - Total: 2.7533, HM: 0.3621, PAF: 0.1838, Cls: 0.6249
  Val   - Total: 2.8263, HM: 0.3632, PAF: 0.1963, Cls: 0.6538
  Metrics - Class Acc: 60.61%, PCK@0.2: 51.87%
  LR: 0.000154
Epoch 49: 100% 73/73 [00:04<00:00, 18.24it/s, loss=2.6792, hm=0.3662, paf=0.1498, cls=0.6100]

Epoch 49 - Train: 2.7539 (HM: 0.3618, PAF: 0.1841, Cls: 0.6256)  LR: 0.000152
Epoch 50: 100% 73/73 [00:04<00:00, 18.20it/s, loss=2.8604, hm=0.3759, paf=0.1885, cls=0.6531]
                                             
Epoch 50 Summary:
  Train - Total: 2.7461, HM: 0.3603, PAF: 0.1836, Cls: 0.6252
  Val   - Total: 2.8136, HM: 0.3629, PAF: 0.1919, Cls: 0.6520
  Metrics - Class Acc: 61.21%, PCK@0.2: 52.82%
  LR: 0.000150
Epoch 51: 100% 73/73 [00:04<00:00, 18.23it/s, loss=2.7242, hm=0.3781, paf=0.1821, cls=0.5651]

Epoch 51 - Train: 2.7438 (HM: 0.3606, PAF: 0.1842, Cls: 0.6219)  LR: 0.000148
Epoch 52: 100% 73/73 [00:04<00:00, 18.06it/s, loss=2.7002, hm=0.3449, paf=0.1527, cls=0.6768]
                                             
Epoch 52 Summary:
  Train - Total: 2.7333, HM: 0.3598, PAF: 0.1833, Cls: 0.6185
  Val   - Total: 2.8038, HM: 0.3622, PAF: 0.1911, Cls: 0.6485
  Metrics - Class Acc: 61.42%, PCK@0.2: 52.96%
  LR: 0.000147
  💾 Saved best model (val_loss: 2.8038)
Epoch 53: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.3069, hm=0.2861, paf=0.1324, cls=0.5985]

Epoch 53 - Train: 2.7343 (HM: 0.3594, PAF: 0.1827, Cls: 0.6208)  LR: 0.000145
Epoch 54: 100% 73/73 [00:04<00:00, 17.98it/s, loss=2.9468, hm=0.3993, paf=0.1826, cls=0.6562]
                                             
Epoch 54 Summary:
  Train - Total: 2.7341, HM: 0.3595, PAF: 0.1828, Cls: 0.6203
  Val   - Total: 2.7994, HM: 0.3624, PAF: 0.1915, Cls: 0.6446
  Metrics - Class Acc: 62.23%, PCK@0.2: 52.08%
  LR: 0.000143
  💾 Saved best model (val_loss: 2.7994)
Epoch 55: 100% 73/73 [00:04<00:00, 17.53it/s, loss=2.4463, hm=0.2921, paf=0.1440, cls=0.6599]

Epoch 55 - Train: 2.7263 (HM: 0.3583, PAF: 0.1809, Cls: 0.6208)  LR: 0.000141
Epoch 56: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.5583, hm=0.3370, paf=0.1532, cls=0.6026]
                                             
Epoch 56 Summary:
  Train - Total: 2.7372, HM: 0.3608, PAF: 0.1820, Cls: 0.6201
  Val   - Total: 2.8010, HM: 0.3618, PAF: 0.1934, Cls: 0.6446
  Metrics - Class Acc: 61.97%, PCK@0.2: 51.87%
  LR: 0.000139
Epoch 57: 100% 73/73 [00:04<00:00, 18.01it/s, loss=2.6354, hm=0.3676, paf=0.1455, cls=0.5826]

Epoch 57 - Train: 2.7291 (HM: 0.3597, PAF: 0.1817, Cls: 0.6180)  LR: 0.000137
Epoch 58: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.5046, hm=0.3026, paf=0.1754, cls=0.6290]
                                             
Epoch 58 Summary:
  Train - Total: 2.7222, HM: 0.3574, PAF: 0.1820, Cls: 0.6191
  Val   - Total: 2.7727, HM: 0.3607, PAF: 0.1886, Cls: 0.6351
  Metrics - Class Acc: 63.75%, PCK@0.2: 52.12%
  LR: 0.000135
  💾 Saved best model (val_loss: 2.7727)
Epoch 59: 100% 73/73 [00:04<00:00, 18.06it/s, loss=2.9055, hm=0.3959, paf=0.1667, cls=0.6590]

Epoch 59 - Train: 2.7232 (HM: 0.3586, PAF: 0.1816, Cls: 0.6169)  LR: 0.000133
Epoch 60: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.5762, hm=0.3113, paf=0.1901, cls=0.6339]
                                             
Epoch 60 Summary:
  Train - Total: 2.7196, HM: 0.3578, PAF: 0.1815, Cls: 0.6170
  Val   - Total: 2.7676, HM: 0.3593, PAF: 0.1892, Cls: 0.6345
  Metrics - Class Acc: 63.26%, PCK@0.2: 53.46%
  LR: 0.000131
  💾 Saved best model (val_loss: 2.7676)
Epoch 61: 100% 73/73 [00:04<00:00, 17.98it/s, loss=2.9703, hm=0.3920, paf=0.1917, cls=0.6791]

Epoch 61 - Train: 2.7243 (HM: 0.3587, PAF: 0.1817, Cls: 0.6174)  LR: 0.000129
Epoch 62: 100% 73/73 [00:04<00:00, 18.02it/s, loss=2.7295, hm=0.3530, paf=0.1816, cls=0.6363]
                                             
Epoch 62 Summary:
  Train - Total: 2.7124, HM: 0.3576, PAF: 0.1796, Cls: 0.6154
  Val   - Total: 2.7655, HM: 0.3588, PAF: 0.1887, Cls: 0.6351
  Metrics - Class Acc: 63.20%, PCK@0.2: 52.93%
  LR: 0.000127
  💾 Saved best model (val_loss: 2.7655)
Epoch 63: 100% 73/73 [00:04<00:00, 18.11it/s, loss=2.7978, hm=0.3496, paf=0.1923, cls=0.6766]

Epoch 63 - Train: 2.7200 (HM: 0.3592, PAF: 0.1808, Cls: 0.6145)  LR: 0.000125
Epoch 64: 100% 73/73 [00:03<00:00, 18.30it/s, loss=2.9710, hm=0.3887, paf=0.1997, cls=0.6780]
                                             
Epoch 64 Summary:
  Train - Total: 2.7206, HM: 0.3579, PAF: 0.1816, Cls: 0.6171
  Val   - Total: 2.7875, HM: 0.3619, PAF: 0.1905, Cls: 0.6394
  Metrics - Class Acc: 63.08%, PCK@0.2: 52.86%
  LR: 0.000123
Epoch 65: 100% 73/73 [00:04<00:00, 18.05it/s, loss=3.0839, hm=0.3968, paf=0.1994, cls=0.7320]

Epoch 65 - Train: 2.7187 (HM: 0.3582, PAF: 0.1809, Cls: 0.6162)  LR: 0.000121
Epoch 66: 100% 73/73 [00:04<00:00, 18.02it/s, loss=2.7320, hm=0.3624, paf=0.1554, cls=0.6478]
                                             
Epoch 66 Summary:
  Train - Total: 2.6991, HM: 0.3556, PAF: 0.1790, Cls: 0.6125
  Val   - Total: 2.7667, HM: 0.3584, PAF: 0.1877, Cls: 0.6386
  Metrics - Class Acc: 62.57%, PCK@0.2: 52.75%
  LR: 0.000119
Epoch 67: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.3855, hm=0.2994, paf=0.1792, cls=0.5530]

Epoch 67 - Train: 2.6959 (HM: 0.3557, PAF: 0.1786, Cls: 0.6105)  LR: 0.000117
Epoch 68: 100% 73/73 [00:04<00:00, 18.22it/s, loss=2.7808, hm=0.3368, paf=0.2021, cls=0.6862]
                                             
Epoch 68 Summary:
  Train - Total: 2.6992, HM: 0.3554, PAF: 0.1794, Cls: 0.6127
  Val   - Total: 2.7667, HM: 0.3594, PAF: 0.1878, Cls: 0.6357
  Metrics - Class Acc: 63.36%, PCK@0.2: 52.79%
  LR: 0.000115
Epoch 69: 100% 73/73 [00:04<00:00, 17.97it/s, loss=3.0461, hm=0.3696, paf=0.2254, cls=0.7446]

Epoch 69 - Train: 2.7062 (HM: 0.3564, PAF: 0.1800, Cls: 0.6136)  LR: 0.000113
Epoch 70: 100% 73/73 [00:04<00:00, 18.05it/s, loss=2.9268, hm=0.3812, paf=0.2222, cls=0.6386]
                                             
Epoch 70 Summary:
  Train - Total: 2.6992, HM: 0.3559, PAF: 0.1790, Cls: 0.6117
  Val   - Total: 2.7745, HM: 0.3604, PAF: 0.1884, Cls: 0.6374
  Metrics - Class Acc: 62.99%, PCK@0.2: 52.96%
  LR: 0.000111
Epoch 71: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.7905, hm=0.3826, paf=0.1624, cls=0.6236]

Epoch 71 - Train: 2.6991 (HM: 0.3567, PAF: 0.1790, Cls: 0.6096)  LR: 0.000109
Epoch 72: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.5399, hm=0.3212, paf=0.1767, cls=0.6011]
                                             
Epoch 72 Summary:
  Train - Total: 2.6997, HM: 0.3561, PAF: 0.1785, Cls: 0.6121
  Val   - Total: 2.7548, HM: 0.3581, PAF: 0.1862, Cls: 0.6333
  Metrics - Class Acc: 63.24%, PCK@0.2: 53.42%
  LR: 0.000107
  💾 Saved best model (val_loss: 2.7548)
Epoch 73: 100% 73/73 [00:04<00:00, 18.08it/s, loss=2.3843, hm=0.3215, paf=0.1639, cls=0.5137]

Epoch 73 - Train: 2.6889 (HM: 0.3545, PAF: 0.1769, Cls: 0.6113)  LR: 0.000105
Epoch 74: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.6555, hm=0.3579, paf=0.1977, cls=0.5525]
                                             
Epoch 74 Summary:
  Train - Total: 2.6987, HM: 0.3565, PAF: 0.1789, Cls: 0.6098
  Val   - Total: 2.7451, HM: 0.3576, PAF: 0.1863, Cls: 0.6281
  Metrics - Class Acc: 64.31%, PCK@0.2: 54.23%
  LR: 0.000103
  💾 Saved best model (val_loss: 2.7451)
Epoch 75: 100% 73/73 [00:04<00:00, 18.23it/s, loss=2.8802, hm=0.3735, paf=0.2253, cls=0.6236]

Epoch 75 - Train: 2.6976 (HM: 0.3558, PAF: 0.1789, Cls: 0.6110)  LR: 0.000100
Epoch 76: 100% 73/73 [00:04<00:00, 17.99it/s, loss=2.9037, hm=0.4110, paf=0.1739, cls=0.6079]
                                             
Epoch 76 Summary:
  Train - Total: 2.6868, HM: 0.3551, PAF: 0.1780, Cls: 0.6069
  Val   - Total: 2.7610, HM: 0.3575, PAF: 0.1895, Cls: 0.6347
  Metrics - Class Acc: 63.24%, PCK@0.2: 53.88%
  LR: 0.000098
Epoch 77: 100% 73/73 [00:04<00:00, 18.17it/s, loss=3.1324, hm=0.4351, paf=0.2287, cls=0.6232]

Epoch 77 - Train: 2.6831 (HM: 0.3547, PAF: 0.1772, Cls: 0.6068)  LR: 0.000096
Epoch 78: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.3089, hm=0.3115, paf=0.1284, cls=0.5375]
                                             
Epoch 78 Summary:
  Train - Total: 2.6870, HM: 0.3549, PAF: 0.1770, Cls: 0.6090
  Val   - Total: 2.7319, HM: 0.3557, PAF: 0.1848, Cls: 0.6262
  Metrics - Class Acc: 64.42%, PCK@0.2: 54.20%
  LR: 0.000094
  💾 Saved best model (val_loss: 2.7319)
Epoch 79: 100% 73/73 [00:04<00:00, 18.17it/s, loss=3.1171, hm=0.4357, paf=0.2185, cls=0.6249]

Epoch 79 - Train: 2.6883 (HM: 0.3553, PAF: 0.1775, Cls: 0.6079)  LR: 0.000092
Epoch 80: 100% 73/73 [00:04<00:00, 18.24it/s, loss=2.4853, hm=0.3184, paf=0.1410, cls=0.6197]
                                             
Epoch 80 Summary:
  Train - Total: 2.6828, HM: 0.3547, PAF: 0.1764, Cls: 0.6077
  Val   - Total: 2.7547, HM: 0.3587, PAF: 0.1869, Cls: 0.6308
  Metrics - Class Acc: 63.86%, PCK@0.2: 53.49%
  LR: 0.000090
Epoch 81: 100% 73/73 [00:04<00:00, 17.58it/s, loss=2.9521, hm=0.3692, paf=0.2283, cls=0.6791]

Epoch 81 - Train: 2.6902 (HM: 0.3550, PAF: 0.1771, Cls: 0.6107)  LR: 0.000088
Epoch 82: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.4390, hm=0.3227, paf=0.1605, cls=0.5515]
                                             
Epoch 82 Summary:
  Train - Total: 2.6769, HM: 0.3540, PAF: 0.1765, Cls: 0.6051
  Val   - Total: 2.7488, HM: 0.3576, PAF: 0.1853, Cls: 0.6318
  Metrics - Class Acc: 64.15%, PCK@0.2: 54.06%
  LR: 0.000086
Epoch 83: 100% 73/73 [00:04<00:00, 18.21it/s, loss=2.6508, hm=0.3152, paf=0.1878, cls=0.6762]

Epoch 83 - Train: 2.6806 (HM: 0.3534, PAF: 0.1768, Cls: 0.6089)  LR: 0.000084
Epoch 84: 100% 73/73 [00:04<00:00, 18.22it/s, loss=2.6874, hm=0.3593, paf=0.1682, cls=0.6090]
                                             
Epoch 84 Summary:
  Train - Total: 2.6815, HM: 0.3539, PAF: 0.1770, Cls: 0.6078
  Val   - Total: 2.7381, HM: 0.3562, PAF: 0.1842, Cls: 0.6298
  Metrics - Class Acc: 64.11%, PCK@0.2: 53.28%
  LR: 0.000082
Epoch 85: 100% 73/73 [00:04<00:00, 18.18it/s, loss=2.5975, hm=0.3266, paf=0.2135, cls=0.5761]

Epoch 85 - Train: 2.6740 (HM: 0.3542, PAF: 0.1758, Cls: 0.6037)  LR: 0.000080
Epoch 86: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.6816, hm=0.3251, paf=0.2067, cls=0.6452]
                                             
Epoch 86 Summary:
  Train - Total: 2.6680, HM: 0.3532, PAF: 0.1757, Cls: 0.6025
  Val   - Total: 2.7442, HM: 0.3574, PAF: 0.1858, Cls: 0.6285
  Metrics - Class Acc: 64.25%, PCK@0.2: 53.78%
  LR: 0.000078
Epoch 87: 100% 73/73 [00:04<00:00, 17.93it/s, loss=2.9965, hm=0.3711, paf=0.2057, cls=0.7339]

Epoch 87 - Train: 2.6842 (HM: 0.3547, PAF: 0.1774, Cls: 0.6071)  LR: 0.000076
Epoch 88: 100% 73/73 [00:04<00:00, 18.14it/s, loss=3.2111, hm=0.4439, paf=0.2453, cls=0.6300]
                                             
Epoch 88 Summary:
  Train - Total: 2.6845, HM: 0.3544, PAF: 0.1765, Cls: 0.6092
  Val   - Total: 2.7394, HM: 0.3557, PAF: 0.1863, Cls: 0.6292
  Metrics - Class Acc: 64.05%, PCK@0.2: 53.85%
  LR: 0.000074
Epoch 89: 100% 73/73 [00:03<00:00, 18.25it/s, loss=2.9064, hm=0.3966, paf=0.1568, cls=0.6709]

Epoch 89 - Train: 2.6711 (HM: 0.3533, PAF: 0.1758, Cls: 0.6041)  LR: 0.000072
Epoch 90: 100% 73/73 [00:04<00:00, 18.15it/s, loss=2.7523, hm=0.3762, paf=0.1830, cls=0.5876]
                                             
Epoch 90 Summary:
  Train - Total: 2.6669, HM: 0.3521, PAF: 0.1758, Cls: 0.6047
  Val   - Total: 2.7412, HM: 0.3564, PAF: 0.1867, Cls: 0.6281
  Metrics - Class Acc: 64.46%, PCK@0.2: 53.99%
  LR: 0.000070
Epoch 91: 100% 73/73 [00:04<00:00, 18.20it/s, loss=2.5201, hm=0.3475, paf=0.1708, cls=0.5257]

Epoch 91 - Train: 2.6570 (HM: 0.3514, PAF: 0.1743, Cls: 0.6017)  LR: 0.000068
Epoch 92: 100% 73/73 [00:04<00:00, 17.96it/s, loss=2.5031, hm=0.3191, paf=0.1473, cls=0.6214]
                                             
Epoch 92 Summary:
  Train - Total: 2.6644, HM: 0.3515, PAF: 0.1748, Cls: 0.6058
  Val   - Total: 2.7342, HM: 0.3562, PAF: 0.1853, Cls: 0.6258
  Metrics - Class Acc: 64.79%, PCK@0.2: 53.46%
  LR: 0.000066
Epoch 93: 100% 73/73 [00:04<00:00, 18.14it/s, loss=3.1222, hm=0.4051, paf=0.2553, cls=0.6609]

Epoch 93 - Train: 2.6697 (HM: 0.3534, PAF: 0.1763, Cls: 0.6023)  LR: 0.000064
Epoch 94: 100% 73/73 [00:04<00:00, 18.13it/s, loss=3.3148, hm=0.4885, paf=0.1825, cls=0.6640]
                                             
Epoch 94 Summary:
  Train - Total: 2.6691, HM: 0.3538, PAF: 0.1745, Cls: 0.6032
  Val   - Total: 2.7411, HM: 0.3558, PAF: 0.1862, Cls: 0.6304
  Metrics - Class Acc: 63.69%, PCK@0.2: 53.49%
  LR: 0.000062
Epoch 95: 100% 73/73 [00:04<00:00, 18.18it/s, loss=2.7274, hm=0.3637, paf=0.1930, cls=0.5910]

Epoch 95 - Train: 2.6710 (HM: 0.3537, PAF: 0.1763, Cls: 0.6024)  LR: 0.000060
Epoch 96: 100% 73/73 [00:04<00:00, 17.65it/s, loss=2.9979, hm=0.4143, paf=0.1875, cls=0.6437]
                                             
Epoch 96 Summary:
  Train - Total: 2.6723, HM: 0.3533, PAF: 0.1757, Cls: 0.6052
  Val   - Total: 2.7393, HM: 0.3558, PAF: 0.1861, Cls: 0.6291
  Metrics - Class Acc: 64.42%, PCK@0.2: 54.59%
  LR: 0.000058
Epoch 97: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.5218, hm=0.3217, paf=0.1576, cls=0.6132]

Epoch 97 - Train: 2.6623 (HM: 0.3517, PAF: 0.1748, Cls: 0.6041)  LR: 0.000056
Epoch 98: 100% 73/73 [00:04<00:00, 18.19it/s, loss=2.4430, hm=0.3215, paf=0.1519, cls=0.5689]
                                             
Epoch 98 Summary:
  Train - Total: 2.6584, HM: 0.3522, PAF: 0.1742, Cls: 0.6009
  Val   - Total: 2.7381, HM: 0.3555, PAF: 0.1855, Cls: 0.6301
  Metrics - Class Acc: 63.99%, PCK@0.2: 53.71%
  LR: 0.000054
Epoch 99: 100% 73/73 [00:04<00:00, 18.20it/s, loss=2.7367, hm=0.3573, paf=0.1712, cls=0.6433]

Epoch 99 - Train: 2.6641 (HM: 0.3524, PAF: 0.1751, Cls: 0.6027)  LR: 0.000053
Epoch 100: 100% 73/73 [00:04<00:00, 18.10it/s, loss=3.2044, hm=0.4203, paf=0.2475, cls=0.6854]
                                             
Epoch 100 Summary:
  Train - Total: 2.6690, HM: 0.3535, PAF: 0.1760, Cls: 0.6021
  Val   - Total: 2.7306, HM: 0.3548, PAF: 0.1848, Cls: 0.6277
  Metrics - Class Acc: 64.39%, PCK@0.2: 54.16%
  LR: 0.000051
  💾 Saved best model (val_loss: 2.7306)
Epoch 101: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.5571, hm=0.3125, paf=0.1704, cls=0.6443]

Epoch 101 - Train: 2.6533 (HM: 0.3510, PAF: 0.1748, Cls: 0.5999)  LR: 0.000049
Epoch 102: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.8324, hm=0.3847, paf=0.1931, cls=0.6049]
                                             
Epoch 102 Summary:
  Train - Total: 2.6474, HM: 0.3506, PAF: 0.1739, Cls: 0.5980
  Val   - Total: 2.7290, HM: 0.3544, PAF: 0.1851, Cls: 0.6274
  Metrics - Class Acc: 64.32%, PCK@0.2: 54.09%
  LR: 0.000047
  💾 Saved best model (val_loss: 2.7290)
Epoch 103: 100% 73/73 [00:04<00:00, 18.18it/s, loss=2.6665, hm=0.3437, paf=0.1796, cls=0.6216]

Epoch 103 - Train: 2.6554 (HM: 0.3518, PAF: 0.1752, Cls: 0.5985)  LR: 0.000045
Epoch 104: 100% 73/73 [00:04<00:00, 18.20it/s, loss=2.4338, hm=0.3161, paf=0.1574, cls=0.5698]
                                             
Epoch 104 Summary:
  Train - Total: 2.6591, HM: 0.3528, PAF: 0.1748, Cls: 0.5989
  Val   - Total: 2.7263, HM: 0.3545, PAF: 0.1853, Cls: 0.6252
  Metrics - Class Acc: 64.94%, PCK@0.2: 54.55%
  LR: 0.000044
  💾 Saved best model (val_loss: 2.7263)
Epoch 105: 100% 73/73 [00:04<00:00, 18.08it/s, loss=2.6419, hm=0.3341, paf=0.2056, cls=0.5961]

Epoch 105 - Train: 2.6578 (HM: 0.3515, PAF: 0.1744, Cls: 0.6021)  LR: 0.000042
Epoch 106: 100% 73/73 [00:04<00:00, 18.18it/s, loss=2.5739, hm=0.3274, paf=0.1977, cls=0.5792]
                                             
Epoch 106 Summary:
  Train - Total: 2.6576, HM: 0.3518, PAF: 0.1746, Cls: 0.6008
  Val   - Total: 2.7290, HM: 0.3555, PAF: 0.1841, Cls: 0.6257
  Metrics - Class Acc: 64.75%, PCK@0.2: 53.46%
  LR: 0.000040
Epoch 107: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.6760, hm=0.3454, paf=0.2125, cls=0.5795]

Epoch 107 - Train: 2.6489 (HM: 0.3510, PAF: 0.1739, Cls: 0.5980)  LR: 0.000039
Epoch 108: 100% 73/73 [00:04<00:00, 18.09it/s, loss=2.7121, hm=0.3190, paf=0.1943, cls=0.6984]
                                             
Epoch 108 Summary:
  Train - Total: 2.6503, HM: 0.3497, PAF: 0.1746, Cls: 0.6017
  Val   - Total: 2.7393, HM: 0.3551, PAF: 0.1864, Cls: 0.6308
  Metrics - Class Acc: 64.00%, PCK@0.2: 53.74%
  LR: 0.000037
Epoch 109: 100% 73/73 [00:04<00:00, 17.98it/s, loss=2.9804, hm=0.4126, paf=0.1880, cls=0.6361]

Epoch 109 - Train: 2.6609 (HM: 0.3529, PAF: 0.1738, Cls: 0.6010)  LR: 0.000035
Epoch 110: 100% 73/73 [00:04<00:00, 18.10it/s, loss=2.9678, hm=0.3513, paf=0.1743, cls=0.8092]
                                             
Epoch 110 Summary:
  Train - Total: 2.6599, HM: 0.3513, PAF: 0.1743, Cls: 0.6040
  Val   - Total: 2.7260, HM: 0.3546, PAF: 0.1855, Cls: 0.6245
  Metrics - Class Acc: 64.99%, PCK@0.2: 53.67%
  LR: 0.000034
  💾 Saved best model (val_loss: 2.7260)
Epoch 111: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.2351, hm=0.2814, paf=0.1572, cls=0.5302]

Epoch 111 - Train: 2.6532 (HM: 0.3506, PAF: 0.1743, Cls: 0.6017)  LR: 0.000032
Epoch 112: 100% 73/73 [00:04<00:00, 18.22it/s, loss=2.8565, hm=0.3632, paf=0.2277, cls=0.6321]
                                             
Epoch 112 Summary:
  Train - Total: 2.6546, HM: 0.3518, PAF: 0.1743, Cls: 0.5992
  Val   - Total: 2.7237, HM: 0.3544, PAF: 0.1845, Cls: 0.6247
  Metrics - Class Acc: 64.83%, PCK@0.2: 53.71%
  LR: 0.000031
  💾 Saved best model (val_loss: 2.7237)
Epoch 113: 100% 73/73 [00:04<00:00, 18.04it/s, loss=2.5586, hm=0.3557, paf=0.1825, cls=0.5139]

Epoch 113 - Train: 2.6560 (HM: 0.3526, PAF: 0.1745, Cls: 0.5978)  LR: 0.000029
Epoch 114: 100% 73/73 [00:04<00:00, 18.22it/s, loss=3.1345, hm=0.4454, paf=0.1948, cls=0.6421]
                                             
Epoch 114 Summary:
  Train - Total: 2.6582, HM: 0.3526, PAF: 0.1743, Cls: 0.5995
  Val   - Total: 2.7298, HM: 0.3544, PAF: 0.1855, Cls: 0.6275
  Metrics - Class Acc: 64.27%, PCK@0.2: 53.32%
  LR: 0.000028
Epoch 115: 100% 73/73 [00:04<00:00, 18.19it/s, loss=2.6430, hm=0.3195, paf=0.1964, cls=0.6482]

Epoch 115 - Train: 2.6522 (HM: 0.3520, PAF: 0.1731, Cls: 0.5987)  LR: 0.000027
Epoch 116: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.9439, hm=0.3867, paf=0.2074, cls=0.6547]
                                             
Epoch 116 Summary:
  Train - Total: 2.6555, HM: 0.3524, PAF: 0.1740, Cls: 0.5986
  Val   - Total: 2.7277, HM: 0.3546, PAF: 0.1846, Cls: 0.6266
  Metrics - Class Acc: 64.52%, PCK@0.2: 54.38%
  LR: 0.000025
Epoch 117: 100% 73/73 [00:04<00:00, 18.14it/s, loss=2.6880, hm=0.3307, paf=0.2101, cls=0.6300]

Epoch 117 - Train: 2.6401 (HM: 0.3504, PAF: 0.1723, Cls: 0.5959)  LR: 0.000024
Epoch 118: 100% 73/73 [00:04<00:00, 18.15it/s, loss=3.0266, hm=0.4030, paf=0.2171, cls=0.6537]
                                             
Epoch 118 Summary:
  Train - Total: 2.6523, HM: 0.3519, PAF: 0.1744, Cls: 0.5972
  Val   - Total: 2.7296, HM: 0.3550, PAF: 0.1849, Cls: 0.6265
  Metrics - Class Acc: 64.48%, PCK@0.2: 54.87%
  LR: 0.000023
Epoch 119: 100% 73/73 [00:04<00:00, 18.24it/s, loss=3.3797, hm=0.4733, paf=0.2049, cls=0.7176]

Epoch 119 - Train: 2.6480 (HM: 0.3511, PAF: 0.1730, Cls: 0.5983)  LR: 0.000021
Epoch 120: 100% 73/73 [00:04<00:00, 18.10it/s, loss=2.8281, hm=0.3593, paf=0.1786, cls=0.6893]
                                             
Epoch 120 Summary:
  Train - Total: 2.6492, HM: 0.3511, PAF: 0.1733, Cls: 0.5987
  Val   - Total: 2.7214, HM: 0.3540, PAF: 0.1842, Cls: 0.6247
  Metrics - Class Acc: 64.82%, PCK@0.2: 54.09%
  LR: 0.000020
  💾 Saved best model (val_loss: 2.7214)
Epoch 121: 100% 73/73 [00:04<00:00, 18.22it/s, loss=2.7711, hm=0.3699, paf=0.2022, cls=0.5915]

Epoch 121 - Train: 2.6430 (HM: 0.3505, PAF: 0.1737, Cls: 0.5957)  LR: 0.000019
Epoch 122: 100% 73/73 [00:04<00:00, 18.20it/s, loss=2.9111, hm=0.4357, paf=0.1326, cls=0.6020]
                                             
Epoch 122 Summary:
  Train - Total: 2.6427, HM: 0.3515, PAF: 0.1714, Cls: 0.5960
  Val   - Total: 2.7233, HM: 0.3538, PAF: 0.1843, Cls: 0.6264
  Metrics - Class Acc: 64.56%, PCK@0.2: 53.81%
  LR: 0.000018
Epoch 123: 100% 73/73 [00:03<00:00, 18.30it/s, loss=2.5529, hm=0.3266, paf=0.1779, cls=0.5940]

Epoch 123 - Train: 2.6410 (HM: 0.3494, PAF: 0.1732, Cls: 0.5980)  LR: 0.000016
Epoch 124: 100% 73/73 [00:03<00:00, 18.26it/s, loss=2.3545, hm=0.3130, paf=0.1519, cls=0.5325]
                                             
Epoch 124 Summary:
  Train - Total: 2.6343, HM: 0.3495, PAF: 0.1722, Cls: 0.5947
  Val   - Total: 2.7214, HM: 0.3544, PAF: 0.1846, Cls: 0.6230
  Metrics - Class Acc: 65.16%, PCK@0.2: 54.30%
  LR: 0.000015
  💾 Saved best model (val_loss: 2.7214)
Epoch 125: 100% 73/73 [00:04<00:00, 17.74it/s, loss=2.4609, hm=0.3304, paf=0.1664, cls=0.5377]

Epoch 125 - Train: 2.6484 (HM: 0.3513, PAF: 0.1724, Cls: 0.5989)  LR: 0.000014
Epoch 126: 100% 73/73 [00:03<00:00, 18.34it/s, loss=3.0924, hm=0.3987, paf=0.2246, cls=0.6988]
                                             
Epoch 126 Summary:
  Train - Total: 2.6512, HM: 0.3506, PAF: 0.1742, Cls: 0.6002
  Val   - Total: 2.7237, HM: 0.3534, PAF: 0.1841, Cls: 0.6279
  Metrics - Class Acc: 64.33%, PCK@0.2: 54.02%
  LR: 0.000013
Epoch 127: 100% 73/73 [00:03<00:00, 18.40it/s, loss=2.9348, hm=0.3648, paf=0.2103, cls=0.7034]

Epoch 127 - Train: 2.6487 (HM: 0.3511, PAF: 0.1734, Cls: 0.5984)  LR: 0.000012
Epoch 128: 100% 73/73 [00:03<00:00, 18.38it/s, loss=2.6864, hm=0.3633, paf=0.1978, cls=0.5583]
                                             
Epoch 128 Summary:
  Train - Total: 2.6418, HM: 0.3505, PAF: 0.1731, Cls: 0.5959
  Val   - Total: 2.7213, HM: 0.3536, PAF: 0.1841, Cls: 0.6257
  Metrics - Class Acc: 64.53%, PCK@0.2: 54.87%
  LR: 0.000011
  💾 Saved best model (val_loss: 2.7213)
Epoch 129: 100% 73/73 [00:04<00:00, 18.19it/s, loss=2.4744, hm=0.3211, paf=0.1591, cls=0.5811]

Epoch 129 - Train: 2.6479 (HM: 0.3515, PAF: 0.1728, Cls: 0.5974)  LR: 0.000010
Epoch 130: 100% 73/73 [00:03<00:00, 18.25it/s, loss=2.7741, hm=0.3534, paf=0.1748, cls=0.6740]
                                             
Epoch 130 Summary:
  Train - Total: 2.6517, HM: 0.3512, PAF: 0.1736, Cls: 0.5999
  Val   - Total: 2.7245, HM: 0.3537, PAF: 0.1848, Cls: 0.6268
  Metrics - Class Acc: 64.41%, PCK@0.2: 53.81%
  LR: 0.000010
Epoch 131: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.6707, hm=0.3450, paf=0.1683, cls=0.6361]

Epoch 131 - Train: 2.6462 (HM: 0.3512, PAF: 0.1731, Cls: 0.5969)  LR: 0.000009
Epoch 132: 100% 73/73 [00:04<00:00, 18.08it/s, loss=3.1858, hm=0.4490, paf=0.2032, cls=0.6555]
                                             
Epoch 132 Summary:
  Train - Total: 2.6473, HM: 0.3508, PAF: 0.1733, Cls: 0.5983
  Val   - Total: 2.7200, HM: 0.3538, PAF: 0.1846, Cls: 0.6238
  Metrics - Class Acc: 64.87%, PCK@0.2: 54.62%
  LR: 0.000008
  💾 Saved best model (val_loss: 2.7200)
Epoch 133: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.7314, hm=0.3423, paf=0.1881, cls=0.6574]

Epoch 133 - Train: 2.6370 (HM: 0.3492, PAF: 0.1726, Cls: 0.5967)  LR: 0.000007
Epoch 134: 100% 73/73 [00:04<00:00, 18.08it/s, loss=2.7890, hm=0.3951, paf=0.1825, cls=0.5624]
                                             
Epoch 134 Summary:
  Train - Total: 2.6438, HM: 0.3506, PAF: 0.1738, Cls: 0.5957
  Val   - Total: 2.7189, HM: 0.3541, PAF: 0.1845, Cls: 0.6222
  Metrics - Class Acc: 65.19%, PCK@0.2: 54.73%
  LR: 0.000007
  💾 Saved best model (val_loss: 2.7189)
Epoch 135: 100% 73/73 [00:04<00:00, 18.07it/s, loss=2.7897, hm=0.3849, paf=0.1893, cls=0.5811]

Epoch 135 - Train: 2.6459 (HM: 0.3507, PAF: 0.1724, Cls: 0.5990)  LR: 0.000006
Epoch 136: 100% 73/73 [00:04<00:00, 17.56it/s, loss=2.6119, hm=0.3348, paf=0.1940, cls=0.5900]
                                             
Epoch 136 Summary:
  Train - Total: 2.6474, HM: 0.3514, PAF: 0.1726, Cls: 0.5978
  Val   - Total: 2.7229, HM: 0.3538, PAF: 0.1842, Cls: 0.6263
  Metrics - Class Acc: 64.42%, PCK@0.2: 54.80%
  LR: 0.000005
Epoch 137: 100% 73/73 [00:04<00:00, 18.05it/s, loss=2.4450, hm=0.3183, paf=0.1395, cls=0.5952]

Epoch 137 - Train: 2.6337 (HM: 0.3492, PAF: 0.1722, Cls: 0.5951)  LR: 0.000005
Epoch 138: 100% 73/73 [00:04<00:00, 17.98it/s, loss=2.9542, hm=0.3588, paf=0.2189, cls=0.7209]
                                             
Epoch 138 Summary:
  Train - Total: 2.6476, HM: 0.3510, PAF: 0.1740, Cls: 0.5970
  Val   - Total: 2.7214, HM: 0.3536, PAF: 0.1847, Cls: 0.6250
  Metrics - Class Acc: 64.61%, PCK@0.2: 54.90%
  LR: 0.000004
Epoch 139: 100% 73/73 [00:04<00:00, 18.02it/s, loss=2.9415, hm=0.3680, paf=0.2395, cls=0.6603]

Epoch 139 - Train: 2.6516 (HM: 0.3518, PAF: 0.1729, Cls: 0.5991)  LR: 0.000004
Epoch 140: 100% 73/73 [00:04<00:00, 18.13it/s, loss=2.4103, hm=0.3246, paf=0.1580, cls=0.5305]
                                             
Epoch 140 Summary:
  Train - Total: 2.6406, HM: 0.3500, PAF: 0.1723, Cls: 0.5973
  Val   - Total: 2.7229, HM: 0.3543, PAF: 0.1842, Cls: 0.6247
  Metrics - Class Acc: 64.62%, PCK@0.2: 54.34%
  LR: 0.000003
Epoch 141: 100% 73/73 [00:04<00:00, 18.11it/s, loss=2.5407, hm=0.3292, paf=0.1667, cls=0.5937]

Epoch 141 - Train: 2.6394 (HM: 0.3501, PAF: 0.1726, Cls: 0.5958)  LR: 0.000003
Epoch 142: 100% 73/73 [00:04<00:00, 18.16it/s, loss=2.3202, hm=0.2910, paf=0.1609, cls=0.5564]
                                             
Epoch 142 Summary:
  Train - Total: 2.6375, HM: 0.3499, PAF: 0.1726, Cls: 0.5951
  Val   - Total: 2.7173, HM: 0.3535, PAF: 0.1841, Cls: 0.6235
  Metrics - Class Acc: 64.95%, PCK@0.2: 54.41%
  LR: 0.000002
  💾 Saved best model (val_loss: 2.7173)
Epoch 143: 100% 73/73 [00:04<00:00, 18.00it/s, loss=2.8262, hm=0.3852, paf=0.1619, cls=0.6411]

Epoch 143 - Train: 2.6493 (HM: 0.3511, PAF: 0.1729, Cls: 0.5992)  LR: 0.000002
Epoch 144: 100% 73/73 [00:04<00:00, 18.00it/s, loss=3.0918, hm=0.3943, paf=0.2108, cls=0.7286]
                                             
Epoch 144 Summary:
  Train - Total: 2.6416, HM: 0.3505, PAF: 0.1731, Cls: 0.5958
  Val   - Total: 2.7193, HM: 0.3537, PAF: 0.1840, Cls: 0.6244
  Metrics - Class Acc: 64.91%, PCK@0.2: 53.99%
  LR: 0.000002
Epoch 145: 100% 73/73 [00:03<00:00, 18.27it/s, loss=2.9937, hm=0.4283, paf=0.1855, cls=0.6063]

Epoch 145 - Train: 2.6341 (HM: 0.3498, PAF: 0.1712, Cls: 0.5949)  LR: 0.000002
Epoch 146: 100% 73/73 [00:03<00:00, 18.33it/s, loss=2.9738, hm=0.4049, paf=0.1887, cls=0.6512]
                                             
Epoch 146 Summary:
  Train - Total: 2.6486, HM: 0.3519, PAF: 0.1722, Cls: 0.5976
  Val   - Total: 2.7159, HM: 0.3538, PAF: 0.1838, Cls: 0.6220
  Metrics - Class Acc: 65.20%, PCK@0.2: 54.38%
  LR: 0.000001
  💾 Saved best model (val_loss: 2.7159)
Epoch 147: 100% 73/73 [00:04<00:00, 18.16it/s, loss=3.1166, hm=0.4558, paf=0.2017, cls=0.5933]

Epoch 147 - Train: 2.6496 (HM: 0.3515, PAF: 0.1734, Cls: 0.5978)  LR: 0.000001
Epoch 148: 100% 73/73 [00:04<00:00, 18.11it/s, loss=2.4822, hm=0.3433, paf=0.1540, cls=0.5338]
                                             
Epoch 148 Summary:
  Train - Total: 2.6382, HM: 0.3502, PAF: 0.1713, Cls: 0.5964
  Val   - Total: 2.7180, HM: 0.3534, PAF: 0.1838, Cls: 0.6244
  Metrics - Class Acc: 64.72%, PCK@0.2: 54.69%
  LR: 0.000001
Epoch 149: 100% 73/73 [00:04<00:00, 17.39it/s, loss=3.0901, hm=0.4260, paf=0.2056, cls=0.6497]

Epoch 149 - Train: 2.6481 (HM: 0.3516, PAF: 0.1735, Cls: 0.5966)  LR: 0.000001
Epoch 150: 100% 73/73 [00:04<00:00, 18.00it/s, loss=2.3846, hm=0.2956, paf=0.1712, cls=0.5731]
                                             
Epoch 150 Summary:
  Train - Total: 2.6388, HM: 0.3500, PAF: 0.1733, Cls: 0.5948
  Val   - Total: 2.7150, HM: 0.3531, PAF: 0.1838, Cls: 0.6234
  Metrics - Class Acc: 64.96%, PCK@0.2: 54.76%
  LR: 0.000001
  💾 Saved best model (val_loss: 2.7150)

================================================================================
✅ Training complete! Best val loss: 2.7150
================================================================================

🎉 Fine-tuning finished successfully!
💾 Checkpoints saved to: /content/shoes_vto_v2/shoes_vto/outputs/m1_finetune_v4/checkpoints
