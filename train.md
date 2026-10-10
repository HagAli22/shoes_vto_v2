/content/shoes_vto_v2/shoes_vto
================================================================================
👟 ARShoe M1v2 Fresh Training Pipeline
================================================================================
  Target Dataset Root   : /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4
  Output Directory      : /content/shoes_vto_v2/shoes_vto/outputs/m1v2_training
  Total Epochs          : 100
  Learning Rate         : 0.0005 (Cosine decay -> 1e-06)
  Batch Size            : 16
  Workers               : 4
  Mixed Precision (AMP) : True
  Pretrained Backbone   : True
================================================================================

📁 Loading datasets...
Loaded 1798 images from /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4/train/images
Loaded 222 images from /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4/valid/images
  Train: 1798 images
  Val  : 222 images

🏗️  Initializing ARShoe M1v2 Architecture...
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
100% 9.83M/9.83M [00:00<00:00, 191MB/s]
  Total parameters: 902,018
  Backbone (MobileNetV3): 870,560
  FPN-lite Fusion       : 12,560
  Heatmap Head (DSConv) : 5,968
  PAF Head (DSConv)     : 6,878
  Class Head (Global+DS): 6,052
  Budget Utilization    : 69.4% of 1.3M budget

🎯 Initializing Trainer with Differential Learning Rates...
  [GPU] Enabled TF32 & cuDNN benchmark for A100 maximum throughput!
  [GPU] Model compiled with torch.compile() for A100 maximum speed!
/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:145: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

🚀 Launching Fresh M1v2 Training for 100 epochs...

================================================================================
Training ARShoe M1 for 100 epochs
Device: cuda
Mixed Precision (AMP): Enabled
Train samples: 1798
Val samples: 222
Batch size: 16
Learning rate: 0.0005
Loss weights: {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
================================================================================
Epoch 1:   0% 0/113 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1:   1% 1/113 [00:49<1:33:04, 49.86s/it, loss=11.9972, hm=2.5797, paf=0.3059, cls=0.7111]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1:  98% 111/113 [00:56<00:00, 17.83it/s, loss=5.9658, hm=1.1163, paf=0.2328, cls=0.6901]/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1: 100% 113/113 [01:41<00:00,  1.12it/s, loss=6.0956, hm=1.1410, paf=0.2547, cls=0.6815]

Epoch 1 - Train: 8.7034 (HM: 1.7843, PAF: 0.2591, Cls: 0.6987)  LR: 0.000250
Epoch 2:   0% 0/113 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 2: 100% 113/113 [00:17<00:00,  6.28it/s, loss=3.2962, hm=0.4573, paf=0.2184, cls=0.6867]
Validation:   0% 0/14 [00:00<?, ?it/s]/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
                                               
Epoch 2 Summary:
  Train - Total: 4.2093, HM: 0.6821, PAF: 0.2182, Cls: 0.6964
  Val   - Total: 3.2660, HM: 0.4541, PAF: 0.2047, Cls: 0.6934
  Metrics - Class Acc: 50.77%, PCK@0.2: 44.65%
  LR: 0.000250
  💾 Saved best model (val_loss: 3.2660)
Epoch 3:   0% 0/113 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 3: 100% 113/113 [00:06<00:00, 16.92it/s, loss=2.7205, hm=0.3194, paf=0.2023, cls=0.6923]

Epoch 3 - Train: 2.8718 (HM: 0.3632, PAF: 0.1903, Cls: 0.6923)  LR: 0.000249
Epoch 4: 100% 113/113 [00:06<00:00, 17.01it/s, loss=2.2445, hm=0.2373, paf=0.1536, cls=0.6588]
                                               
Epoch 4 Summary:
  Train - Total: 2.4918, HM: 0.2841, PAF: 0.1630, Cls: 0.6862
  Val   - Total: 2.4028, HM: 0.2700, PAF: 0.1496, Cls: 0.6825
  Metrics - Class Acc: 55.29%, PCK@0.2: 59.38%
  LR: 0.000249
  💾 Saved best model (val_loss: 2.4028)
Epoch 5: 100% 113/113 [00:06<00:00, 17.06it/s, loss=2.1111, hm=0.2379, paf=0.1236, cls=0.6083]

Epoch 5 - Train: 2.2819 (HM: 0.2539, PAF: 0.1399, Cls: 0.6577)  LR: 0.000248
Epoch 6: 100% 113/113 [00:06<00:00, 17.71it/s, loss=1.8005, hm=0.2080, paf=0.1055, cls=0.5051]
                                               
Epoch 6 Summary:
  Train - Total: 2.0743, HM: 0.2396, PAF: 0.1226, Cls: 0.5804
  Val   - Total: 2.0791, HM: 0.2467, PAF: 0.1183, Cls: 0.5705
  Metrics - Class Acc: 69.70%, PCK@0.2: 60.92%
  LR: 0.000248
  💾 Saved best model (val_loss: 2.0791)
Epoch 7: 100% 113/113 [00:06<00:00, 17.12it/s, loss=2.1968, hm=0.2962, paf=0.0961, cls=0.5464]

Epoch 7 - Train: 1.9164 (HM: 0.2303, PAF: 0.1071, Cls: 0.5207)  LR: 0.000247
Epoch 8: 100% 113/113 [00:06<00:00, 17.54it/s, loss=1.7925, hm=0.2170, paf=0.1194, cls=0.4572]
                                               
Epoch 8 Summary:
  Train - Total: 1.7746, HM: 0.2222, PAF: 0.0976, Cls: 0.4604
  Val   - Total: 1.8431, HM: 0.2335, PAF: 0.1009, Cls: 0.4715
  Metrics - Class Acc: 75.92%, PCK@0.2: 65.89%
  LR: 0.000246
  💾 Saved best model (val_loss: 1.8431)
Epoch 9: 100% 113/113 [00:06<00:00, 17.34it/s, loss=1.7590, hm=0.2294, paf=0.1115, cls=0.4122]

Epoch 9 - Train: 1.6736 (HM: 0.2155, PAF: 0.0912, Cls: 0.4193)  LR: 0.000245
Epoch 10: 100% 113/113 [00:06<00:00, 17.08it/s, loss=1.4637, hm=0.2018, paf=0.0481, cls=0.3734]
                                               
Epoch 10 Summary:
  Train - Total: 1.5514, HM: 0.2082, PAF: 0.0855, Cls: 0.3652
  Val   - Total: 1.6785, HM: 0.2212, PAF: 0.0889, Cls: 0.4105
  Metrics - Class Acc: 79.32%, PCK@0.2: 68.28%
  LR: 0.000244
  💾 Saved best model (val_loss: 1.6785)
Epoch 11: 100% 113/113 [00:06<00:00, 17.11it/s, loss=1.2963, hm=0.1894, paf=0.0764, cls=0.2573]

Epoch 11 - Train: 1.5013 (HM: 0.2031, PAF: 0.0842, Cls: 0.3471)  LR: 0.000243
Epoch 12: 100% 113/113 [00:06<00:00, 17.18it/s, loss=1.7770, hm=0.1932, paf=0.0867, cls=0.5540]
                                               
Epoch 12 Summary:
  Train - Total: 1.4187, HM: 0.1975, PAF: 0.0819, Cls: 0.3100
  Val   - Total: 1.6408, HM: 0.2102, PAF: 0.0887, Cls: 0.4150
  Metrics - Class Acc: 81.08%, PCK@0.2: 70.70%
  LR: 0.000241
  💾 Saved best model (val_loss: 1.6408)
Epoch 13: 100% 113/113 [00:06<00:00, 17.15it/s, loss=1.9631, hm=0.2191, paf=0.0733, cls=0.6267]

Epoch 13 - Train: 1.3839 (HM: 0.1945, PAF: 0.0792, Cls: 0.2984)  LR: 0.000240
Epoch 14: 100% 113/113 [00:06<00:00, 17.19it/s, loss=1.1792, hm=0.1687, paf=0.1085, cls=0.1917]
                                               
Epoch 14 Summary:
  Train - Total: 1.3152, HM: 0.1898, PAF: 0.0769, Cls: 0.2681
  Val   - Total: 1.6135, HM: 0.2034, PAF: 0.0856, Cls: 0.4190
  Metrics - Class Acc: 80.98%, PCK@0.2: 72.15%
  LR: 0.000238
  💾 Saved best model (val_loss: 1.6135)
Epoch 15: 100% 113/113 [00:06<00:00, 17.38it/s, loss=1.4870, hm=0.1977, paf=0.1058, cls=0.3231]

Epoch 15 - Train: 1.2647 (HM: 0.1855, PAF: 0.0740, Cls: 0.2498)  LR: 0.000236
Epoch 16: 100% 113/113 [00:06<00:00, 16.84it/s, loss=1.3366, hm=0.1759, paf=0.0751, cls=0.3218]
                                               
Epoch 16 Summary:
  Train - Total: 1.2387, HM: 0.1845, PAF: 0.0729, Cls: 0.2366
  Val   - Total: 1.5719, HM: 0.1972, PAF: 0.0781, Cls: 0.4178
  Metrics - Class Acc: 82.50%, PCK@0.2: 73.16%
  LR: 0.000235
  💾 Saved best model (val_loss: 1.5719)
Epoch 17: 100% 113/113 [00:06<00:00, 17.08it/s, loss=1.8426, hm=0.2355, paf=0.1141, cls=0.4482]

Epoch 17 - Train: 1.1778 (HM: 0.1805, PAF: 0.0706, Cls: 0.2098)  LR: 0.000233
Epoch 18: 100% 113/113 [00:06<00:00, 17.21it/s, loss=2.4276, hm=0.1815, paf=0.0912, cls=1.0128]
                                               
Epoch 18 Summary:
  Train - Total: 1.1686, HM: 0.1777, PAF: 0.0687, Cls: 0.2135
  Val   - Total: 1.5943, HM: 0.1953, PAF: 0.0755, Cls: 0.4414
  Metrics - Class Acc: 83.44%, PCK@0.2: 73.98%
  LR: 0.000231
Epoch 19: 100% 113/113 [00:06<00:00, 17.42it/s, loss=2.2345, hm=0.2156, paf=0.1109, cls=0.7669]

Epoch 19 - Train: 1.1534 (HM: 0.1771, PAF: 0.0679, Cls: 0.2062)  LR: 0.000228
Epoch 20: 100% 113/113 [00:06<00:00, 17.54it/s, loss=0.9555, hm=0.1475, paf=0.0661, cls=0.1555]
                                               
Epoch 20 Summary:
  Train - Total: 1.1169, HM: 0.1744, PAF: 0.0659, Cls: 0.1916
  Val   - Total: 1.5394, HM: 0.1892, PAF: 0.0731, Cls: 0.4243
  Metrics - Class Acc: 84.54%, PCK@0.2: 74.66%
  LR: 0.000226
  💾 Saved best model (val_loss: 1.5394)
Epoch 21: 100% 113/113 [00:06<00:00, 17.41it/s, loss=1.0863, hm=0.1655, paf=0.0479, cls=0.2190]

Epoch 21 - Train: 1.0662 (HM: 0.1701, PAF: 0.0644, Cls: 0.1714)  LR: 0.000224
Epoch 22: 100% 113/113 [00:06<00:00, 17.34it/s, loss=1.2940, hm=0.1936, paf=0.1030, cls=0.2092]
                                               
Epoch 22 Summary:
  Train - Total: 1.0709, HM: 0.1700, PAF: 0.0654, Cls: 0.1734
  Val   - Total: 1.5032, HM: 0.1855, PAF: 0.0745, Cls: 0.4080
  Metrics - Class Acc: 84.79%, PCK@0.2: 75.65%
  LR: 0.000221
  💾 Saved best model (val_loss: 1.5032)
Epoch 23: 100% 113/113 [00:06<00:00, 17.49it/s, loss=1.4448, hm=0.1869, paf=0.1284, cls=0.2937]

Epoch 23 - Train: 1.0316 (HM: 0.1674, PAF: 0.0623, Cls: 0.1582)  LR: 0.000219
Epoch 24: 100% 113/113 [00:06<00:00, 17.52it/s, loss=1.1104, hm=0.1422, paf=0.0700, cls=0.2678]
                                               
Epoch 24 Summary:
  Train - Total: 1.0441, HM: 0.1655, PAF: 0.0620, Cls: 0.1722
  Val   - Total: 1.4566, HM: 0.1836, PAF: 0.0686, Cls: 0.3901
  Metrics - Class Acc: 86.52%, PCK@0.2: 76.85%
  LR: 0.000216
  💾 Saved best model (val_loss: 1.4566)
Epoch 25: 100% 113/113 [00:06<00:00, 17.50it/s, loss=1.3386, hm=0.1725, paf=0.0648, cls=0.3460]

Epoch 25 - Train: 1.0082 (HM: 0.1646, PAF: 0.0600, Cls: 0.1532)  LR: 0.000214
Epoch 26: 100% 113/113 [00:06<00:00, 17.24it/s, loss=1.0460, hm=0.1521, paf=0.0523, cls=0.2219]
                                               
Epoch 26 Summary:
  Train - Total: 0.9862, HM: 0.1621, PAF: 0.0595, Cls: 0.1460
  Val   - Total: 1.4381, HM: 0.1797, PAF: 0.0670, Cls: 0.3902
  Metrics - Class Acc: 85.92%, PCK@0.2: 77.34%
  LR: 0.000211
  💾 Saved best model (val_loss: 1.4381)
Epoch 27: 100% 113/113 [00:06<00:00, 17.43it/s, loss=1.0529, hm=0.1459, paf=0.0607, cls=0.2318]

Epoch 27 - Train: 0.9702 (HM: 0.1600, PAF: 0.0582, Cls: 0.1424)  LR: 0.000208
Epoch 28: 100% 113/113 [00:06<00:00, 17.19it/s, loss=1.1549, hm=0.1938, paf=0.0834, cls=0.1419]
                                               
Epoch 28 Summary:
  Train - Total: 0.9585, HM: 0.1595, PAF: 0.0587, Cls: 0.1355
  Val   - Total: 1.5149, HM: 0.1799, PAF: 0.0666, Cls: 0.4412
  Metrics - Class Acc: 85.41%, PCK@0.2: 76.71%
  LR: 0.000205
Epoch 29: 100% 113/113 [00:06<00:00, 17.33it/s, loss=1.2728, hm=0.1533, paf=0.0689, cls=0.3477]

Epoch 29 - Train: 0.9357 (HM: 0.1563, PAF: 0.0566, Cls: 0.1315)  LR: 0.000202
Epoch 30: 100% 113/113 [00:06<00:00, 17.38it/s, loss=1.1361, hm=0.1515, paf=0.0807, cls=0.2458]
                                               
Epoch 30 Summary:
  Train - Total: 0.9126, HM: 0.1549, PAF: 0.0561, Cls: 0.1204
  Val   - Total: 1.4692, HM: 0.1754, PAF: 0.0640, Cls: 0.4264
  Metrics - Class Acc: 86.24%, PCK@0.2: 77.43%
  LR: 0.000199
Epoch 31: 100% 113/113 [00:06<00:00, 17.51it/s, loss=0.8333, hm=0.1559, paf=0.0621, cls=0.0570]

Epoch 31 - Train: 0.9162 (HM: 0.1552, PAF: 0.0563, Cls: 0.1220)  LR: 0.000195
Epoch 32: 100% 113/113 [00:06<00:00, 17.43it/s, loss=0.8284, hm=0.1347, paf=0.0530, cls=0.1224]
                                               
Epoch 32 Summary:
  Train - Total: 0.9091, HM: 0.1534, PAF: 0.0555, Cls: 0.1230
  Val   - Total: 1.5193, HM: 0.1746, PAF: 0.0656, Cls: 0.4599
  Metrics - Class Acc: 85.46%, PCK@0.2: 78.11%
  LR: 0.000192
Epoch 33: 100% 113/113 [00:06<00:00, 17.32it/s, loss=0.7276, hm=0.1342, paf=0.0477, cls=0.0636]

Epoch 33 - Train: 0.8929 (HM: 0.1520, PAF: 0.0543, Cls: 0.1175)  LR: 0.000189
Epoch 34: 100% 113/113 [00:06<00:00, 17.50it/s, loss=1.3084, hm=0.2173, paf=0.0892, cls=0.1738]
                                               
Epoch 34 Summary:
  Train - Total: 0.8759, HM: 0.1512, PAF: 0.0528, Cls: 0.1105
  Val   - Total: 1.5040, HM: 0.1707, PAF: 0.0659, Cls: 0.4595
  Metrics - Class Acc: 85.51%, PCK@0.2: 78.71%
  LR: 0.000185
Epoch 35: 100% 113/113 [00:06<00:00, 17.74it/s, loss=1.3690, hm=0.1586, paf=0.0493, cls=0.4241]

Epoch 35 - Train: 0.8852 (HM: 0.1508, PAF: 0.0526, Cls: 0.1179)  LR: 0.000182
Epoch 36: 100% 113/113 [00:06<00:00, 17.78it/s, loss=1.0358, hm=0.1853, paf=0.0371, cls=0.1469]
                                               
Epoch 36 Summary:
  Train - Total: 0.8733, HM: 0.1492, PAF: 0.0525, Cls: 0.1143
  Val   - Total: 1.4701, HM: 0.1700, PAF: 0.0629, Cls: 0.4429
  Metrics - Class Acc: 85.74%, PCK@0.2: 79.15%
  LR: 0.000179
Epoch 37: 100% 113/113 [00:06<00:00, 17.57it/s, loss=1.0918, hm=0.1486, paf=0.0663, cls=0.2432]

Epoch 37 - Train: 0.8508 (HM: 0.1471, PAF: 0.0513, Cls: 0.1064)  LR: 0.000175
Epoch 38: 100% 113/113 [00:06<00:00, 17.54it/s, loss=1.0987, hm=0.1914, paf=0.0883, cls=0.1045]
                                               
Epoch 38 Summary:
  Train - Total: 0.8443, HM: 0.1458, PAF: 0.0510, Cls: 0.1063
  Val   - Total: 1.4703, HM: 0.1679, PAF: 0.0617, Cls: 0.4500
  Metrics - Class Acc: 85.60%, PCK@0.2: 79.46%
  LR: 0.000171
Epoch 39: 100% 113/113 [00:06<00:00, 17.43it/s, loss=1.4302, hm=0.2054, paf=0.0851, cls=0.2925]

Epoch 39 - Train: 0.8420 (HM: 0.1459, PAF: 0.0517, Cls: 0.1035)  LR: 0.000168
Epoch 40: 100% 113/113 [00:06<00:00, 17.26it/s, loss=0.8161, hm=0.1594, paf=0.0447, cls=0.0594]
                                               
Epoch 40 Summary:
  Train - Total: 0.8252, HM: 0.1440, PAF: 0.0498, Cls: 0.0998
  Val   - Total: 1.5105, HM: 0.1692, PAF: 0.0625, Cls: 0.4724
  Metrics - Class Acc: 85.98%, PCK@0.2: 79.02%
  LR: 0.000164
Epoch 41: 100% 113/113 [00:06<00:00, 17.33it/s, loss=1.0051, hm=0.1478, paf=0.0448, cls=0.2162]

Epoch 41 - Train: 0.8268 (HM: 0.1433, PAF: 0.0504, Cls: 0.1020)  LR: 0.000160
Epoch 42: 100% 113/113 [00:06<00:00, 17.56it/s, loss=1.0546, hm=0.1384, paf=0.0645, cls=0.2479]
                                               
Epoch 42 Summary:
  Train - Total: 0.8245, HM: 0.1426, PAF: 0.0493, Cls: 0.1036
  Val   - Total: 1.5495, HM: 0.1673, PAF: 0.0608, Cls: 0.5059
  Metrics - Class Acc: 85.31%, PCK@0.2: 79.87%
  LR: 0.000156
Epoch 43: 100% 113/113 [00:06<00:00, 17.55it/s, loss=1.0328, hm=0.1536, paf=0.1013, cls=0.1438]

Epoch 43 - Train: 0.8143 (HM: 0.1413, PAF: 0.0494, Cls: 0.1000)  LR: 0.000153
Epoch 44: 100% 113/113 [00:06<00:00, 17.32it/s, loss=0.8202, hm=0.1388, paf=0.0731, cls=0.0792]
                                               
Epoch 44 Summary:
  Train - Total: 0.7969, HM: 0.1402, PAF: 0.0495, Cls: 0.0914
  Val   - Total: 1.5440, HM: 0.1657, PAF: 0.0623, Cls: 0.5044
  Metrics - Class Acc: 86.46%, PCK@0.2: 79.41%
  LR: 0.000149
Epoch 45: 100% 113/113 [00:06<00:00, 17.70it/s, loss=1.2935, hm=0.1870, paf=0.0842, cls=0.2513]

Epoch 45 - Train: 0.8024 (HM: 0.1399, PAF: 0.0489, Cls: 0.0966)  LR: 0.000145
Epoch 46: 100% 113/113 [00:06<00:00, 17.77it/s, loss=0.9059, hm=0.1516, paf=0.0683, cls=0.1085]
                                               
Epoch 46 Summary:
  Train - Total: 0.7992, HM: 0.1396, PAF: 0.0480, Cls: 0.0965
  Val   - Total: 1.5150, HM: 0.1660, PAF: 0.0612, Cls: 0.4856
  Metrics - Class Acc: 85.89%, PCK@0.2: 79.53%
  LR: 0.000141
Epoch 47: 100% 113/113 [00:06<00:00, 17.75it/s, loss=0.8590, hm=0.1587, paf=0.0428, cls=0.0925]

Epoch 47 - Train: 0.7765 (HM: 0.1374, PAF: 0.0468, Cls: 0.0888)  LR: 0.000137
Epoch 48: 100% 113/113 [00:06<00:00, 17.73it/s, loss=1.2460, hm=0.1358, paf=0.0630, cls=0.3844]
                                               
Epoch 48 Summary:
  Train - Total: 0.7797, HM: 0.1372, PAF: 0.0479, Cls: 0.0902
  Val   - Total: 1.5570, HM: 0.1635, PAF: 0.0618, Cls: 0.5196
  Metrics - Class Acc: 85.46%, PCK@0.2: 79.70%
  LR: 0.000133
Epoch 49: 100% 113/113 [00:06<00:00, 17.67it/s, loss=0.8996, hm=0.1836, paf=0.0526, cls=0.0399]

Epoch 49 - Train: 0.7773 (HM: 0.1371, PAF: 0.0469, Cls: 0.0902)  LR: 0.000129
Epoch 50: 100% 113/113 [00:06<00:00, 17.06it/s, loss=0.8110, hm=0.1331, paf=0.0492, cls=0.1200]
                                               
Epoch 50 Summary:
  Train - Total: 0.7677, HM: 0.1357, PAF: 0.0461, Cls: 0.0884
  Val   - Total: 1.4569, HM: 0.1627, PAF: 0.0592, Cls: 0.4584
  Metrics - Class Acc: 86.34%, PCK@0.2: 79.56%
  LR: 0.000126
Epoch 51: 100% 113/113 [00:06<00:00, 17.58it/s, loss=0.9955, hm=0.1915, paf=0.0373, cls=0.1032]

Epoch 51 - Train: 0.7597 (HM: 0.1350, PAF: 0.0460, Cls: 0.0852)  LR: 0.000122
Epoch 52: 100% 113/113 [00:06<00:00, 17.80it/s, loss=0.7336, hm=0.1092, paf=0.0646, cls=0.1117]
                                               
Epoch 52 Summary:
  Train - Total: 0.7567, HM: 0.1345, PAF: 0.0460, Cls: 0.0844
  Val   - Total: 1.4796, HM: 0.1596, PAF: 0.0587, Cls: 0.4826
  Metrics - Class Acc: 86.63%, PCK@0.2: 80.06%
  LR: 0.000118
Epoch 53: 100% 113/113 [00:06<00:00, 16.96it/s, loss=0.6825, hm=0.1099, paf=0.0452, cls=0.1017]

Epoch 53 - Train: 0.7479 (HM: 0.1330, PAF: 0.0457, Cls: 0.0831)  LR: 0.000114
Epoch 54: 100% 113/113 [00:06<00:00, 16.84it/s, loss=0.6062, hm=0.1132, paf=0.0501, cls=0.0356]
                                               
Epoch 54 Summary:
  Train - Total: 0.7493, HM: 0.1322, PAF: 0.0451, Cls: 0.0868
  Val   - Total: 1.4800, HM: 0.1607, PAF: 0.0595, Cls: 0.4789
  Metrics - Class Acc: 86.26%, PCK@0.2: 80.42%
  LR: 0.000110
Epoch 55: 100% 113/113 [00:06<00:00, 16.88it/s, loss=0.8885, hm=0.1354, paf=0.0344, cls=0.1856]

Epoch 55 - Train: 0.7436 (HM: 0.1317, PAF: 0.0434, Cls: 0.0866)  LR: 0.000106
Epoch 56: 100% 113/113 [00:06<00:00, 16.88it/s, loss=0.6485, hm=0.1274, paf=0.0603, cls=0.0121]
                                               
Epoch 56 Summary:
  Train - Total: 0.7340, HM: 0.1313, PAF: 0.0439, Cls: 0.0807
  Val   - Total: 1.4567, HM: 0.1580, PAF: 0.0589, Cls: 0.4713
  Metrics - Class Acc: 86.27%, PCK@0.2: 80.74%
  LR: 0.000102
Epoch 57: 100% 113/113 [00:06<00:00, 17.10it/s, loss=0.6138, hm=0.1122, paf=0.0469, cls=0.0476]

Epoch 57 - Train: 0.7313 (HM: 0.1304, PAF: 0.0442, Cls: 0.0810)  LR: 0.000098
Epoch 58: 100% 113/113 [00:06<00:00, 17.39it/s, loss=0.6763, hm=0.1131, paf=0.0466, cls=0.0871]
                                               
Epoch 58 Summary:
  Train - Total: 0.7238, HM: 0.1295, PAF: 0.0437, Cls: 0.0788
  Val   - Total: 1.5115, HM: 0.1571, PAF: 0.0575, Cls: 0.5121
  Metrics - Class Acc: 86.27%, PCK@0.2: 80.74%
  LR: 0.000095
Epoch 59: 100% 113/113 [00:06<00:00, 17.62it/s, loss=1.1037, hm=0.1387, paf=0.0660, cls=0.2780]

Epoch 59 - Train: 0.7181 (HM: 0.1291, PAF: 0.0428, Cls: 0.0773)  LR: 0.000091
Epoch 60: 100% 113/113 [00:06<00:00, 17.68it/s, loss=1.0897, hm=0.1794, paf=0.0576, cls=0.1713]
                                               
Epoch 60 Summary:
  Train - Total: 0.7136, HM: 0.1292, PAF: 0.0432, Cls: 0.0737
  Val   - Total: 1.5098, HM: 0.1575, PAF: 0.0588, Cls: 0.5082
  Metrics - Class Acc: 86.16%, PCK@0.2: 80.30%
  LR: 0.000087
Epoch 61: 100% 113/113 [00:06<00:00, 17.56it/s, loss=0.8793, hm=0.1644, paf=0.0421, cls=0.0917]

Epoch 61 - Train: 0.7171 (HM: 0.1288, PAF: 0.0429, Cls: 0.0775)  LR: 0.000083
Epoch 62: 100% 113/113 [00:06<00:00, 17.46it/s, loss=0.8635, hm=0.1477, paf=0.0803, cls=0.0747]
                                               
Epoch 62 Summary:
  Train - Total: 0.7088, HM: 0.1275, PAF: 0.0430, Cls: 0.0751
  Val   - Total: 1.4922, HM: 0.1572, PAF: 0.0578, Cls: 0.4986
  Metrics - Class Acc: 86.07%, PCK@0.2: 80.67%
  LR: 0.000080
Epoch 63: 100% 113/113 [00:06<00:00, 17.27it/s, loss=1.1358, hm=0.1540, paf=0.0566, cls=0.2710]

Epoch 63 - Train: 0.7207 (HM: 0.1278, PAF: 0.0428, Cls: 0.0826)  LR: 0.000076
Epoch 64: 100% 113/113 [00:06<00:00, 17.42it/s, loss=0.7697, hm=0.1366, paf=0.0339, cls=0.1036]
                                               
Epoch 64 Summary:
  Train - Total: 0.7084, HM: 0.1268, PAF: 0.0427, Cls: 0.0771
  Val   - Total: 1.5125, HM: 0.1564, PAF: 0.0579, Cls: 0.5141
  Metrics - Class Acc: 86.86%, PCK@0.2: 81.08%
  LR: 0.000072
Epoch 65: 100% 113/113 [00:06<00:00, 17.76it/s, loss=0.8190, hm=0.1648, paf=0.0282, cls=0.0688]

Epoch 65 - Train: 0.7043 (HM: 0.1274, PAF: 0.0422, Cls: 0.0735)  LR: 0.000069
Epoch 66: 100% 113/113 [00:06<00:00, 17.11it/s, loss=0.7404, hm=0.1318, paf=0.0394, cls=0.0896]
                                               
Epoch 66 Summary:
  Train - Total: 0.6945, HM: 0.1260, PAF: 0.0414, Cls: 0.0717
  Val   - Total: 1.5214, HM: 0.1549, PAF: 0.0582, Cls: 0.5235
  Metrics - Class Acc: 86.35%, PCK@0.2: 81.34%
  LR: 0.000066
Epoch 67: 100% 113/113 [00:06<00:00, 17.02it/s, loss=0.5996, hm=0.0986, paf=0.0453, cls=0.0765]

Epoch 67 - Train: 0.6957 (HM: 0.1259, PAF: 0.0421, Cls: 0.0720)  LR: 0.000062
Epoch 68: 100% 113/113 [00:06<00:00, 17.27it/s, loss=0.7928, hm=0.1475, paf=0.0492, cls=0.0695]
                                               
Epoch 68 Summary:
  Train - Total: 0.6957, HM: 0.1256, PAF: 0.0414, Cls: 0.0737
  Val   - Total: 1.5072, HM: 0.1548, PAF: 0.0571, Cls: 0.5160
  Metrics - Class Acc: 86.43%, PCK@0.2: 81.37%
  LR: 0.000059
Epoch 69: 100% 113/113 [00:06<00:00, 17.44it/s, loss=0.6066, hm=0.1234, paf=0.0313, cls=0.0336]

Epoch 69 - Train: 0.6961 (HM: 0.1262, PAF: 0.0418, Cls: 0.0718)  LR: 0.000056
Epoch 70: 100% 113/113 [00:06<00:00, 17.60it/s, loss=0.6870, hm=0.1266, paf=0.0347, cls=0.0742]
                                               
Epoch 70 Summary:
  Train - Total: 0.6929, HM: 0.1253, PAF: 0.0413, Cls: 0.0728
  Val   - Total: 1.5026, HM: 0.1542, PAF: 0.0574, Cls: 0.5140
  Metrics - Class Acc: 86.57%, PCK@0.2: 81.29%
  LR: 0.000052
Epoch 71: 100% 113/113 [00:06<00:00, 17.66it/s, loss=0.9749, hm=0.1582, paf=0.0837, cls=0.1164]

Epoch 71 - Train: 0.6898 (HM: 0.1251, PAF: 0.0418, Cls: 0.0706)  LR: 0.000049
Epoch 72: 100% 113/113 [00:06<00:00, 17.36it/s, loss=0.7447, hm=0.1353, paf=0.0436, cls=0.0775]
                                               
Epoch 72 Summary:
  Train - Total: 0.6868, HM: 0.1241, PAF: 0.0411, Cls: 0.0722
  Val   - Total: 1.5100, HM: 0.1540, PAF: 0.0572, Cls: 0.5198
  Metrics - Class Acc: 86.52%, PCK@0.2: 81.51%
  LR: 0.000046
Epoch 73: 100% 113/113 [00:06<00:00, 17.00it/s, loss=0.9643, hm=0.1383, paf=0.0709, cls=0.1795]

Epoch 73 - Train: 0.6824 (HM: 0.1237, PAF: 0.0409, Cls: 0.0705)  LR: 0.000043
Epoch 74: 100% 113/113 [00:06<00:00, 17.14it/s, loss=1.0243, hm=0.1714, paf=0.0849, cls=0.1126]
                                               
Epoch 74 Summary:
  Train - Total: 0.6813, HM: 0.1235, PAF: 0.0406, Cls: 0.0706
  Val   - Total: 1.4967, HM: 0.1536, PAF: 0.0562, Cls: 0.5133
  Metrics - Class Acc: 86.75%, PCK@0.2: 81.46%
  LR: 0.000040
Epoch 75: 100% 113/113 [00:06<00:00, 17.19it/s, loss=0.8669, hm=0.1696, paf=0.0575, cls=0.0490]

Epoch 75 - Train: 0.6808 (HM: 0.1232, PAF: 0.0415, Cls: 0.0700)  LR: 0.000037
Epoch 76: 100% 113/113 [00:06<00:00, 17.59it/s, loss=0.7811, hm=0.1226, paf=0.0483, cls=0.1295]
                                               
Epoch 76 Summary:
  Train - Total: 0.6758, HM: 0.1227, PAF: 0.0405, Cls: 0.0693
  Val   - Total: 1.5094, HM: 0.1530, PAF: 0.0557, Cls: 0.5240
  Metrics - Class Acc: 86.61%, PCK@0.2: 81.58%
  LR: 0.000035
Epoch 77: 100% 113/113 [00:06<00:00, 17.54it/s, loss=0.7936, hm=0.1435, paf=0.0604, cls=0.0657]

Epoch 77 - Train: 0.6686 (HM: 0.1220, PAF: 0.0406, Cls: 0.0663)  LR: 0.000032
Epoch 78: 100% 113/113 [00:06<00:00, 17.44it/s, loss=0.7874, hm=0.1414, paf=0.0669, cls=0.0587]
                                               
Epoch 78 Summary:
  Train - Total: 0.6742, HM: 0.1224, PAF: 0.0406, Cls: 0.0689
  Val   - Total: 1.4796, HM: 0.1529, PAF: 0.0564, Cls: 0.5033
  Metrics - Class Acc: 86.63%, PCK@0.2: 81.32%
  LR: 0.000030
Epoch 79: 100% 113/113 [00:06<00:00, 16.85it/s, loss=0.9234, hm=0.1487, paf=0.0449, cls=0.1591]

Epoch 79 - Train: 0.6781 (HM: 0.1235, PAF: 0.0399, Cls: 0.0695)  LR: 0.000027
Epoch 80: 100% 113/113 [00:06<00:00, 16.84it/s, loss=0.8553, hm=0.1593, paf=0.0566, cls=0.0701]
                                               
Epoch 80 Summary:
  Train - Total: 0.6708, HM: 0.1222, PAF: 0.0408, Cls: 0.0669
  Val   - Total: 1.5272, HM: 0.1528, PAF: 0.0562, Cls: 0.5358
  Metrics - Class Acc: 86.52%, PCK@0.2: 81.25%
  LR: 0.000025
Epoch 81: 100% 113/113 [00:06<00:00, 16.41it/s, loss=0.5502, hm=0.1003, paf=0.0298, cls=0.0597]

Epoch 81 - Train: 0.6682 (HM: 0.1222, PAF: 0.0403, Cls: 0.0657)  LR: 0.000023
Epoch 82: 100% 113/113 [00:06<00:00, 16.99it/s, loss=1.1164, hm=0.1985, paf=0.0344, cls=0.1691]
                                               
Epoch 82 Summary:
  Train - Total: 0.6726, HM: 0.1224, PAF: 0.0403, Cls: 0.0682
  Val   - Total: 1.5385, HM: 0.1529, PAF: 0.0559, Cls: 0.5435
  Metrics - Class Acc: 86.41%, PCK@0.2: 81.32%
  LR: 0.000020
Epoch 83: 100% 113/113 [00:06<00:00, 17.07it/s, loss=0.8326, hm=0.1291, paf=0.0645, cls=0.1249]

Epoch 83 - Train: 0.6707 (HM: 0.1219, PAF: 0.0397, Cls: 0.0691)  LR: 0.000018
Epoch 84: 100% 113/113 [00:06<00:00, 17.10it/s, loss=0.9056, hm=0.1706, paf=0.0503, cls=0.0816]
                                               
Epoch 84 Summary:
  Train - Total: 0.6646, HM: 0.1215, PAF: 0.0394, Cls: 0.0666
  Val   - Total: 1.5356, HM: 0.1528, PAF: 0.0555, Cls: 0.5422
  Metrics - Class Acc: 86.48%, PCK@0.2: 81.44%
  LR: 0.000016
Epoch 85: 100% 113/113 [00:06<00:00, 17.23it/s, loss=0.5218, hm=0.1004, paf=0.0299, cls=0.0403]

Epoch 85 - Train: 0.6644 (HM: 0.1212, PAF: 0.0396, Cls: 0.0671)  LR: 0.000015
Epoch 86: 100% 113/113 [00:06<00:00, 17.09it/s, loss=0.6629, hm=0.1115, paf=0.0347, cls=0.0984]
                                               
Epoch 86 Summary:
  Train - Total: 0.6638, HM: 0.1211, PAF: 0.0395, Cls: 0.0669
  Val   - Total: 1.5107, HM: 0.1526, PAF: 0.0558, Cls: 0.5257
  Metrics - Class Acc: 86.45%, PCK@0.2: 81.44%
  LR: 0.000013
Epoch 87: 100% 113/113 [00:06<00:00, 17.20it/s, loss=0.8481, hm=0.1339, paf=0.0596, cls=0.1287]

Epoch 87 - Train: 0.6635 (HM: 0.1204, PAF: 0.0396, Cls: 0.0684)  LR: 0.000011
Epoch 88: 100% 113/113 [00:06<00:00, 17.21it/s, loss=1.0660, hm=0.1793, paf=0.0546, cls=0.1597]
                                               
Epoch 88 Summary:
  Train - Total: 0.6673, HM: 0.1216, PAF: 0.0398, Cls: 0.0676
  Val   - Total: 1.5227, HM: 0.1521, PAF: 0.0558, Cls: 0.5350
  Metrics - Class Acc: 86.51%, PCK@0.2: 81.34%
  LR: 0.000010
Epoch 89: 100% 113/113 [00:06<00:00, 17.40it/s, loss=0.7613, hm=0.1277, paf=0.0397, cls=0.1139]

Epoch 89 - Train: 0.6645 (HM: 0.1217, PAF: 0.0396, Cls: 0.0658)  LR: 0.000008
Epoch 90: 100% 113/113 [00:06<00:00, 17.22it/s, loss=0.8279, hm=0.1455, paf=0.0416, cls=0.1083]
                                               
Epoch 90 Summary:
  Train - Total: 0.6634, HM: 0.1210, PAF: 0.0393, Cls: 0.0671
  Val   - Total: 1.5115, HM: 0.1521, PAF: 0.0557, Cls: 0.5278
  Metrics - Class Acc: 86.57%, PCK@0.2: 81.41%
  LR: 0.000007
Epoch 91: 100% 113/113 [00:06<00:00, 17.23it/s, loss=0.7280, hm=0.1423, paf=0.0509, cls=0.0380]

Epoch 91 - Train: 0.6572 (HM: 0.1212, PAF: 0.0391, Cls: 0.0628)  LR: 0.000006
Epoch 92: 100% 113/113 [00:06<00:00, 17.17it/s, loss=0.5217, hm=0.0981, paf=0.0225, cls=0.0563]
                                               
Epoch 92 Summary:
  Train - Total: 0.6591, HM: 0.1207, PAF: 0.0391, Cls: 0.0654
  Val   - Total: 1.5038, HM: 0.1526, PAF: 0.0558, Cls: 0.5211
  Metrics - Class Acc: 86.57%, PCK@0.2: 81.46%
  LR: 0.000005
Epoch 93: 100% 113/113 [00:06<00:00, 17.07it/s, loss=0.7456, hm=0.1418, paf=0.0618, cls=0.0366]

Epoch 93 - Train: 0.6554 (HM: 0.1204, PAF: 0.0396, Cls: 0.0632)  LR: 0.000004
Epoch 94: 100% 113/113 [00:06<00:00, 17.27it/s, loss=0.5662, hm=0.0988, paf=0.0316, cls=0.0718]
                                               
Epoch 94 Summary:
  Train - Total: 0.6591, HM: 0.1211, PAF: 0.0393, Cls: 0.0641
  Val   - Total: 1.5226, HM: 0.1520, PAF: 0.0559, Cls: 0.5352
  Metrics - Class Acc: 86.51%, PCK@0.2: 81.39%
  LR: 0.000003
Epoch 95: 100% 113/113 [00:06<00:00, 16.94it/s, loss=0.7028, hm=0.1139, paf=0.0280, cls=0.1276]

Epoch 95 - Train: 0.6557 (HM: 0.1209, PAF: 0.0389, Cls: 0.0629)  LR: 0.000003
Epoch 96: 100% 113/113 [00:06<00:00, 17.40it/s, loss=1.0482, hm=0.1620, paf=0.0490, cls=0.2016]

Epoch 96 Summary:
  Train - Total: 0.6602, HM: 0.1209, PAF: 0.0399, Cls: 0.0647
  Val   - Total: 1.5236, HM: 0.1521, PAF: 0.0560, Cls: 0.5354
  Metrics - Class Acc: 86.59%, PCK@0.2: 81.15%
  LR: 0.000002
Epoch 97: 100% 113/113 [00:06<00:00, 17.36it/s, loss=0.8539, hm=0.1630, paf=0.0359, cls=0.0868]

Epoch 97 - Train: 0.6667 (HM: 0.1218, PAF: 0.0394, Cls: 0.0672)  LR: 0.000002
Epoch 98: 100% 113/113 [00:06<00:00, 17.48it/s, loss=0.7410, hm=0.1258, paf=0.0477, cls=0.0950]
                                               
Epoch 98 Summary:
  Train - Total: 0.6619, HM: 0.1215, PAF: 0.0392, Cls: 0.0650
  Val   - Total: 1.5326, HM: 0.1523, PAF: 0.0559, Cls: 0.5412
  Metrics - Class Acc: 86.47%, PCK@0.2: 81.34%
  LR: 0.000001
Epoch 99: 100% 113/113 [00:06<00:00, 17.43it/s, loss=0.6400, hm=0.1353, paf=0.0368, cls=0.0166]

Epoch 99 - Train: 0.6579 (HM: 0.1205, PAF: 0.0392, Cls: 0.0650)  LR: 0.000001
Epoch 100: 100% 113/113 [00:06<00:00, 17.37it/s, loss=1.0924, hm=0.1799, paf=0.0555, cls=0.1745]
                                               
Epoch 100 Summary:
  Train - Total: 0.6635, HM: 0.1216, PAF: 0.0399, Cls: 0.0649
  Val   - Total: 1.5218, HM: 0.1525, PAF: 0.0557, Cls: 0.5338
  Metrics - Class Acc: 86.53%, PCK@0.2: 81.27%
  LR: 0.000001

================================================================================
✅ Training complete! Best val loss: 1.4381
================================================================================

🎉 M1v2 Training finished successfully!
💾 Checkpoints saved to: /content/shoes_vto_v2/shoes_vto/outputs/m1v2_training/checkpoints
