================================================================================
👟 ARShoe M1v2 13-Keypoint Fresh Training Pipeline
================================================================================
  Target Dataset Root   : /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4_13kp
  PAF Configuration     : /content/shoes_vto_v2/shoes_vto/configs/paf_connections_13kp.yaml
  Output Directory      : /content/shoes_vto_v2/shoes_vto/outputs/m1v2_13kp_training
  Total Epochs          : 150
  Learning Rate         : 0.0005 (Cosine decay -> 1e-06)
  Batch Size            : 16
  Workers               : 4
  Mixed Precision (AMP) : True
  Pretrained Backbone   : True
================================================================================

📁 Loading datasets...
Loaded 1798 images from /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4_13kp/train/images
Loaded 222 images from /content/shoes_vto_v2/shoes_vto/dataset/merged_v3_v4_13kp/valid/images
  Train: 1798 images
  Val  : 222 images

🏗️  Initializing ARShoe M1v2 (13-KP) Architecture...
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
100% 9.83M/9.83M [00:00<00:00, 105MB/s]
  Total parameters: 901,953
  Backbone (MobileNetV3): 870,560
  FPN-lite Fusion       : 12,560
  Heatmap Head (DSConv) : 5,773 (13 channels)
  PAF Head (DSConv)     : 7,008 (32 channels)
  Class Head (Global+DS): 6,052 (2 channels)
  Budget Utilization    : 69.4% of 1.3M budget

🎯 Initializing Trainer with Differential Learning Rates...
  [GPU] Enabled TF32 & cuDNN benchmark for A100 maximum throughput!
  [GPU] Model compiled with torch.compile() for A100 maximum speed!
/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:145: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None

🚀 Launching Fresh 13-KP M1v2 Training for 150 epochs...

================================================================================
Training ARShoe M1 for 150 epochs
Device: cuda
Mixed Precision (AMP): Enabled
Train samples: 1798
Val samples: 222
Batch size: 16
Learning rate: 0.0005
Loss weights: {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
================================================================================
Epoch 1:   0% 0/113 [00:00<?, ?it/s]/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
W1012 11:45:31.096000 9714 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1:   1% 1/113 [02:31<4:42:10, 151.17s/it, loss=12.5103, hm=2.7153, paf=0.2987, cls=0.7012]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1:  99% 112/113 [03:18<00:00,  4.41it/s, loss=6.0968, hm=1.1412, paf=0.2385, cls=0.7034]/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1: 100% 113/113 [05:29<00:00,  2.91s/it, loss=5.8762, hm=1.0808, paf=0.2469, cls=0.7061]

Epoch 1 - Train: 9.0719 (HM: 1.8751, PAF: 0.2589, Cls: 0.7025)  LR: 0.000250
Epoch 2:   0% 0/113 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 2: 100% 113/113 [00:55<00:00,  2.03it/s, loss=3.1620, hm=0.4356, paf=0.1819, cls=0.7038]
Validation:   0% 0/14 [00:00<?, ?it/s]/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
                                               
Epoch 2 Summary:
  Train - Total: 4.2569, HM: 0.6918, PAF: 0.2222, Cls: 0.6969
  Val   - Total: 3.3139, HM: 0.4627, PAF: 0.2101, Cls: 0.6953
  Metrics - Class Acc: 50.80%, PCK@0.2: 49.30%
  LR: 0.000250
  💾 Saved best model (val_loss: 3.3139)
Epoch 3:   0% 0/113 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 3: 100% 113/113 [00:40<00:00,  2.76it/s, loss=3.0219, hm=0.3962, paf=0.1995, cls=0.6922]

Epoch 3 - Train: 2.8518 (HM: 0.3537, PAF: 0.1984, Cls: 0.6936)  LR: 0.000250
Epoch 4: 100% 113/113 [00:36<00:00,  3.09it/s, loss=2.3184, hm=0.2453, paf=0.1506, cls=0.6907]
                                               
Epoch 4 Summary:
  Train - Total: 2.4892, HM: 0.2736, PAF: 0.1803, Cls: 0.6895
  Val   - Total: 2.4466, HM: 0.2649, PAF: 0.1752, Cls: 0.6913
  Metrics - Class Acc: 52.30%, PCK@0.2: 63.73%
  LR: 0.000250
  💾 Saved best model (val_loss: 2.4466)
Epoch 5: 100% 113/113 [00:36<00:00,  3.11it/s, loss=2.2641, hm=0.2223, paf=0.1865, cls=0.6679]

Epoch 5 - Train: 2.2981 (HM: 0.2423, PAF: 0.1635, Cls: 0.6681)  LR: 0.000249
Epoch 6: 100% 113/113 [00:34<00:00,  3.27it/s, loss=2.6500, hm=0.3156, paf=0.1494, cls=0.7260]
                                               
Epoch 6 Summary:
  Train - Total: 2.1403, HM: 0.2287, PAF: 0.1538, Cls: 0.6120
  Val   - Total: 2.0993, HM: 0.2341, PAF: 0.1552, Cls: 0.5683
  Metrics - Class Acc: 68.57%, PCK@0.2: 66.65%
  LR: 0.000249
  💾 Saved best model (val_loss: 2.0993)
Epoch 7: 100% 113/113 [00:33<00:00,  3.34it/s, loss=2.0606, hm=0.2293, paf=0.1604, cls=0.5484]

Epoch 7 - Train: 2.0054 (HM: 0.2199, PAF: 0.1463, Cls: 0.5556)  LR: 0.000249
Epoch 8: 100% 113/113 [00:32<00:00,  3.50it/s, loss=1.7849, hm=0.1824, paf=0.1429, cls=0.5130]
                                               
Epoch 8 Summary:
  Train - Total: 1.8556, HM: 0.2139, PAF: 0.1389, Cls: 0.4815
  Val   - Total: 1.8324, HM: 0.2208, PAF: 0.1360, Cls: 0.4514
  Metrics - Class Acc: 77.28%, PCK@0.2: 69.77%
  LR: 0.000248
  💾 Saved best model (val_loss: 1.8324)
Epoch 9: 100% 113/113 [00:33<00:00,  3.34it/s, loss=1.9435, hm=0.2075, paf=0.1314, cls=0.5670]

Epoch 9 - Train: 1.7617 (HM: 0.2089, PAF: 0.1334, Cls: 0.4394)  LR: 0.000248
Epoch 10: 100% 113/113 [00:31<00:00,  3.55it/s, loss=1.4525, hm=0.2144, paf=0.1051, cls=0.2566]
                                               
Epoch 10 Summary:
  Train - Total: 1.6783, HM: 0.2047, PAF: 0.1293, Cls: 0.4006
  Val   - Total: 1.7674, HM: 0.2157, PAF: 0.1297, Cls: 0.4301
  Metrics - Class Acc: 78.09%, PCK@0.2: 69.20%
  LR: 0.000247
  💾 Saved best model (val_loss: 1.7674)
Epoch 11: 100% 113/113 [00:31<00:00,  3.62it/s, loss=2.0283, hm=0.2432, paf=0.1690, cls=0.4785]

Epoch 11 - Train: 1.5957 (HM: 0.2020, PAF: 0.1249, Cls: 0.3585)  LR: 0.000247
Epoch 12: 100% 113/113 [00:31<00:00,  3.59it/s, loss=1.3439, hm=0.1936, paf=0.1312, cls=0.2047]
                                               
Epoch 12 Summary:
  Train - Total: 1.5347, HM: 0.1989, PAF: 0.1212, Cls: 0.3310
  Val   - Total: 1.7006, HM: 0.2130, PAF: 0.1187, Cls: 0.4073
  Metrics - Class Acc: 79.82%, PCK@0.2: 70.51%
  LR: 0.000246
  💾 Saved best model (val_loss: 1.7006)
Epoch 13: 100% 113/113 [00:30<00:00,  3.68it/s, loss=1.9490, hm=0.1909, paf=0.1687, cls=0.5653]

Epoch 13 - Train: 1.5009 (HM: 0.1986, PAF: 0.1202, Cls: 0.3106)  LR: 0.000245
Epoch 14: 100% 113/113 [00:30<00:00,  3.71it/s, loss=1.4013, hm=0.1890, paf=0.1154, cls=0.2763]
                                               
Epoch 14 Summary:
  Train - Total: 1.4586, HM: 0.1951, PAF: 0.1169, Cls: 0.2964
  Val   - Total: 1.6881, HM: 0.2097, PAF: 0.1176, Cls: 0.4094
  Metrics - Class Acc: 81.65%, PCK@0.2: 71.94%
  LR: 0.000245
  💾 Saved best model (val_loss: 1.6881)
Epoch 15: 100% 113/113 [00:30<00:00,  3.71it/s, loss=1.5463, hm=0.2242, paf=0.1513, cls=0.2313]

Epoch 15 - Train: 1.3906 (HM: 0.1925, PAF: 0.1157, Cls: 0.2593)  LR: 0.000244
Epoch 16: 100% 113/113 [00:31<00:00,  3.63it/s, loss=1.9161, hm=0.2820, paf=0.1087, cls=0.3804]
                                               
Epoch 16 Summary:
  Train - Total: 1.3399, HM: 0.1899, PAF: 0.1109, Cls: 0.2390
  Val   - Total: 1.6406, HM: 0.2043, PAF: 0.1132, Cls: 0.3980
  Metrics - Class Acc: 82.70%, PCK@0.2: 73.29%
  LR: 0.000243
  💾 Saved best model (val_loss: 1.6406)
Epoch 17: 100% 113/113 [00:30<00:00,  3.73it/s, loss=1.4572, hm=0.2179, paf=0.0964, cls=0.2620]

Epoch 17 - Train: 1.3251 (HM: 0.1883, PAF: 0.1107, Cls: 0.2337)  LR: 0.000242
Epoch 18: 100% 113/113 [00:29<00:00,  3.77it/s, loss=1.4155, hm=0.2235, paf=0.1286, cls=0.1762]
                                               
Epoch 18 Summary:
  Train - Total: 1.2804, HM: 0.1868, PAF: 0.1075, Cls: 0.2121
  Val   - Total: 1.6080, HM: 0.1997, PAF: 0.1079, Cls: 0.3956
  Metrics - Class Acc: 84.15%, PCK@0.2: 73.32%
  LR: 0.000241
  💾 Saved best model (val_loss: 1.6080)
Epoch 19: 100% 113/113 [00:30<00:00,  3.69it/s, loss=1.6344, hm=0.1895, paf=0.0898, cls=0.4646]

Epoch 19 - Train: 1.2644 (HM: 0.1850, PAF: 0.1062, Cls: 0.2081)  LR: 0.000240
Epoch 20: 100% 113/113 [00:31<00:00,  3.64it/s, loss=1.4288, hm=0.1996, paf=0.1084, cls=0.2759]
                                               
Epoch 20 Summary:
  Train - Total: 1.2411, HM: 0.1838, PAF: 0.1039, Cls: 0.1986
  Val   - Total: 1.6149, HM: 0.2009, PAF: 0.1050, Cls: 0.4010
  Metrics - Class Acc: 84.35%, PCK@0.2: 73.60%
  LR: 0.000239
Epoch 21: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.9606, hm=0.1420, paf=0.1056, cls=0.1210]

Epoch 21 - Train: 1.2101 (HM: 0.1822, PAF: 0.1032, Cls: 0.1834)  LR: 0.000238
Epoch 22: 100% 113/113 [00:30<00:00,  3.68it/s, loss=1.2963, hm=0.1792, paf=0.0944, cls=0.2604]
                                               
Epoch 22 Summary:
  Train - Total: 1.2044, HM: 0.1795, PAF: 0.1007, Cls: 0.1900
  Val   - Total: 1.5859, HM: 0.1975, PAF: 0.1069, Cls: 0.3881
  Metrics - Class Acc: 84.34%, PCK@0.2: 73.63%
  LR: 0.000237
  💾 Saved best model (val_loss: 1.5859)
Epoch 23: 100% 113/113 [00:30<00:00,  3.68it/s, loss=0.8425, hm=0.1278, paf=0.1161, cls=0.0661]

Epoch 23 - Train: 1.1668 (HM: 0.1778, PAF: 0.1009, Cls: 0.1693)  LR: 0.000236
Epoch 24: 100% 113/113 [00:30<00:00,  3.68it/s, loss=1.2552, hm=0.1811, paf=0.0972, cls=0.2241]
                                               
Epoch 24 Summary:
  Train - Total: 1.1618, HM: 0.1775, PAF: 0.0974, Cls: 0.1714
  Val   - Total: 1.5622, HM: 0.1934, PAF: 0.0994, Cls: 0.3933
  Metrics - Class Acc: 85.03%, PCK@0.2: 75.26%
  LR: 0.000235
  💾 Saved best model (val_loss: 1.5622)
Epoch 25: 100% 113/113 [00:30<00:00,  3.75it/s, loss=1.5757, hm=0.2474, paf=0.1070, cls=0.2481]

Epoch 25 - Train: 1.1491 (HM: 0.1768, PAF: 0.0972, Cls: 0.1651)  LR: 0.000233
Epoch 26: 100% 113/113 [00:30<00:00,  3.76it/s, loss=2.0360, hm=0.2693, paf=0.0931, cls=0.5151]
                                               
Epoch 26 Summary:
  Train - Total: 1.1127, HM: 0.1741, PAF: 0.0957, Cls: 0.1498
  Val   - Total: 1.5459, HM: 0.1902, PAF: 0.0982, Cls: 0.3925
  Metrics - Class Acc: 85.09%, PCK@0.2: 75.87%
  LR: 0.000232
  💾 Saved best model (val_loss: 1.5459)
Epoch 27: 100% 113/113 [00:30<00:00,  3.76it/s, loss=1.1939, hm=0.1789, paf=0.1257, cls=0.1515]

Epoch 27 - Train: 1.1179 (HM: 0.1739, PAF: 0.0947, Cls: 0.1553)  LR: 0.000231
Epoch 28: 100% 113/113 [00:30<00:00,  3.72it/s, loss=0.9330, hm=0.1480, paf=0.0818, cls=0.1184]
                                               
Epoch 28 Summary:
  Train - Total: 1.0885, HM: 0.1714, PAF: 0.0918, Cls: 0.1463
  Val   - Total: 1.5965, HM: 0.1899, PAF: 0.0946, Cls: 0.4317
  Metrics - Class Acc: 84.80%, PCK@0.2: 76.21%
  LR: 0.000229
Epoch 29: 100% 113/113 [00:30<00:00,  3.74it/s, loss=1.1420, hm=0.1678, paf=0.0897, cls=0.1943]

Epoch 29 - Train: 1.0611 (HM: 0.1691, PAF: 0.0902, Cls: 0.1362)  LR: 0.000228
Epoch 30: 100% 113/113 [00:30<00:00,  3.74it/s, loss=1.5119, hm=0.2189, paf=0.1244, cls=0.2584]
                                               
Epoch 30 Summary:
  Train - Total: 1.0655, HM: 0.1696, PAF: 0.0894, Cls: 0.1390
  Val   - Total: 1.6494, HM: 0.1863, PAF: 0.0937, Cls: 0.4780
  Metrics - Class Acc: 85.09%, PCK@0.2: 76.41%
  LR: 0.000226
Epoch 31: 100% 113/113 [00:29<00:00,  3.79it/s, loss=1.2380, hm=0.1733, paf=0.1343, cls=0.1841]

Epoch 31 - Train: 1.0421 (HM: 0.1671, PAF: 0.0887, Cls: 0.1307)  LR: 0.000225
Epoch 32: 100% 113/113 [00:30<00:00,  3.76it/s, loss=1.1583, hm=0.1560, paf=0.0664, cls=0.2677]
                                               
Epoch 32 Summary:
  Train - Total: 1.0232, HM: 0.1656, PAF: 0.0860, Cls: 0.1259
  Val   - Total: 1.6026, HM: 0.1837, PAF: 0.0911, Cls: 0.4572
  Metrics - Class Acc: 84.62%, PCK@0.2: 76.52%
  LR: 0.000223
Epoch 33: 100% 113/113 [00:30<00:00,  3.76it/s, loss=0.9453, hm=0.1576, paf=0.0902, cls=0.0897]

Epoch 33 - Train: 1.0116 (HM: 0.1646, PAF: 0.0866, Cls: 0.1201)  LR: 0.000221
Epoch 34: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.9786, hm=0.1735, paf=0.0989, cls=0.0578]
                                               
Epoch 34 Summary:
  Train - Total: 1.0007, HM: 0.1630, PAF: 0.0837, Cls: 0.1210
  Val   - Total: 1.5781, HM: 0.1828, PAF: 0.0869, Cls: 0.4488
  Metrics - Class Acc: 85.10%, PCK@0.2: 77.56%
  LR: 0.000220
Epoch 35: 100% 113/113 [00:30<00:00,  3.73it/s, loss=1.4401, hm=0.2364, paf=0.1010, cls=0.1949]

Epoch 35 - Train: 0.9805 (HM: 0.1617, PAF: 0.0830, Cls: 0.1119)  LR: 0.000218
Epoch 36: 100% 113/113 [00:29<00:00,  3.77it/s, loss=1.4208, hm=0.1975, paf=0.0991, cls=0.2884]
                                               
Epoch 36 Summary:
  Train - Total: 0.9792, HM: 0.1603, PAF: 0.0812, Cls: 0.1170
  Val   - Total: 1.5784, HM: 0.1797, PAF: 0.0891, Cls: 0.4542
  Metrics - Class Acc: 84.86%, PCK@0.2: 77.58%
  LR: 0.000216
Epoch 37: 100% 113/113 [00:30<00:00,  3.73it/s, loss=1.0338, hm=0.1885, paf=0.0937, cls=0.0617]

Epoch 37 - Train: 0.9760 (HM: 0.1589, PAF: 0.0813, Cls: 0.1186)  LR: 0.000214
Epoch 38: 100% 113/113 [00:30<00:00,  3.71it/s, loss=1.2374, hm=0.1789, paf=0.0931, cls=0.2237]
                                               
Epoch 38 Summary:
  Train - Total: 0.9674, HM: 0.1578, PAF: 0.0801, Cls: 0.1173
  Val   - Total: 1.6089, HM: 0.1773, PAF: 0.0862, Cls: 0.4847
  Metrics - Class Acc: 85.15%, PCK@0.2: 78.30%
  LR: 0.000213
Epoch 39: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.7192, hm=0.1273, paf=0.0707, cls=0.0458]

Epoch 39 - Train: 0.9368 (HM: 0.1561, PAF: 0.0782, Cls: 0.1039)  LR: 0.000211
Epoch 40: 100% 113/113 [00:30<00:00,  3.72it/s, loss=0.8674, hm=0.1357, paf=0.0912, cls=0.0949]
                                               
Epoch 40 Summary:
  Train - Total: 0.9313, HM: 0.1546, PAF: 0.0769, Cls: 0.1061
  Val   - Total: 1.5529, HM: 0.1769, PAF: 0.0853, Cls: 0.4497
  Metrics - Class Acc: 86.46%, PCK@0.2: 78.04%
  LR: 0.000209
Epoch 41: 100% 113/113 [00:30<00:00,  3.76it/s, loss=0.7765, hm=0.1268, paf=0.0803, cls=0.0724]

Epoch 41 - Train: 0.9170 (HM: 0.1534, PAF: 0.0758, Cls: 0.1012)  LR: 0.000207
Epoch 42: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.9457, hm=0.1534, paf=0.1085, cls=0.0766]

Epoch 42 Summary:
  Train - Total: 0.9072, HM: 0.1526, PAF: 0.0757, Cls: 0.0969
  Val   - Total: 1.6114, HM: 0.1747, PAF: 0.0830, Cls: 0.4977
  Metrics - Class Acc: 84.96%, PCK@0.2: 78.41%
  LR: 0.000205
Epoch 43: 100% 113/113 [00:30<00:00,  3.73it/s, loss=1.0231, hm=0.1596, paf=0.0669, cls=0.1673]

Epoch 43 - Train: 0.9067 (HM: 0.1511, PAF: 0.0737, Cls: 0.1031)  LR: 0.000203
Epoch 44: 100% 113/113 [00:30<00:00,  3.74it/s, loss=1.0951, hm=0.1941, paf=0.0850, cls=0.0993]
                                               
Epoch 44 Summary:
  Train - Total: 0.9031, HM: 0.1505, PAF: 0.0734, Cls: 0.1028
  Val   - Total: 1.6548, HM: 0.1739, PAF: 0.0849, Cls: 0.5264
  Metrics - Class Acc: 84.59%, PCK@0.2: 79.42%
  LR: 0.000201
Epoch 45: 100% 113/113 [00:30<00:00,  3.76it/s, loss=1.6059, hm=0.2144, paf=0.1026, cls=0.3621]

Epoch 45 - Train: 0.9009 (HM: 0.1502, PAF: 0.0726, Cls: 0.1032)  LR: 0.000199
Epoch 46: 100% 113/113 [00:29<00:00,  3.79it/s, loss=1.1342, hm=0.2117, paf=0.1038, cls=0.0531]
                                               
Epoch 46 Summary:
  Train - Total: 0.8796, HM: 0.1489, PAF: 0.0722, Cls: 0.0931
  Val   - Total: 1.5894, HM: 0.1720, PAF: 0.0797, Cls: 0.4946
  Metrics - Class Acc: 85.41%, PCK@0.2: 79.76%
  LR: 0.000197
Epoch 47: 100% 113/113 [00:29<00:00,  3.79it/s, loss=1.2363, hm=0.1612, paf=0.0979, cls=0.2639]

Epoch 47 - Train: 0.8638 (HM: 0.1475, PAF: 0.0699, Cls: 0.0892)  LR: 0.000194
Epoch 48: 100% 113/113 [00:29<00:00,  3.79it/s, loss=0.8692, hm=0.1347, paf=0.0485, cls=0.1556]
                                               
Epoch 48 Summary:
  Train - Total: 0.8624, HM: 0.1459, PAF: 0.0700, Cls: 0.0925
  Val   - Total: 1.5420, HM: 0.1673, PAF: 0.0806, Cls: 0.4745
  Metrics - Class Acc: 85.58%, PCK@0.2: 80.16%
  LR: 0.000192
  💾 Saved best model (val_loss: 1.5420)
Epoch 49: 100% 113/113 [00:30<00:00,  3.75it/s, loss=0.8848, hm=0.1599, paf=0.0556, cls=0.0892]

Epoch 49 - Train: 0.8569 (HM: 0.1454, PAF: 0.0677, Cls: 0.0931)  LR: 0.000190
Epoch 50: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.9421, hm=0.1486, paf=0.0922, cls=0.1090]
                                               
Epoch 50 Summary:
  Train - Total: 0.8591, HM: 0.1444, PAF: 0.0679, Cls: 0.0972
  Val   - Total: 1.6830, HM: 0.1714, PAF: 0.0824, Cls: 0.5550
  Metrics - Class Acc: 84.52%, PCK@0.2: 79.36%
  LR: 0.000188
Epoch 51: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.9199, hm=0.1459, paf=0.0740, cls=0.1256]

Epoch 51 - Train: 0.8543 (HM: 0.1440, PAF: 0.0670, Cls: 0.0962)  LR: 0.000185
Epoch 52: 100% 113/113 [00:29<00:00,  3.80it/s, loss=0.9367, hm=0.1433, paf=0.0662, cls=0.1539]
                                               
Epoch 52 Summary:
  Train - Total: 0.8357, HM: 0.1436, PAF: 0.0663, Cls: 0.0859
  Val   - Total: 1.5732, HM: 0.1649, PAF: 0.0792, Cls: 0.5033
  Metrics - Class Acc: 85.36%, PCK@0.2: 80.42%
  LR: 0.000183
Epoch 53: 100% 113/113 [00:30<00:00,  3.75it/s, loss=0.9680, hm=0.1453, paf=0.1090, cls=0.1124]

Epoch 53 - Train: 0.8219 (HM: 0.1413, PAF: 0.0649, Cls: 0.0846)  LR: 0.000181
Epoch 54: 100% 113/113 [00:30<00:00,  3.68it/s, loss=0.8735, hm=0.1499, paf=0.0634, cls=0.0981]
                                               
Epoch 54 Summary:
  Train - Total: 0.8231, HM: 0.1414, PAF: 0.0640, Cls: 0.0863
  Val   - Total: 1.6471, HM: 0.1652, PAF: 0.0780, Cls: 0.5534
  Metrics - Class Acc: 85.44%, PCK@0.2: 79.99%
  LR: 0.000179
Epoch 55: 100% 113/113 [00:30<00:00,  3.72it/s, loss=0.8827, hm=0.1344, paf=0.0946, cls=0.1041]

Epoch 55 - Train: 0.8029 (HM: 0.1385, PAF: 0.0632, Cls: 0.0817)  LR: 0.000176
Epoch 56: 100% 113/113 [00:30<00:00,  3.75it/s, loss=0.9322, hm=0.1483, paf=0.0620, cls=0.1433]
                                               
Epoch 56 Summary:
  Train - Total: 0.8050, HM: 0.1393, PAF: 0.0620, Cls: 0.0824
  Val   - Total: 1.5876, HM: 0.1633, PAF: 0.0796, Cls: 0.5169
  Metrics - Class Acc: 85.68%, PCK@0.2: 79.87%
  LR: 0.000174
Epoch 57: 100% 113/113 [00:30<00:00,  3.74it/s, loss=0.8460, hm=0.1432, paf=0.0675, cls=0.0920]

Epoch 57 - Train: 0.8005 (HM: 0.1378, PAF: 0.0618, Cls: 0.0838)  LR: 0.000171
Epoch 58: 100% 113/113 [00:30<00:00,  3.70it/s, loss=1.0150, hm=0.1853, paf=0.0818, cls=0.0735]
                                               
Epoch 58 Summary:
  Train - Total: 0.7883, HM: 0.1372, PAF: 0.0607, Cls: 0.0787
  Val   - Total: 1.5347, HM: 0.1596, PAF: 0.0772, Cls: 0.4946
  Metrics - Class Acc: 86.12%, PCK@0.2: 80.93%
  LR: 0.000169
  💾 Saved best model (val_loss: 1.5347)
Epoch 59: 100% 113/113 [00:30<00:00,  3.75it/s, loss=1.1592, hm=0.1461, paf=0.0928, cls=0.2594]

Epoch 59 - Train: 0.7870 (HM: 0.1360, PAF: 0.0602, Cls: 0.0818)  LR: 0.000166
Epoch 60: 100% 113/113 [00:30<00:00,  3.75it/s, loss=0.7060, hm=0.1211, paf=0.0453, cls=0.0874]
                                               
Epoch 60 Summary:
  Train - Total: 0.7786, HM: 0.1353, PAF: 0.0598, Cls: 0.0785
  Val   - Total: 1.5823, HM: 0.1595, PAF: 0.0742, Cls: 0.5305
  Metrics - Class Acc: 85.94%, PCK@0.2: 81.02%
  LR: 0.000164
Epoch 61: 100% 113/113 [00:30<00:00,  3.76it/s, loss=1.0143, hm=0.1633, paf=0.0628, cls=0.1569]

Epoch 61 - Train: 0.7883 (HM: 0.1360, PAF: 0.0605, Cls: 0.0823)  LR: 0.000161
Epoch 62: 100% 113/113 [00:30<00:00,  3.70it/s, loss=1.1498, hm=0.1781, paf=0.0817, cls=0.1827]
                                               
Epoch 62 Summary:
  Train - Total: 0.7724, HM: 0.1340, PAF: 0.0581, Cls: 0.0801
  Val   - Total: 1.5600, HM: 0.1606, PAF: 0.0737, Cls: 0.5133
  Metrics - Class Acc: 85.80%, PCK@0.2: 81.22%
  LR: 0.000159
Epoch 63: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.9000, hm=0.1588, paf=0.0682, cls=0.0856]

Epoch 63 - Train: 0.7692 (HM: 0.1333, PAF: 0.0578, Cls: 0.0803)  LR: 0.000156
Epoch 64: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.6870, hm=0.1180, paf=0.0565, cls=0.0679]
                                               
Epoch 64 Summary:
  Train - Total: 0.7548, HM: 0.1317, PAF: 0.0571, Cls: 0.0759
  Val   - Total: 1.5550, HM: 0.1581, PAF: 0.0738, Cls: 0.5165
  Metrics - Class Acc: 86.36%, PCK@0.2: 81.08%
  LR: 0.000154
Epoch 65: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.9609, hm=0.1563, paf=0.0819, cls=0.1146]

Epoch 65 - Train: 0.7609 (HM: 0.1326, PAF: 0.0569, Cls: 0.0778)  LR: 0.000151
Epoch 66: 100% 113/113 [00:30<00:00,  3.72it/s, loss=0.7659, hm=0.1445, paf=0.0548, cls=0.0521]
                                               
Epoch 66 Summary:
  Train - Total: 0.7441, HM: 0.1308, PAF: 0.0555, Cls: 0.0731
  Val   - Total: 1.5433, HM: 0.1590, PAF: 0.0741, Cls: 0.5061
  Metrics - Class Acc: 86.35%, PCK@0.2: 81.59%
  LR: 0.000149
Epoch 67: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.7400, hm=0.1273, paf=0.0613, cls=0.0722]

Epoch 67 - Train: 0.7488 (HM: 0.1303, PAF: 0.0556, Cls: 0.0777)  LR: 0.000146
Epoch 68: 100% 113/113 [00:29<00:00,  3.79it/s, loss=0.9293, hm=0.1375, paf=0.0840, cls=0.1409]
                                               
Epoch 68 Summary:
  Train - Total: 0.7366, HM: 0.1293, PAF: 0.0550, Cls: 0.0729
  Val   - Total: 1.5786, HM: 0.1555, PAF: 0.0747, Cls: 0.5381
  Metrics - Class Acc: 86.39%, PCK@0.2: 81.79%
  LR: 0.000144
Epoch 69: 100% 113/113 [00:29<00:00,  3.79it/s, loss=0.7262, hm=0.1254, paf=0.0503, cls=0.0827]

Epoch 69 - Train: 0.7393 (HM: 0.1288, PAF: 0.0550, Cls: 0.0761)  LR: 0.000141
Epoch 70: 100% 113/113 [00:29<00:00,  3.80it/s, loss=1.1003, hm=0.1627, paf=0.0782, cls=0.1955]
                                               
Epoch 70 Summary:
  Train - Total: 0.7291, HM: 0.1282, PAF: 0.0535, Cls: 0.0728
  Val   - Total: 1.6180, HM: 0.1568, PAF: 0.0729, Cls: 0.5633
  Metrics - Class Acc: 85.99%, PCK@0.2: 81.36%
  LR: 0.000139
Epoch 71: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.6646, hm=0.1021, paf=0.0472, cls=0.1079]

Epoch 71 - Train: 0.7287 (HM: 0.1278, PAF: 0.0536, Cls: 0.0736)  LR: 0.000136
Epoch 72: 100% 113/113 [00:29<00:00,  3.81it/s, loss=0.5660, hm=0.1005, paf=0.0436, cls=0.0513]
                                               
Epoch 72 Summary:
  Train - Total: 0.7204, HM: 0.1270, PAF: 0.0526, Cls: 0.0715
  Val   - Total: 1.6580, HM: 0.1549, PAF: 0.0719, Cls: 0.5964
  Metrics - Class Acc: 85.93%, PCK@0.2: 81.59%
  LR: 0.000133
Epoch 73: 100% 113/113 [00:29<00:00,  3.81it/s, loss=0.9385, hm=0.1401, paf=0.0623, cls=0.1690]

Epoch 73 - Train: 0.7183 (HM: 0.1259, PAF: 0.0523, Cls: 0.0734)  LR: 0.000131
Epoch 74: 100% 113/113 [00:29<00:00,  3.80it/s, loss=0.6858, hm=0.1188, paf=0.0549, cls=0.0672]
                                               
Epoch 74 Summary:
  Train - Total: 0.7130, HM: 0.1261, PAF: 0.0519, Cls: 0.0699
  Val   - Total: 1.5718, HM: 0.1535, PAF: 0.0713, Cls: 0.5435
  Metrics - Class Acc: 86.10%, PCK@0.2: 81.31%
  LR: 0.000128
Epoch 75: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.6295, hm=0.1109, paf=0.0505, cls=0.0567]

Epoch 75 - Train: 0.7015 (HM: 0.1244, PAF: 0.0508, Cls: 0.0682)  LR: 0.000126
Epoch 76: 100% 113/113 [00:30<00:00,  3.73it/s, loss=1.1333, hm=0.1582, paf=0.1023, cls=0.1972]
                                               
Epoch 76 Summary:
  Train - Total: 0.7011, HM: 0.1242, PAF: 0.0509, Cls: 0.0683
  Val   - Total: 1.5740, HM: 0.1528, PAF: 0.0718, Cls: 0.5461
  Metrics - Class Acc: 86.06%, PCK@0.2: 81.76%
  LR: 0.000123
Epoch 77: 100% 113/113 [00:29<00:00,  3.82it/s, loss=0.8840, hm=0.1639, paf=0.0556, cls=0.0782]

Epoch 77 - Train: 0.7006 (HM: 0.1236, PAF: 0.0503, Cls: 0.0703)  LR: 0.000120
Epoch 78: 100% 113/113 [00:29<00:00,  3.80it/s, loss=0.6221, hm=0.1157, paf=0.0582, cls=0.0285]
                                               
Epoch 78 Summary:
  Train - Total: 0.6926, HM: 0.1232, PAF: 0.0500, Cls: 0.0665
  Val   - Total: 1.5290, HM: 0.1519, PAF: 0.0706, Cls: 0.5200
  Metrics - Class Acc: 86.13%, PCK@0.2: 82.77%
  LR: 0.000118
  💾 Saved best model (val_loss: 1.5290)
Epoch 79: 100% 113/113 [00:29<00:00,  3.83it/s, loss=0.7020, hm=0.1139, paf=0.0701, cls=0.0708]

Epoch 79 - Train: 0.6936 (HM: 0.1230, PAF: 0.0496, Cls: 0.0681)  LR: 0.000115
Epoch 80: 100% 113/113 [00:30<00:00,  3.73it/s, loss=0.8483, hm=0.1354, paf=0.0627, cls=0.1208]
                                               
Epoch 80 Summary:
  Train - Total: 0.6875, HM: 0.1222, PAF: 0.0490, Cls: 0.0670
  Val   - Total: 1.6118, HM: 0.1516, PAF: 0.0699, Cls: 0.5770
  Metrics - Class Acc: 85.75%, PCK@0.2: 82.39%
  LR: 0.000112
Epoch 81: 100% 113/113 [00:29<00:00,  3.84it/s, loss=1.0813, hm=0.1720, paf=0.0825, cls=0.1522]

Epoch 81 - Train: 0.6861 (HM: 0.1220, PAF: 0.0490, Cls: 0.0667)  LR: 0.000110
Epoch 82: 100% 113/113 [00:30<00:00,  3.75it/s, loss=0.9647, hm=0.1756, paf=0.0511, cls=0.1067]
                                               
Epoch 82 Summary:
  Train - Total: 0.6811, HM: 0.1202, PAF: 0.0480, Cls: 0.0696
  Val   - Total: 1.6105, HM: 0.1527, PAF: 0.0710, Cls: 0.5718
  Metrics - Class Acc: 86.08%, PCK@0.2: 81.88%
  LR: 0.000107
Epoch 83: 100% 113/113 [00:30<00:00,  3.74it/s, loss=0.8557, hm=0.1598, paf=0.0651, cls=0.0577]

Epoch 83 - Train: 0.6806 (HM: 0.1209, PAF: 0.0486, Cls: 0.0666)  LR: 0.000105
Epoch 84: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.9621, hm=0.1524, paf=0.0583, cls=0.1573]
                                               
Epoch 84 Summary:
  Train - Total: 0.6740, HM: 0.1201, PAF: 0.0483, Cls: 0.0647
  Val   - Total: 1.5143, HM: 0.1508, PAF: 0.0693, Cls: 0.5149
  Metrics - Class Acc: 86.70%, PCK@0.2: 82.22%
  LR: 0.000102
  💾 Saved best model (val_loss: 1.5143)
Epoch 85: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.5738, hm=0.1061, paf=0.0400, cls=0.0462]

Epoch 85 - Train: 0.6741 (HM: 0.1196, PAF: 0.0474, Cls: 0.0672)  LR: 0.000100
Epoch 86: 100% 113/113 [00:29<00:00,  3.80it/s, loss=0.8933, hm=0.1591, paf=0.0629, cls=0.0874]
                                               
Epoch 86 Summary:
  Train - Total: 0.6670, HM: 0.1195, PAF: 0.0470, Cls: 0.0634
  Val   - Total: 1.5646, HM: 0.1508, PAF: 0.0697, Cls: 0.5480
  Metrics - Class Acc: 86.04%, PCK@0.2: 82.68%
  LR: 0.000097
Epoch 87: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.9111, hm=0.1804, paf=0.0451, cls=0.0662]

Epoch 87 - Train: 0.6604 (HM: 0.1187, PAF: 0.0465, Cls: 0.0619)  LR: 0.000095
Epoch 88: 100% 113/113 [00:29<00:00,  3.82it/s, loss=0.5142, hm=0.1018, paf=0.0337, cls=0.0264]

Epoch 88 Summary:
  Train - Total: 0.6518, HM: 0.1175, PAF: 0.0463, Cls: 0.0595
  Val   - Total: 1.6397, HM: 0.1495, PAF: 0.0692, Cls: 0.6021
  Metrics - Class Acc: 86.13%, PCK@0.2: 82.79%
  LR: 0.000092
Epoch 89: 100% 113/113 [00:30<00:00,  3.65it/s, loss=0.7551, hm=0.1340, paf=0.0469, cls=0.0836]

Epoch 89 - Train: 0.6635 (HM: 0.1186, PAF: 0.0462, Cls: 0.0644)  LR: 0.000090
Epoch 90: 100% 113/113 [00:29<00:00,  3.77it/s, loss=1.0153, hm=0.1641, paf=0.0921, cls=0.1166]
                                               
Epoch 90 Summary:
  Train - Total: 0.6498, HM: 0.1170, PAF: 0.0453, Cls: 0.0607
  Val   - Total: 1.6260, HM: 0.1494, PAF: 0.0678, Cls: 0.5953
  Metrics - Class Acc: 85.94%, PCK@0.2: 82.51%
  LR: 0.000087
Epoch 91: 100% 113/113 [00:29<00:00,  3.77it/s, loss=0.7753, hm=0.1183, paf=0.0729, cls=0.1042]

Epoch 91 - Train: 0.6541 (HM: 0.1169, PAF: 0.0458, Cls: 0.0634)  LR: 0.000085
Epoch 92: 100% 113/113 [00:29<00:00,  3.78it/s, loss=1.0578, hm=0.1616, paf=0.0942, cls=0.1486]
                                               
Epoch 92 Summary:
  Train - Total: 0.6561, HM: 0.1168, PAF: 0.0460, Cls: 0.0647
  Val   - Total: 1.6487, HM: 0.1485, PAF: 0.0696, Cls: 0.6104
  Metrics - Class Acc: 86.33%, PCK@0.2: 83.02%
  LR: 0.000082
Epoch 93: 100% 113/113 [00:30<00:00,  3.70it/s, loss=0.8247, hm=0.1294, paf=0.0635, cls=0.1200]

Epoch 93 - Train: 0.6429 (HM: 0.1156, PAF: 0.0448, Cls: 0.0605)  LR: 0.000080
Epoch 94: 100% 113/113 [00:30<00:00,  3.74it/s, loss=1.0937, hm=0.1330, paf=0.0600, cls=0.2945]
                                               
Epoch 94 Summary:
  Train - Total: 0.6497, HM: 0.1158, PAF: 0.0448, Cls: 0.0646
  Val   - Total: 1.6014, HM: 0.1474, PAF: 0.0685, Cls: 0.5832
  Metrics - Class Acc: 86.07%, PCK@0.2: 82.51%
  LR: 0.000077
Epoch 95: 100% 113/113 [00:29<00:00,  3.78it/s, loss=0.6767, hm=0.1033, paf=0.0521, cls=0.1061]

Epoch 95 - Train: 0.6485 (HM: 0.1155, PAF: 0.0448, Cls: 0.0647)  LR: 0.000075
Epoch 96: 100% 113/113 [00:30<00:00,  3.71it/s, loss=0.6262, hm=0.1005, paf=0.0432, cls=0.0920]
                                               
Epoch 96 Summary:
  Train - Total: 0.6433, HM: 0.1154, PAF: 0.0441, Cls: 0.0624
  Val   - Total: 1.6024, HM: 0.1474, PAF: 0.0669, Cls: 0.5861
  Metrics - Class Acc: 86.36%, PCK@0.2: 83.34%
  LR: 0.000072
Epoch 97: 100% 113/113 [00:31<00:00,  3.63it/s, loss=0.9315, hm=0.1517, paf=0.0716, cls=0.1210]

Epoch 97 - Train: 0.6464 (HM: 0.1151, PAF: 0.0447, Cls: 0.0646)  LR: 0.000070
Epoch 98: 100% 113/113 [00:30<00:00,  3.66it/s, loss=0.6023, hm=0.1053, paf=0.0512, cls=0.0524]
                                               
Epoch 98 Summary:
  Train - Total: 0.6315, HM: 0.1140, PAF: 0.0439, Cls: 0.0584
  Val   - Total: 1.6115, HM: 0.1475, PAF: 0.0668, Cls: 0.5919
  Metrics - Class Acc: 86.17%, PCK@0.2: 82.94%
  LR: 0.000068
Epoch 99: 100% 113/113 [00:30<00:00,  3.70it/s, loss=0.5645, hm=0.1044, paf=0.0531, cls=0.0271]

Epoch 99 - Train: 0.6259 (HM: 0.1136, PAF: 0.0434, Cls: 0.0563)  LR: 0.000066
Epoch 100: 100% 113/113 [00:30<00:00,  3.69it/s, loss=0.7957, hm=0.1341, paf=0.0393, cls=0.1205]
                                               
Epoch 100 Summary:
  Train - Total: 0.6286, HM: 0.1134, PAF: 0.0434, Cls: 0.0588
  Val   - Total: 1.5899, HM: 0.1465, PAF: 0.0663, Cls: 0.5810
  Metrics - Class Acc: 86.27%, PCK@0.2: 83.17%
  LR: 0.000063
Epoch 101: 100% 113/113 [00:31<00:00,  3.61it/s, loss=0.4977, hm=0.0912, paf=0.0371, cls=0.0391]
