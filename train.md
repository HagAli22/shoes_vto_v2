================================================================================
👟 ARShoe M1v2 13-Keypoint Fresh Training Pipeline
================================================================================
  Target Dataset Root   : /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp
  PAF Configuration     : /content/shoes_vto_v2/shoes_vto/configs/paf_connections_13kp.yaml
  Output Directory      : /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training
  Total Epochs          : 100
  Learning Rate         : 0.0005 (Cosine decay -> 1e-06)
  Batch Size            : 16
  Workers               : 4
  Mixed Precision (AMP) : True
  Pretrained Backbone   : True
================================================================================

📁 Loading datasets...
Loaded 1285 images from /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp/train/images
Loaded 161 images from /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp/valid/images
  Train: 1285 images
  Val  : 161 images

🏗️  Initializing ARShoe M1v2 (13-KP) Architecture...
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
100% 9.83M/9.83M [00:00<00:00, 167MB/s]
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

🚀 Launching Fresh 13-KP M1v2 Training for 100 epochs...

================================================================================
Training ARShoe M1 for 100 epochs
Device: cuda
Mixed Precision (AMP): Enabled
Train samples: 1285
Val samples: 161
Batch size: 16
Learning rate: 0.0005
Loss weights: {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
================================================================================
Epoch 1:   0% 0/81 [00:00<?, ?it/s]/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
W1024 09:56:09.062000 7350 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1:   1% 1/81 [02:10<2:54:32, 130.90s/it, loss=11.9166, hm=2.5309, paf=0.2925, cls=0.8053]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 1:  99% 80/81 [02:38<00:00,  4.13it/s, loss=7.0608, hm=1.3845, paf=0.2531, cls=0.6778]/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Epoch 1: 100% 81/81 [04:36<00:00,  3.41s/it, loss=6.7542, hm=1.3094, paf=0.2428, cls=0.6872]

Epoch 1 - Train: 9.2469 (HM: 1.9129, PAF: 0.2686, Cls: 0.7054)  LR: 0.000250
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(

  🖼️ [Epoch 1] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_001_vis.jpg
<IPython.core.display.Image object>
Epoch 2:   0% 0/81 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 2: 100% 81/81 [00:37<00:00,  2.18it/s, loss=4.0539, hm=0.6281, paf=0.2576, cls=0.6841]
Validation:   0% 0/11 [00:00<?, ?it/s]/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
Validation:  91% 10/11 [00:33<00:00,  1.62it/s]/usr/local/lib/python3.13/dist-packages/torch/_inductor/lowering.py:7836: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
                                               
Epoch 2 Summary:
  Train - Total: 5.3688, HM: 0.9587, PAF: 0.2446, Cls: 0.6965
  Val   - Total: 4.0620, HM: 0.6382, PAF: 0.2370, Cls: 0.6902
  Metrics - Class Acc: 53.27%, PCK@0.2: 44.85%
  LR: 0.000250
  💾 Saved best model (val_loss: 4.0620)
Epoch 3:   0% 0/81 [00:00<?, ?it/s]/content/shoes_vto_v2/shoes_vto/src/training/trainer_m1.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.use_amp):
Epoch 3: 100% 81/81 [00:26<00:00,  3.10it/s, loss=2.9706, hm=0.3844, paf=0.2082, cls=0.6778]

Epoch 3 - Train: 3.4500 (HM: 0.4932, PAF: 0.2224, Cls: 0.6882)  LR: 0.000249
Epoch 4: 100% 81/81 [00:23<00:00,  3.40it/s, loss=3.0786, hm=0.4001, paf=0.2074, cls=0.7088]
                                               
Epoch 4 Summary:
  Train - Total: 2.7880, HM: 0.3409, PAF: 0.2046, Cls: 0.6767
  Val   - Total: 2.6840, HM: 0.3166, PAF: 0.2048, Cls: 0.6719
  Metrics - Class Acc: 57.06%, PCK@0.2: 55.71%
  LR: 0.000249
  💾 Saved best model (val_loss: 2.6840)
Epoch 5: 100% 81/81 [00:23<00:00,  3.48it/s, loss=2.4585, hm=0.3032, paf=0.1948, cls=0.5708]

Epoch 5 - Train: 2.4732 (HM: 0.2855, PAF: 0.1902, Cls: 0.6339)  LR: 0.000248
Epoch 6: 100% 81/81 [00:23<00:00,  3.52it/s, loss=2.3989, hm=0.2730, paf=0.2048, cls=0.5982]
                                               
Epoch 6 Summary:
  Train - Total: 2.2924, HM: 0.2607, PAF: 0.1794, Cls: 0.5938
  Val   - Total: 2.2373, HM: 0.2611, PAF: 0.1723, Cls: 0.5655
  Metrics - Class Acc: 69.79%, PCK@0.2: 62.39%
  LR: 0.000248
  💾 Saved best model (val_loss: 2.2373)
Epoch 7: 100% 81/81 [00:22<00:00,  3.68it/s, loss=2.3222, hm=0.2660, paf=0.1754, cls=0.6049]

Epoch 7 - Train: 2.1104 (HM: 0.2438, PAF: 0.1696, Cls: 0.5308)  LR: 0.000247
Epoch 8: 100% 81/81 [00:22<00:00,  3.63it/s, loss=1.9804, hm=0.2611, paf=0.1591, cls=0.4120]
                                               
Epoch 8 Summary:
  Train - Total: 1.9756, HM: 0.2343, PAF: 0.1614, Cls: 0.4771
  Val   - Total: 2.0002, HM: 0.2435, PAF: 0.1619, Cls: 0.4682
  Metrics - Class Acc: 78.17%, PCK@0.2: 63.84%
  LR: 0.000246
  💾 Saved best model (val_loss: 2.0002)
Epoch 9: 100% 81/81 [00:21<00:00,  3.71it/s, loss=1.5511, hm=0.1994, paf=0.1138, cls=0.3505]

Epoch 9 - Train: 1.8542 (HM: 0.2261, PAF: 0.1530, Cls: 0.4291)  LR: 0.000245
Epoch 10: 100% 81/81 [00:21<00:00,  3.80it/s, loss=1.6807, hm=0.1917, paf=0.1458, cls=0.4149]
                                               
Epoch 10 Summary:
  Train - Total: 1.7557, HM: 0.2201, PAF: 0.1476, Cls: 0.3868
  Val   - Total: 1.8345, HM: 0.2324, PAF: 0.1537, Cls: 0.3984
  Metrics - Class Acc: 80.60%, PCK@0.2: 65.44%
  LR: 0.000244
  💾 Saved best model (val_loss: 1.8345)

  🖼️ [Epoch 10] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_010_vis.jpg
<IPython.core.display.Image object>
Epoch 11: 100% 81/81 [00:21<00:00,  3.80it/s, loss=1.6530, hm=0.2153, paf=0.1324, cls=0.3515]

Epoch 11 - Train: 1.6766 (HM: 0.2155, PAF: 0.1425, Cls: 0.3530)  LR: 0.000243
Epoch 12: 100% 81/81 [00:21<00:00,  3.80it/s, loss=1.4861, hm=0.2006, paf=0.1435, cls=0.2644]
                                               
Epoch 12 Summary:
  Train - Total: 1.6161, HM: 0.2115, PAF: 0.1377, Cls: 0.3300
  Val   - Total: 1.7361, HM: 0.2275, PAF: 0.1459, Cls: 0.3563
  Metrics - Class Acc: 83.01%, PCK@0.2: 67.40%
  LR: 0.000241
  💾 Saved best model (val_loss: 1.7361)
Epoch 13: 100% 81/81 [00:21<00:00,  3.81it/s, loss=1.5825, hm=0.2097, paf=0.1420, cls=0.3065]

Epoch 13 - Train: 1.5346 (HM: 0.2087, PAF: 0.1333, Cls: 0.2889)  LR: 0.000240
Epoch 14: 100% 81/81 [00:20<00:00,  3.86it/s, loss=1.5382, hm=0.2022, paf=0.1565, cls=0.2775]

Epoch 14 Summary:
  Train - Total: 1.4836, HM: 0.2045, PAF: 0.1304, Cls: 0.2698
  Val   - Total: 1.6948, HM: 0.2249, PAF: 0.1407, Cls: 0.3426
  Metrics - Class Acc: 84.10%, PCK@0.2: 67.20%
  LR: 0.000238
  💾 Saved best model (val_loss: 1.6948)
Epoch 15: 100% 81/81 [00:21<00:00,  3.83it/s, loss=1.4351, hm=0.2178, paf=0.1052, cls=0.2357]

Epoch 15 - Train: 1.4303 (HM: 0.2014, PAF: 0.1267, Cls: 0.2475)  LR: 0.000236
Epoch 16: 100% 81/81 [00:20<00:00,  3.86it/s, loss=2.2108, hm=0.2894, paf=0.1410, cls=0.5142]
                                               
Epoch 16 Summary:
  Train - Total: 1.3903, HM: 0.2009, PAF: 0.1226, Cls: 0.2278
  Val   - Total: 1.7261, HM: 0.2148, PAF: 0.1385, Cls: 0.3934
  Metrics - Class Acc: 83.04%, PCK@0.2: 69.68%
  LR: 0.000235
Epoch 17: 100% 81/81 [00:20<00:00,  3.86it/s, loss=1.3339, hm=0.1928, paf=0.1331, cls=0.1978]

Epoch 17 - Train: 1.3535 (HM: 0.1968, PAF: 0.1217, Cls: 0.2152)  LR: 0.000233
Epoch 18: 100% 81/81 [00:20<00:00,  3.89it/s, loss=1.2399, hm=0.1990, paf=0.1069, cls=0.1534]
                                               
Epoch 18 Summary:
  Train - Total: 1.3214, HM: 0.1956, PAF: 0.1198, Cls: 0.1995
  Val   - Total: 1.6906, HM: 0.2132, PAF: 0.1296, Cls: 0.3856
  Metrics - Class Acc: 84.82%, PCK@0.2: 70.48%
  LR: 0.000231
  💾 Saved best model (val_loss: 1.6906)
Epoch 19: 100% 81/81 [00:20<00:00,  3.92it/s, loss=1.7612, hm=0.2462, paf=0.1952, cls=0.2574]

Epoch 19 - Train: 1.2867 (HM: 0.1939, PAF: 0.1171, Cls: 0.1846)  LR: 0.000228
Epoch 20: 100% 81/81 [00:20<00:00,  3.86it/s, loss=3.5515, hm=0.2217, paf=0.1472, cls=1.5804]
                                               
Epoch 20 Summary:
  Train - Total: 1.2783, HM: 0.1913, PAF: 0.1128, Cls: 0.1917
  Val   - Total: 1.6080, HM: 0.2107, PAF: 0.1277, Cls: 0.3399
  Metrics - Class Acc: 86.52%, PCK@0.2: 70.64%
  LR: 0.000226
  💾 Saved best model (val_loss: 1.6080)

  🖼️ [Epoch 20] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_020_vis.jpg
<IPython.core.display.Image object>
Epoch 21: 100% 81/81 [00:21<00:00,  3.84it/s, loss=1.5617, hm=0.2527, paf=0.1013, cls=0.2321]

Epoch 21 - Train: 1.2495 (HM: 0.1912, PAF: 0.1114, Cls: 0.1746)  LR: 0.000224
Epoch 22: 100% 81/81 [00:21<00:00,  3.84it/s, loss=1.3114, hm=0.1790, paf=0.1131, cls=0.2461]
                                               
Epoch 22 Summary:
  Train - Total: 1.2120, HM: 0.1884, PAF: 0.1091, Cls: 0.1600
  Val   - Total: 1.5907, HM: 0.2070, PAF: 0.1186, Cls: 0.3503
  Metrics - Class Acc: 87.28%, PCK@0.2: 71.29%
  LR: 0.000221
  💾 Saved best model (val_loss: 1.5907)
Epoch 23: 100% 81/81 [00:20<00:00,  3.89it/s, loss=0.9224, hm=0.1643, paf=0.0871, cls=0.0607]

Epoch 23 - Train: 1.1916 (HM: 0.1862, PAF: 0.1063, Cls: 0.1562)  LR: 0.000219
Epoch 24: 100% 81/81 [00:20<00:00,  3.87it/s, loss=1.2852, hm=0.1837, paf=0.1461, cls=0.1721]
                                               
Epoch 24 Summary:
  Train - Total: 1.1624, HM: 0.1841, PAF: 0.1042, Cls: 0.1452
  Val   - Total: 1.6077, HM: 0.2028, PAF: 0.1173, Cls: 0.3747
  Metrics - Class Acc: 86.71%, PCK@0.2: 71.49%
  LR: 0.000216
Epoch 25: 100% 81/81 [00:21<00:00,  3.85it/s, loss=1.4112, hm=0.2056, paf=0.1173, cls=0.2361]

Epoch 25 - Train: 1.1484 (HM: 0.1835, PAF: 0.1027, Cls: 0.1392)  LR: 0.000214
Epoch 26: 100% 81/81 [00:20<00:00,  3.90it/s, loss=1.2559, hm=0.2124, paf=0.1006, cls=0.1366]
                                               
Epoch 26 Summary:
  Train - Total: 1.1344, HM: 0.1818, PAF: 0.1026, Cls: 0.1346
  Val   - Total: 1.5718, HM: 0.2007, PAF: 0.1148, Cls: 0.3597
  Metrics - Class Acc: 87.17%, PCK@0.2: 73.49%
  LR: 0.000211
  💾 Saved best model (val_loss: 1.5718)
Epoch 27: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.1604, hm=0.1583, paf=0.0897, cls=0.2319]

Epoch 27 - Train: 1.1183 (HM: 0.1788, PAF: 0.1000, Cls: 0.1354)  LR: 0.000208
Epoch 28: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.2939, hm=0.1979, paf=0.0940, cls=0.2095]
                                               
Epoch 28 Summary:
  Train - Total: 1.1005, HM: 0.1786, PAF: 0.0986, Cls: 0.1260
  Val   - Total: 1.5335, HM: 0.1980, PAF: 0.1072, Cls: 0.3514
  Metrics - Class Acc: 88.74%, PCK@0.2: 73.53%
  LR: 0.000205
  💾 Saved best model (val_loss: 1.5335)
Epoch 29: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.8934, hm=0.1712, paf=0.1386, cls=0.6210]

Epoch 29 - Train: 1.0933 (HM: 0.1770, PAF: 0.0953, Cls: 0.1297)  LR: 0.000202
Epoch 30: 100% 81/81 [00:20<00:00,  3.89it/s, loss=2.4411, hm=0.2472, paf=0.1282, cls=0.7974]
                                               
Epoch 30 Summary:
  Train - Total: 1.1012, HM: 0.1771, PAF: 0.0954, Cls: 0.1347
  Val   - Total: 1.5284, HM: 0.1961, PAF: 0.1057, Cls: 0.3551
  Metrics - Class Acc: 88.17%, PCK@0.2: 73.33%
  LR: 0.000199
  💾 Saved best model (val_loss: 1.5284)

  🖼️ [Epoch 30] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_030_vis.jpg
<IPython.core.display.Image object>
Epoch 31: 100% 81/81 [00:20<00:00,  3.87it/s, loss=1.4294, hm=0.2219, paf=0.1278, cls=0.1907]

Epoch 31 - Train: 1.0717 (HM: 0.1761, PAF: 0.0933, Cls: 0.1204)  LR: 0.000195
Epoch 32: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.0891, hm=0.1920, paf=0.0871, cls=0.0981]
                                               
Epoch 32 Summary:
  Train - Total: 1.0538, HM: 0.1730, PAF: 0.0903, Cls: 0.1207
  Val   - Total: 1.4744, HM: 0.1956, PAF: 0.1024, Cls: 0.3247
  Metrics - Class Acc: 88.39%, PCK@0.2: 73.81%
  LR: 0.000192
  💾 Saved best model (val_loss: 1.4744)
Epoch 33: 100% 81/81 [00:20<00:00,  3.87it/s, loss=0.8718, hm=0.1583, paf=0.0906, cls=0.0382]

Epoch 33 - Train: 1.0365 (HM: 0.1735, PAF: 0.0886, Cls: 0.1101)  LR: 0.000189
Epoch 34: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.2953, hm=0.2388, paf=0.0726, cls=0.1298]
                                               
Epoch 34 Summary:
  Train - Total: 1.0182, HM: 0.1721, PAF: 0.0877, Cls: 0.1029
  Val   - Total: 1.5290, HM: 0.1928, PAF: 0.1016, Cls: 0.3698
  Metrics - Class Acc: 88.42%, PCK@0.2: 73.81%
  LR: 0.000185
Epoch 35: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.3325, hm=0.1925, paf=0.1291, cls=0.2030]

Epoch 35 - Train: 1.0004 (HM: 0.1703, PAF: 0.0855, Cls: 0.0988)  LR: 0.000182
Epoch 36: 100% 81/81 [00:21<00:00,  3.86it/s, loss=1.3408, hm=0.2084, paf=0.1245, cls=0.1721]
                                               
Epoch 36 Summary:
  Train - Total: 0.9997, HM: 0.1689, PAF: 0.0853, Cls: 0.1025
  Val   - Total: 1.4907, HM: 0.1915, PAF: 0.0961, Cls: 0.3550
  Metrics - Class Acc: 88.66%, PCK@0.2: 73.81%
  LR: 0.000179
Epoch 37: 100% 81/81 [00:20<00:00,  3.90it/s, loss=0.8495, hm=0.1414, paf=0.0691, cls=0.0971]

Epoch 37 - Train: 0.9842 (HM: 0.1667, PAF: 0.0833, Cls: 0.1005)  LR: 0.000175
Epoch 38: 100% 81/81 [00:20<00:00,  3.88it/s, loss=1.2998, hm=0.1908, paf=0.1165, cls=0.2023]
                                               
Epoch 38 Summary:
  Train - Total: 0.9796, HM: 0.1667, PAF: 0.0833, Cls: 0.0975
  Val   - Total: 1.5113, HM: 0.1906, PAF: 0.0968, Cls: 0.3702
  Metrics - Class Acc: 88.20%, PCK@0.2: 73.85%
  LR: 0.000171
Epoch 39: 100% 81/81 [00:20<00:00,  3.90it/s, loss=1.9647, hm=0.1576, paf=0.0905, cls=0.7689]

Epoch 39 - Train: 0.9851 (HM: 0.1651, PAF: 0.0826, Cls: 0.1065)  LR: 0.000168
Epoch 40: 100% 81/81 [00:20<00:00,  3.88it/s, loss=0.9425, hm=0.1758, paf=0.0819, cls=0.0503]
                                               
Epoch 40 Summary:
  Train - Total: 0.9619, HM: 0.1641, PAF: 0.0802, Cls: 0.0967
  Val   - Total: 1.5059, HM: 0.1899, PAF: 0.0953, Cls: 0.3706
  Metrics - Class Acc: 88.18%, PCK@0.2: 73.97%
  LR: 0.000164

  🖼️ [Epoch 40] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_040_vis.jpg
<IPython.core.display.Image object>
Epoch 41: 100% 81/81 [00:20<00:00,  3.89it/s, loss=0.8770, hm=0.1503, paf=0.1133, cls=0.0327]

Epoch 41 - Train: 0.9466 (HM: 0.1627, PAF: 0.0802, Cls: 0.0903)  LR: 0.000160
Epoch 42: 100% 81/81 [00:21<00:00,  3.85it/s, loss=1.0293, hm=0.1738, paf=0.0882, cls=0.1050]
                                               
Epoch 42 Summary:
  Train - Total: 0.9432, HM: 0.1613, PAF: 0.0768, Cls: 0.0961
  Val   - Total: 1.4683, HM: 0.1882, PAF: 0.0916, Cls: 0.3548
  Metrics - Class Acc: 89.17%, PCK@0.2: 74.37%
  LR: 0.000156
  💾 Saved best model (val_loss: 1.4683)
Epoch 43: 100% 81/81 [00:20<00:00,  3.89it/s, loss=1.2973, hm=0.2271, paf=0.0854, cls=0.1455]

Epoch 43 - Train: 0.9375 (HM: 0.1608, PAF: 0.0766, Cls: 0.0940)  LR: 0.000153
Epoch 44: 100% 81/81 [00:21<00:00,  3.82it/s, loss=1.5466, hm=0.2394, paf=0.1947, cls=0.1331]
                                               
Epoch 44 Summary:
  Train - Total: 0.9242, HM: 0.1596, PAF: 0.0764, Cls: 0.0886
  Val   - Total: 1.4642, HM: 0.1846, PAF: 0.0902, Cls: 0.3635
  Metrics - Class Acc: 88.76%, PCK@0.2: 75.01%
  LR: 0.000149
  💾 Saved best model (val_loss: 1.4642)
Epoch 45: 100% 81/81 [00:21<00:00,  3.77it/s, loss=0.9093, hm=0.1630, paf=0.0792, cls=0.0659]

Epoch 45 - Train: 0.9104 (HM: 0.1580, PAF: 0.0740, Cls: 0.0870)  LR: 0.000145
Epoch 46: 100% 81/81 [00:21<00:00,  3.78it/s, loss=1.7424, hm=0.2998, paf=0.1268, cls=0.1931]
                                               
Epoch 46 Summary:
  Train - Total: 0.9138, HM: 0.1585, PAF: 0.0751, Cls: 0.0864
  Val   - Total: 1.4677, HM: 0.1839, PAF: 0.0885, Cls: 0.3700
  Metrics - Class Acc: 88.70%, PCK@0.2: 75.45%
  LR: 0.000141
Epoch 47: 100% 81/81 [00:21<00:00,  3.80it/s, loss=1.0160, hm=0.1722, paf=0.0530, cls=0.1475]

Epoch 47 - Train: 0.8978 (HM: 0.1561, PAF: 0.0728, Cls: 0.0852)  LR: 0.000137
Epoch 48: 100% 81/81 [00:21<00:00,  3.82it/s, loss=0.8526, hm=0.1457, paf=0.0700, cls=0.0864]
                                               
Epoch 48 Summary:
  Train - Total: 0.8877, HM: 0.1546, PAF: 0.0715, Cls: 0.0842
  Val   - Total: 1.4679, HM: 0.1808, PAF: 0.0880, Cls: 0.3791
  Metrics - Class Acc: 88.25%, PCK@0.2: 75.45%
  LR: 0.000133
Epoch 49: 100% 81/81 [00:21<00:00,  3.84it/s, loss=1.0904, hm=0.1703, paf=0.1018, cls=0.1369]

Epoch 49 - Train: 0.8727 (HM: 0.1529, PAF: 0.0706, Cls: 0.0799)  LR: 0.000129
Epoch 50: 100% 81/81 [00:21<00:00,  3.81it/s, loss=0.9599, hm=0.1616, paf=0.0761, cls=0.1077]
                                               
Epoch 50 Summary:
  Train - Total: 0.8761, HM: 0.1528, PAF: 0.0719, Cls: 0.0807
  Val   - Total: 1.4787, HM: 0.1803, PAF: 0.0852, Cls: 0.3915
  Metrics - Class Acc: 88.71%, PCK@0.2: 76.97%
  LR: 0.000126

  🖼️ [Epoch 50] Saved test visualization -> /content/shoes_vto_v2/shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/epoch_visualizations/epoch_050_vis.jpg
<IPython.core.display.Image object>
Epoch 51: 100% 81/81 [00:21<00:00,  3.81it/s, loss=0.9011, hm=0.1400, paf=0.1001, cls=0.0938]

Epoch 51 - Train: 0.8600 (HM: 0.1517, PAF: 0.0695, Cls: 0.0761)  LR: 0.000122
Epoch 52: 100% 81/81 [00:21<00:00,  3.78it/s, loss=0.9159, hm=0.1375, paf=0.0810, cls=0.1360]
                                               
Epoch 52 Summary:
  Train - Total: 0.8521, HM: 0.1511, PAF: 0.0674, Cls: 0.0753
  Val   - Total: 1.4171, HM: 0.1779, PAF: 0.0857, Cls: 0.3561
  Metrics - Class Acc: 89.09%, PCK@0.2: 76.45%
  LR: 0.000118
  💾 Saved best model (val_loss: 1.4171)
Epoch 53: 100% 81/81 [00:21<00:00,  3.82it/s, loss=0.8481, hm=0.1405, paf=0.0558, cls=0.1163]

Epoch 53 - Train: 0.8550 (HM: 0.1497, PAF: 0.0670, Cls: 0.0814)  LR: 0.000114
Epoch 54:  47% 38/81 [00:10<00:11,  3.61it/s, loss=0.8035, hm=0.1392, paf=0.0854, cls=0.0506]