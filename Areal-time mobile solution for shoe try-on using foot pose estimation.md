Complex & Intelligent Systems (2026) 12:65 https://doi.org/10.1007/s40747-025-02188-x 

**~~ORIGINAL ARTICLE~~** 



# **A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques** 

**Nguyen Hoang Vu**<sup>**1**</sup> **· Tran Van Duc**<sup>**1**</sup> **· Pham Quang Tien**<sup>**1**</sup> **· Nguyen Thi Ngoc Anh**<sup>**2**</sup> **· Nguyen Tien Dat**<sup>**1**</sup> 

Received: 2 January 2024 / Accepted: 18 November 2025 / Published online: 5 December 2025 © The Author(s) 2025 

#### **Abstract** 

Implementing Augmented Reality (AR) in virtual try-on technology has revolutionized the online shopping experience, transforming how clients engage with products. This technology allows customers to try on clothes without direct physical contact, which has become convenient and valuable in the age of online shopping. Despite clothing being a dominant sector, shoes also hold a significant portion of the market. However, there has been limited previous research conducted on the subject of virtual shoe try-on. This paper presents an innovative solution to the Foot Pose Estimation problem, offering a deep learning model that delivers accurate results while operating in real-time on the CPU. Due to the insufficiency of public foot keypoint datasets, a medium-scale self-collected 2D foot keypoints dataset has been conducted with 9 keypoints each foot for training and evaluating the model. In addition, this model has successfully been utilized to create a shoe AR try-on application for smartphones. It provides a solution that generates a realistic 3D shoe model for a smooth and stable try-on experience. Practical tests have proven that the system gives real-time performance under mobile computing conditions. 

**Keywords** Foot pose estimation · Real-time keypoint detection · Virtual try-on 

## **Introduction** 

COVID-19 pandemic has greatly contributed to the substantial increase in online shopping, leading to a significant shift in consumer behavior in recent times. One important part of online shopping is that customers want to see themselves wearing clothes and accessories. This creates a connection between the virtual world and the real world. In response to this demand, the development of virtual try-on systems has gained substantial momentum. These systems offer a dynamic platform for customers to simulate the act of trying on various clothing and accessory items. This not only offers a novel and efficient shopping experience but also contributes to reducing the rate of returns, a vexing concern for online retailers. While substantial research has been directed 

B Nguyen Tien Dat datnt65@viettel.com.vn 

Nguyen Thi Ngoc Anh anhnguyenngoc@vnu.edu.vn 

> 1 Viettel High Technology Industries Corporation, Hanoi 100000, Vietnam 

> 2 VNU University of Engineering and Technology, Hanoi 100000, Vietnam 

towards image-based virtual try-on (VTON) techniques [1– 6], the focus predominantly remains centered on clothing, with only a handful of studies exploring the virtual try-on of footwear, such as PITONS [7], which seeks to replicate shoe samples onto photographs featuring the wearer’s feet, using the Generative model as the foundation. However, the computational resource-intensive nature of such models presents a notable barrier to accessibility. 

In the field of the shoe market, shoe AR try-On technology has become an emerging solution that addresses users’ online shoe-shopping needs. This technology allows shoppers to try on shoes without physical impact—which is extremely convenient and useful in the online era. This requires finding the right position and orientation of the user’s foot in front of the camera to render the 3D shoe model, which includes the camera’s rotation and translation matrix. In recent years, some world-wide brands such as Adidas, Nike, and Dior have been investing in shoe AR try-on technology to apply into their websites or mobile applications. Beside that, a start-up company called Wanna,<sup>1</sup> which is founded in 2018, introduced their implementation of 3D and AR technologies in shoe AR try-on with a demo website and soon spread 

> 1 https://wanna.fashion/. 

123 

**65** Page 2 of 22 

Complex & Intelligent Systems (2026) 12 :65 

on multiple platforms including iOS, Android, and WeChat mini-program. Industrial researches on shoe AR try-on technology conducted by world-wide brands and start-ups prove the important this technology in the shoe market. 

Traditionally, foot pose estimation has been achieved through a marker-based motion capture system, that uses sensors to measure and capture the information related to foot movement. Duong et al. [4] proposed a methodology that combines low-cost distance sensors and an inertial sensor unit to estimate foot pose. This approach requires special hardware as well as sensor settings which is also difficult to apply. Furthermore, since most users shop online through mobile devices, developing applications on mobile platforms with real-time processing capabilities is needed for the user’s experience. 

In the past few years, deep learning came up as a general solution for this foot pose estimation problem. In the recent publication [8], the authors present a method for estimating foot pose based on the correlation between 2D and 3D foot keypoints. This approach uses a convolutional neural network (CNN) model capable of extracting precisely 2D-foot keypoint in real-time before using the Perspective-n-Point (PnP) algorithm applied to estimate foot pose. The biggest obstacletothisresearchisthelackofasuitablequalitydataset of 2D foot keypoint, which is not referenced by the author. The available public dataset that includes human foot keypoints, extracted from the original COCO [9] dataset, falls short of adequately addressing the problem. The dataset is not large enough and the quality of the foot keypoints is not good enough to train a deep learning model. Therefore, the authors had to use a 3D Render Tool to create a synthetic dataset of 2D foot keypoints. This dataset is used to train a deep learning model that can extract 2D foot keypoints in real-time. The model is then evaluated on a self-built 2D foot keypoint dataset. The results show that the model can extract 2D foot keypoints in real-time with high accuracy. 

To make the virtual try-on realistic, it’s not only about getting the foot position and direction right but also how the shoes cover the foot naturally. Moreover, the existence of jitter in a real-time system can affect the user’s experience, therefore, a stabilization method is needed to handle this problem. In this work, a deep learning model is proposed with the capability of extracting 2D foot keypoints in realtime on CPU which was trained and evaluated on a self-built 2D foot keypoint dataset. Also to achieve a realistic occlusion, Ray-Casting algorithm is utilized for the virtual camera and 3D shoe model to find the shoe area obscured by the 3D foot model. After that, an alpha-beta filter is applied with a custom Intersection Over Union (IOU) threshold to achieve a simple and efficient stabilization effect that can smooth and eliminate jitters. However, due to the the high product cost of 3D shoe models, our proposed method are applied on a 

limited number of 3D shoe models, which is able to affect the qualitative results in experiments. 

In summary, this research provides three main contributions: (1) a self-built 2D foot keypoints dataset based on a 3D Render Tool Support for labeling and checking data annotation process; (2) A 2D keypoints detection model that works accuratelyinreal-time;(3)Theformulationofamethodology to emulate realistic occlusion in 3D shoe models combining with a mechanism that stabilizes a real-time shoe try-on system, which effectively fixes issues caused by jitter. Additionally, an implementation of a real-time Shoe AR Try-on application on mobile devices improving the user experience when shopping online, which has user interface (UI) illustrated on Fig. 1. This research is divided into five main parts: section “Related work” summarizes previous works on shoe virtual try-on; our self-built foot keypoint dataset is introduced in section “Self-built dataset”; the proposed shoe virtual try-on system is provided in section “Shoe virtual try-on system”; section “Experiment” explains our experimental settings and evaluates our results in both quantitative and quanlitative outcomes; section“Discussions and future work” concludes the paper and opens future works. 

## **Related work** 

### **Keypoints dataset** 

Over the past ten years, many human keypoint datasets have been publicly available and introduced. One of the typical datasets is UIUC [10] data with 539 images (346 for training, 247 for testing), which describe personal data under certain activities such as walking, standing, and the main activity playing badminton. These are very diverse poses, but the activities number is quite limited. Sport Image [11] sports data is more abundant with the addition of sports activities such as football, golf, horseback riding,..., etc. The total number of images in this dataset is 1299 images (649 training, 650 experiments). Some larger datasets like MPII Human Pose [12] include 24,589 images having more than 28,000 labeled people in the training set. MS-COCO dataset [9] 2017 includes 118,287 training images and 5000 images in the evaluation set, totaling more than 150,000 people with approximately 1.7 million keypoints. This dataset provides multiple points on the body part consisting of three keypoints per foot. Despite considerable human pose datasets, the number of foot keypoints is limited and difficult to find in the publicly available data. 

123 

Page 3 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
ses) 4) a4) Fa)4)<br><!-- End of picture text -->

**Fig. 1** Shoe AR try-on application UI 

### **Pose estimation** 

#### **2D pose estimation** 

In the context of the human body, 2D pose estimation involves determining the positions of 2D keypoints from an image. There are two common approaches to tackle this challenge: heatmap prediction-based and coordinate regressionbased methods. The coordinate regression-based approach directly predicts the coordinates of matching points using a nonlinear function. However, this method has received less attention recently due to the challenges in developing a generalized nonlinear function. On the other hand, the heatmap prediction-based method indirectly predicts an imageheatmapandthenextractsthecorrespondingkeypoints from it. This approach has garnered significant interest and extensive study, especially for human pose estimation problems. Various studies [13–15] have utilized CNN to learn image features and generate heatmaps as outputs for human pose estimation. Considering the characteristics of these two approaches, the project focuses on solving the 2D foot pose estimation problem using the heatmap prediction-based method. Shan An et al. [8] also used a heatmap predictionbased method in their CNN model to predict keypoints on the 

foot. The effectiveness of the neural network model is highly dependent on the dataset used. In their paper [8], the authors introduced a benchmark dataset containing 86,040 images of human feet, annotated with 8 keypoints for each foot. Unfortunately, no available reference to access this dataset has been provided. Currently, the only public dataset for 2D foot pose estimation is the Human Foot Keypoint Dataset, which includes only three keypoints per foot. The limited quantity and lower quality of foot keypoint data in this dataset present challenges in effectively addressing the 2D foot pose estimation problem. 

#### **3D pose estimation** 

Accurately estimating the 3D pose or 6 degrees of freedom (6DoF) of an object involves determining its rotation and translation matrix with respect to the camera. Previous research has explored various methods for achieving accurate 6DoF poses. A common approach for 6DoF pose estimation consists of a two-stage process. First, 2D keypoints are extracted, and then correspondences between these 2D and 3D keypoints are used to estimate the 3D pose. Methods [16, 17] combine segmentation and CNN to detect the 2D projection of an object’s 3D bounding box and predict its 3D 

123 

**65** Page 4 of 22 

Complex & Intelligent Systems (2026) 12 :65 

pose. Another approach, known as DPOD [18], estimates the 2D-3D correspondences between an image and available 3D models in a multi-class setting. It utilizes PnP and random sample consensus (RANSAC) algorithms to compute these correspondences and incorporates a custom deep learning-based refinement scheme to further enhance the initial pose estimates. Generally, the above approaches utilize CNN models to estimate the 6DoF of an object based on the relationships between the coordinates of 2D and 3D points. This method achieves high accuracy and generalizability because it is trained on a large amount of data. However, the use of large deep-learning models can impact the real-time performance of the system. In this paper, to address time constraints, the PnP algorithm has been implemented for rapid estimation of the 6DoF of the foot pose relative to the camera. Among many versions of the PnP algorithm given with different advantages and disadvantages, Iterative PnP, also called IPnP [19], using Levenberg Marquardt nonlinear minimization with complexity O(n<sup>5</sup> ) and EPnP [20] algorithm developed with high precision and O(n) linear complexity are two chosen and optimized algorithm in this paper. The PnP algorithm will achieve the best results when the 2D points found a match with the corresponding 3D points. 

### **Deep learning architecture** 

#### **MobileNetV2** 

MobileNet [21–23] is a classification model that can be applied to devices with limited computing power. The first version of MobileNet was released in 2017 [21] popularly known as an efficient and not very computationally intensive convolutional neural network for mobile vision applications. The architecture has a lower number of parameters by millions compared to other neural networks at that time. However, it still maintains good accuracy, which is based on the one called Depthwise Separable Convolution. MobileNetV2 [22] builds upon the foundation of Depthwise Separable Convolutions and introduces additional components such as Linear bottlenecks and Inverted Residual Blocks. This combination illustrates a superior performance and efficiency of MobileNetV2 compared to MobileNet. 

tional Neural Network [24]. This research aims at restoring a high-resolution image with rich details from a single lowresolution image. Pixel shuffle uses deep learning algorithms to extract important information from nearby pixels and rearrange them in a specific order to create a higher-resolution representation. This technique has proven to be successful in preserving and enhancing the image features while upscaling, leading to improved visual fidelity and increased spatial resolution. It is an operation that arranges elements in a shape _(_ ∗ _, C_ × _r_<sup>2</sup> _, H , W )_ to a shape of _(_ ∗ _, C, H_ × _r , W_ × _r )_ . Pixel shuffle has gained popularity in various computer vision tasks, including super-resolution, image generation, and style transfer. Based on the structure and dependencies among adjacent pixels, pixel shuffle can generate highquality images with enhanced details and improved visual fidelity. 

#### **Squeeze and excitation block** 

The Squeeze and Excitation (SE) block was initially introduced in 2017 by Jie Hu et al. in their work on Squeezeand-Excitation Networks (SENets) [25]. This block offers a cost-effective solution for enhancing channel interdependencies in CNNs. Its effectiveness lies in the fusion of spatial and channel information during feature extraction from images. SENets introduce a content-aware mechanism that assigns weights to individual channels, unlike a regular convolution operation that treats each channel equally. Basically, this can involve assigning a single parameter to each channel and assigning a linear scalar value to represent its relevance. After globally compressing each channel, the feature maps are reduced to a single numerical value. This yields a vector of size n, where n corresponds to the number of convolutional channels. The input is then passed through a two-layer neural network, resulting in the generation of a vector of equal dimensions. Now, these n values can serve as weights for the original feature maps, allowing us to scale each channel according to its significance. The SE block presents a simple and efficient addition to any model, reducing the minimal computational burden of the process. Its integration has the potential to enhance the performance of previously built models and enable improved results by retraining pre-built models. 

#### **Pixel shuffle layer** 

### **Virtual try-on techniques** 

Pixel shuffle is a computer vision and image processing technique used to enhance the resolution of low-resolution images. Increasing the size of an image in both width and height can cause a loss of image quality, resulting in a soft and blurry output. Fortunately, with the advancements in deep learning, there are several effective solutions available to address this issue. The pixel shuffle technique was first introduced by Shi et al. in Efficient Sub-Pixel Convolu- 

The virtual try-on community has gained a lot of interest and positive results in recent years in the development of shoe and clothing try-on products. The approach of this technology is divided into two main categories, image-based and video-based. With image-based, a vast of research has been done, for example, VITON [4] and CP-VTON [6] with systems consisting of two blocks—clothing wrapping and try-on 

123 

Page 5 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 

image synthesis, or VTNFP [13], SieveNet [26] has developed a third block—human segmentation generation for the target clothing. Regarding video-based, FW-GAN [27] synthesizes virtual clothes-on videos based on the images and poses of the target person. Besides clothes, different virtual try-on objective has also caught the attention in the last few years such as nail try-on [28]—develops a semantic segmentation of small objects that can run real-time in web mobile application, eyeglasses try-on [29]—enables display 3D glasses by tracking face and head motion of the user, or shoe try-on [8]—uses deep learning model to extract 2D foot keypoints and segment foot area to visualize occluded shoe object. Virtual try-on for fashion accessories such as glasses or shoes have not received as much attention as clothing in general. Furthermore, while the previously mentioned methods may yield favorable outcomes in certain regards, their practical implementation on mobile devices is hindered by the substantial computational expenses involved. A real-time system running on devices with limited resources is proposed in this paper. 

## **Self-built dataset** 

To have the deep-learning model with the capability to extract 2D foot keypoints, training it with a foot keypoint dataset becomes essential. As mentioned in section“Introduction”, the number of public shoe keypoint datasets is limited, a self-build shoe keypoint dataset is essential for our proposed method, which is one of our main contributions. This section will describe the process of building the self-built dataset, including data collection, data annotation, and data augmentation. 

### **Data collection** 

The dataset was created to develop foot keypoint detection models. Due to the constraint of time and devices, various iOS and Android mobile devices are used to collect data videos. To ensure diversity, videos were collected from 22 participants with different ages, genders, costumes, backgrounds, and common foot postures. The image contribution of each category of the self-built dataset is listed on Table 1. These videos were shot from both first and third-person perspectives and have an average length of six seconds, with a ratio of 7:3 for the first view and third view. The dataset currently includes barefoot and socks only, not shoes, sandals, or other accessories. The video will be divided into frames with a gap of five frames. These split frames will then be combined, filtered, and cleaned to remove any blurriness or noise caused by motion. In summary, more than 98 short videos have been collected with 6655 images extracted. The example data image is shown in Fig. 2. 

**Table 1** Image contribution of each category of the self-built dataset 

|Category||No. of images|
|---|---|---|
|Foot|Left|1478|
||Right|1621|
||Both|3556|
|Point of view|First view|4646|
||Third view|2009|
|Gender|Male|3793|
||Female|2882|
|Cloth type|Long|3783|
||Short|2882|
|Background|Indoor|5344|
||Outdoor|1311|



### **Data annotation** 

The keypoint data is structured similarly to the COCO dataset [9], comprising 18 key points representing 9 points per leg with 18 connections, as depicted in Fig. 3. These data points are defined in the context of the foot’s threedimensional space, encompassing areas like the big toe, little toe, instep, heel, and adjacent regions. The order of key points is as follows: big toe, little toe, near little toe side, near big toe side, far big toe side, far little toe side, dorsum, heel, and upper heel. The selection of these key points is guided by their ease of recognition and distinctive characteristics when projected from 3D to 2D space. 

The primary objective behind defining these key points is to establish a 6DoF system for the feet, thereby enhancing the algorithm’s accuracy in addressing the perspective-n-point problem. During the data labeling process, obscured points on the foot may arise due to varying viewing angles. Based on the degree of occlusion and the ability to estimate the location of the occluded point, each point is labeled as visible or not visible, drawing on personal experience. 

To facilitate the labeling process, a custom label tool named "3D Render Tool" (shown in Fig. 4) was developed, involving two main functions: Support for labeling and checking labeled data. For labeling purposes, firstly, the "3D Render Tool" places a 3D standard foot object manually in the correct position and direction so that its projection fit to the input image’s foot area. For convenience, an automatic mapping module is utilized to determine raw position and direction of 3D standard foot. This module firstly predict the 2D foot keypoints through a network trained on small datasets, and then use PnP algorithm to generate position and direction of 3D foot. Rotated and translated 3D foot can be refined via three its axes (top left frame in Fig. 4), so that achieve the desire fitting in the given image (shown in right frame in Fig. 4). Secondly, the defined 3D points 

123 

**65** Page 6 of 22 

Complex & Intelligent Systems (2026) 12 :65 



**Fig. 2** Foot image dataset 

123 

Page 7 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 

**Fig. 3** Foot key points dataset 



<!-- Start of picture text -->
. a ae a<br><!-- End of picture text -->

**Fig. 4** 3D render tool 

are then projected to obtain corresponding 2D foot points. Since there might be differences between the 3D standard foot and the actual human foot, an adjustment step aligns the 2D projected points to their correct positions on the image, considering the visibility of each point based on the difficulty of human determination. 

The mapping module use the 2D points after alignment to generate position and direction of 3D standard foot. The task of checking whether the labeled data meets the requirements or not is achieved through observing the fitting results on the right frame of the 3D Render Tool. 

### **Data augmentation** 

To increase the diversity of the dataset, some offline augment method was done such as brightness-contrast augment (ran- 



**Fig. 5** Left right foot remove augment 

domly changing the brightness and contrast of the image), rotation augment (randomly rotating the image in a range of −180 to 180 degree), crop augment (cropping foot area with respect to three ratios 1:1, 2:1, 4:3), left-right remove augment (cropping left or right foot area then remove them from the image, described as Fig. 5). The total number of images in the dataset will be over 33,000 after the augmentation process. 

## **Shoe virtual try-on system** 

### **System overview** 

The architecture for foot pose estimation is depicted in Fig. 6. The input of the system will be an RGB image while the output is the image with rendered 3D foot object placed in the correct position of the user’s foot. The system comprises three main components: 

1. The 2D pose estimation module, which utilizes the Single MobilePose for predicting pafmap and heatmap from an RGB image. The keypoint localization block then utilizes the deep-learning model’s output to estimate the 2D foot pose. 

2. The 3D pose estimation module, which includes a 6DoF Estimation block responsible for determining the Rotation (R) and Translation (T) vector corresponding to the camera. The resulting values are subsequently passed to the 3D Pose Stabilization block, which ensures the stability-controlled changes to R and T while processing a sequence of images. 

3. The 3D post-process module, which applies determined R,Tfrom3Dposeestimationtogeneratea3Dshoeobject through 3D Rendering block and remove the occluded ankle part before projecting it to the 2D image. 

123 

**65** Page 8 of 22 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
‘Model Shoe 30<br>Input es Output<br>omeea ee ee<br>i‘: ' 2D Pose Estimation iu' 3D Pose Estimation " 1 3D Post-process 1 : : pikes |<br>4 . Single Keypoints 6DoF 3p Pose | th 3D oestasion +!<br>as 1 | MobilePose Localization |" "| Estimation |~”| Stabilization | ,1”| Generation v LY<br>n uw u 1| Fe<br>; ee pe eee See wee Sas eee Seeees Sassi aoe eseme smal bo<br><!-- End of picture text -->

**Fig. 6** Shoe try-on system 

### **2D pose estimation** 

Figure 7 illustrates the workflow for the 2D Pose Estimation Module, one of our proposed method’s contributions. The module takes the RGB image as input, which is then passed through a deep-learning model called Single MobilePose, which extracts heatmap and pafmap information. The keypoint localization processes this information to group and localize the final foot keypoints and their connections. Finally, the output result of the module is the list of 2D foot keypoints, which are used as input for the 3D Pose Estimation Module. 

#### **Single MobilePose** 

The Single MobilePose architecture shown in Fig. 8 was built based on Part Affinity Fields (PAF) architecture [30] with two outputs, heatmap contains information of 2D foot keypoint localization and pafmap contains information of vector for each defined pair of keypoint. It utilizes MobileNetV2 as the backbone, with pre-trained weights from ImageNet [31]. This backbone choice enables efficient feature extraction, fast inference speed, and an optimal parameter count. 

In a CNN model, high-level and low-level convolutions capture different levels of abstraction and information. Lowlevel convolutions focus on extracting fundamental features like edges, corners, and textures. On the other hand, highlevelconvolutionsareresponsibleforcapturingmoreabstract and semantic information. Two output features from block5depthwise and block7-depthwise of the MobileNetV2 backbone are used to provide context and understanding to the learning process. These layers provide valuable information at different levels of abstraction. Additionally, Since the spatial resolution of low-level convolutional features is higher than that of the high-level features, a pixel shuffle block [24] is used to up-sample the block7 feature to match the resolution. After up-sampling, the block7 feature is concatenated with the low-level features. The output stride (OS) of output features compared to the input is 8. 

The output feature from the backbone MobileNetV2 will be forwarded through three PAF Stages with pafmap as output for each stage sequentially. This approach deploys an iterative refinement process to predict PAF maps with the intention of using previous knowledge to refine and get better results of pafmap. The pafmap from PAF Stage 3 will be concatenated with the extracted features from the backbone before passing through Heatmap Stage to detect heatmaps for keypoint localization. A CNN which was a sequence of convolution blocks using separable convolution layers is put in each stage. Each "SeparableConvLayer" corresponds to Separable Convolution and PReLu sequence. We used the separable convolution instead of the standard one in order to optimize the number of parameters as well as the inference time of the model. In the tail of each CNN block, a SE block is used as an attention mechanism that helps figure out the important information from each channel to further enhance the key feature extraction. From Fig. 8, there are two different CNN blocks used in PAF Stage and Heamap Stage. The difference is because of training process focuses more on PAF compared to Heatmap with three refinement stages, therefore, the CNN with more parameters is used with the intuition of balanced training results. 

In the detail workflow of the architecture, the feature map **F** , generated by the backbone, serves as input for the initial PAF Stage of the network _φ_<sup>_(_1</sup><sup>_)_</sup> . This stage predicts a set of PAFs, denoted as **L**<sup>_(_1</sup><sup>_)_</sup> and defined in Eq.(1). In the following PAF Stages, the PAFs from all previous stages are concatenated with F and refined to produce a new set of PAFs, denoted as **L**<sup>_(t)_</sup> and defined in Eq.(2). This refinement process iterates over three stages and obtains the results in the final set of PAF channels **L** = **L**<sup>_(_3</sup><sup>_)_</sup> (Eq. 3). Following this, **F** and **L** are concatenated and passed through a Heatmap Stage _ρ_ , where keypoint confidence maps **S** are predicted to estimate keypoint locations with corresponding confidence scores as Eq.(4) shows. 



123 

Page 9 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
Fig. 7 2D pose estimation Data Input 2D Foot Pose<br>module<br>Pe poe ey — a<br>1 2D Pose Estimation Module ! ae<br>oo ' wo ca<br>4<br>d Single Keypoint 1 v wy<br>{ MobilePose Localization i. “<br><!-- End of picture text -->



<!-- Start of picture text -->
vnHtI me iean MobileNetV2oe PAF Stage 1 PAF Stage 2 PAF Stage 3<br>: Depthwise| ><br>256 x 192x3 an ate28n96<br>H<br>ony Sa) Se parate [Ey  _ Savsese |<br>: ea = lel : ‘<br>' — 1 Heatmap Stage 32x24K19<br><!-- End of picture text -->

**Fig. 8** Single MobilePose architecture 





At the end of each stage, apply l2 loss function to compare the estimated predictions and the ground-truth maps ( **S**<sup>∗</sup> ) and fields ( **L**<sup>∗</sup> ) for each pixel (p) on each heat map (c) and PAF (f) channel: 





where F is the number of PAF stages and M is a score mask with M _( p)_ = 3 when that pixel is not in the area of the left or right foot bounding box, otherwise M _( p)_ = 1. The bounding box of each foot is generated based on ground truth _S_<sup>∗</sup> . 

The overall loss function is: 



_Heatmap ground-truth generation_ : ground-truth of heatmap **S**<sup>∗</sup> is generated from annotated 2D keypoints to evaluate _f S_ in training. Each heatmap represents how confident the pixel that keypoint can be located in 2D. A single peak in the heatmap should be the location of keypoint if it is visible. Let x _j_ ∈ R<sup>2</sup> be the ground-truth location of part **j** . The value at location **p** ∈ R<sup>2</sup> in heatmap channel j will be defined as: 



where _σ_ controls the spread of the peak in the heatmap. 

_Pafmap ground-truth generation_ : ground-truth of pafmap **L**<sup>∗</sup> is calculated from annotated 2D keypoints to evaluate _f L_ in training. The left foot and right foot have similar parts, like the heel, that can be difficult to distinguish from certain viewpoints. This can cause the heatmap to make mistakes when predicting the true location. PAF can address these limitations. Each PAF is a 2D vector field for connection containing both location and orientation in the supported region. Pixel in the area belongs to a specific connection preserved encoded 2D vector from one side of the connection to the other. 

123 

**65** Page 10 of 22 

Complex & Intelligent Systems (2026) 12 :65 



#### **Keypoint localization** 

The Keypoint localization block has the goal of determining the final 2D foot keypoint localization based on the extracted heatmap and pafmap of the previous block. By using the information of vector for each pair of points in pafmap, we can correctly select the correct keypoint position and eliminate noise in heatmap. 

First, non-maximum suppression is performed on predicted heatmaps to find all local maximum peaks. For each channel of heatmaps, the number of candidates may be higher than one due to a misunderstanding of the model. A large number of sets of possible connections are defined based on these candidates. In general, to find valid set of keypoints and connections related to right and left foot, find valid connections [30] and depth first search algorithm are applied with the process as Algorithm 1. 

**Algorithm 1** Find pair of foot function 

**Require:** _paf map_ **Require:** _jointList Per JointT ype_ **Ensure:** _right Foot, lef t Foot_ 

_validConnectionList_ ←{} **for** _type_ in _def inedT ypeConnection_ **do** 

_jointsSrc, jointsDst_ ← _jointList Per JointT ype(type)_ **for** _jointSrc_ in _jointsSrc_ **do** 

**for** _joint Dst_ in _jointsDst_ **do** 

Sample the line segment between two points of the connection to find _n_ interpolated points. 

_lines_ ← _LineSegFunct( jointsSrc, jointsDst)_ 

**Fig. 9** Foot PAF 

Consider a single connection shown in Fig. 9, let x _j_ 1 and x _j_ 2 be the ground-truth location of foot parts _j_ 1 and _j_ 2 from connection _c_ in the image. The value at **L**<sup>∗</sup> _c_<sup>_(_</sup><sup>**p**</sup><sup>_)_is a unit vector</sup> if point **p** lies on the pair that points from _j_ 1 to _j_ 2, otherwise the value is zero-valued. 

To evaluate _f L_ in Eq.(7) during training, the ground-truth PAF, **L** _c_ at image point **p** will be defined as: 



Here, **v** = _(_ x _j_ 2 − x _j_ 1 _)/_ ∥x _j_ 2 − x _j_ 1 ∥2 is the unit vector in the direction of the connection. The set of points on the pair is defined as those within a distance threshold of the line segment. Those points **p** satisfy the condition as: 



where the connection width _σl_ is a distance in pixels, and the connection length is _lc_ = ∥x _j_ 2 − x _j_ 1 ∥2 and **v** ⊥ is a vector perpendicular to **v** . 

_count_ ← 0 _v_ ← ∥ _joint Dstjoint Dst_ −− _jointSrcjointSrc_ ∥2 **for** _line_ in _lines_ **do** 

Check if the vector of the line from pafmap has the same direction as of the connection 

_v_ 1 ← _paf map(line)_ **if** _v_ 1 × _v > paf T hresh_ **then** _count_ ← _count_ + 1 **end if end for if** _count > thesh_ **then** _connection_ ← _( jointSrc, joint Dst)_ validConnectionList.insert(connection) 

**end if end for end for end for** 

_right Foot, lef t Foot_ ← _Foot DFSFunct(validConnectionList)_ ▷ Consider joint and connection as a graph structure, apply DFS to associate connection to the same foot together 

Since all the key points are joined into pairs as Fig. 10 shows, pairs that share the same part detection candidates are assembled into the correct foot. An empty list is created to store the key points for each foot. Go over each pair, and check if part A of the pair is already present in any of the lists. If it is present, then it means that the key point belongs 

123 

Page 11 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
HO H1 FO FL<br>je<br>=<br><!-- End of picture text -->

**Fig. 10** Pair between key points. Two key points with the same pair will be grouped 

to this list and part B of this pair should also belong to this foot. Thus, add part B of this pair to the list where part A was found. If part A is not present in any of the lists, then it means that the pair belongs to a new foot, not in the list and thus, a new list is created. In the end, lists of group pairs of foot parts for each foot right and left is created. In case there is more than one group of pairs for each foot, pick the highest total score of confidence in their association. The confidence in their association is measured as: 



where c is the connection, N is the number of interpolates points of the foot parts _d j_ 1 and _d j_ 2 while **p** _i_ ∈ R<sup>2</sup> is the corresponding position. 

### **3D pose estimation module** 

The 3D Pose Estimation Module contributes to the system by determining the 6DoF foot pose in real-world space. The module consists of two main blocks: 6DoF Estimation and 3D Pose Stabilization. The input of the 3D Pose Estimation Module is the list of foot keypoints extracted from the aforementioned 2D Pose Estimation Module. First, it is passed through the 6DoF Estimation block with the purpose of 6DoF foot pose determination. This 6DoF information is forwarded to the 3D Pose Stabilization process to stabilize and update 

its value. Finally, the module produces the stable 6DoF foot pose as its output. 

#### **6DoF estimation** 

6DoF refers to the ability of an object or system to move freely in three-dimensional space. It contains three translational degrees of freedom and three rotational degrees of freedom. Translational degrees are often described as translation vector T ∈ R<sup>3×1</sup> including values moving along the x-axis (forward/backward), y-axis (left/right), and z-axis (up/down). Rotational degrees illustrate rotation around the x-axis (roll), y-axis (pitch), and z-axis (yaw), defined as rotation vector R ∈ R<sup>3×1</sup> . Through R, T, the position and orientation of each foot in real-world space can be captured. 

Inthisblock,IPnP[19]isusedtogivekeypoints’locations corresponding to the left and right foot from the 2D pose estimation module, and corresponding 3D points from 3D model standard foot to find the R, T vector of each foot. Here, the R and T vectors obtained from EPnP algorithm [20] are used as initial values of the IPnP algorithm. This approach is beneficial because it improves the speed and accuracy of the iterative convergence process of IPnP. It also has minimal impact on real-time performance because of the low time complexity of EPnP. 

123 

**65** Page 12 of 22 

Complex & Intelligent Systems (2026) 12 :65 

#### **3D pose stabilization** 

The importance of the 3D Pose Stabilization block becomes apparent when working with a sequence of images in reallife applications, particularly when directly streaming from a camera. This eliminates jitter from the continuous system output,resultinginasmootherandmoreconsistentuserexperience. Jitter is the inconsistency in the time interval between consecutive images in a real-time system, which negatively impacts the performance and the experience of the user. To prevent jitter, the Alpha-Beta filter is used to stabilize and smooth values in R, T based on consecutive input, and output of the system. 

this issue, a motion detection block is employed, relying on the Intersection over Union (IOU) value between consecutive foot area frames in the image sequence. The IOU is calculated through the foot bounding box of the image sequence. In particular, the foot bounding box in the image sequence is compared with an image that has an interval of 10 images to the past. A movement threshold value of 0.96 is established, signifying that the foot is considered stable if the IOU score surpasses this threshold. When the foot is deemed stable, the parameters of both rotation (R) and translation (T) are no longer updated, ensuring stability in the algorithm. 

### **3D pose-process module** 

R, T are defined as: 





The total state variables required to update is 



Here, assume that the velocity of change with respect to different values in R and T is constant, the state of values and velocity of R and T is updated following the State Update Equation for position and velocity respectively. The _α_ − _β_ track update equations are defined as: 

Details of the 3D Pose-process Module are shown in Fig. 11. 3D shoe models, having color textures and manually preprocessed to fit the standard foot object shape, are transformed by applying R, T in the previous module to get the 3D shoe and foot object with the position and direction corresponding to the foot in the image. Then a virtual camera is set up at coordinate origin with fixed intrinsic parameters that used in PnP algorithm. The occlusion utilizes the Ray-Casting algorithm for the virtual camera and 3D shoe model to find the shoe area obscured by the 3D foot model. Finally, the resulting image is obtained by rendering the visible shoe area and then merging it with the original image. 

The Stage Update Equation for position: 



## **Experiment** 

### **Experimental setting** 

The State Update Equation for velocity: 



In (15) and (16), _xn_ , _x_ ˙ _n_ , _zn_ present state, speed change of that state, and system result respectively at iteration n. Additionally, the need for _α_ and _β_ values are to control the error in system measurement. When the system measurement range is not as expected, probably due to two reasons: The imprecision in measurement or the change in velocity of the user’s foot. The factor _β_ in (16) is determined by the radar precision level. If the precision of radar is high the gap between the predicted and measured range may be the result of velocity change. In this case, the _β_ value should be set high. In reverse, the low _β_ is set. The factor _α_ in (15) depends on the radar measurement precision to control the change in the measured range. For high-precision radar, high _α_ should be chosen, giving high weight to the measurements. 

Although the Alpha-beta algorithm effectively reduces most of the foot movement jitter, it still struggles to eliminate noise when the foot remains entirely stationary. To address 

**Dataset** : Due to the limitation of public foot keypoint datasets,onlyourself-builtdatasetdescribedinsection“Selfbuilt dataset” is selected for experiments and evaluation. The data is divided into three sets, 33,274 argumented training images from 5799 original images, 577 evaluation images, and 279 test images. The images in each set do not have overlap characteristics to ensure data independence. In addition, the data in the test and evaluation sets are not augmented to ensure the training goal. All images are padded zero-value and resized to 256x192 resolution. 

**Training** : In order to train the model, AdamW opimization [32] is used with weight decay 1 × 10<sup>−4</sup> , momentum variables _B_ 1 = 0 _._ 9 _B_ 2 = 0 _._ 999. AdamW is an extended version of Adam optimization, which improved the generalization and convergence properties compared to Adam. The learning rate is initialized as 1 × 10<sup>−4</sup> , with an Exponential Decay Learning rate scheduler. The involved experiment was performed on platforms including an NVIDIA 1080 GPU (8GB), an Intel Core i5-8500 CPU (3.00GHz x 6). The network was trained with loss L2 as depicted in sec- 

123 

Page 13 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
Po Tnputs 1| r 73D Post-process I~ ~ “Outputs<br>] 1 Hy eam<br>| om 1)!| | CNeS ||<br>—= JA | | eee ~_<br>| SP i} Hit =<br>| it a, | I} ole<br>] | i! !<br>ee ee te<br><!-- End of picture text -->

**Fig. 11** 3D pose-process module 

tion“Single MobilePose” and the entire training takes place in 150 epochs. 

**Baselines** : To evaluate the performance of the 2D Pose Estimation Module, a comparison was made with the OpenPose [30] approach, a well-known real-time multi-person keypoint detection model employing the PAF network architecture. OpenPose [30] is designed not only for accurate detection and tracking of human body keypoints in images and videos but also for achieving real-time performance on standard hardware. Additionally, to propose a novel architecture of Single MobilePose, the number of PAF stages in the pipeline is reduced without retraining the model. This decision was based on observing minimal differences in the result of the refinement stage in PAF after a hard training process, indicating that reducing the tail PAF stage would not significantly affect the system’s output. Regarding the evaluation of 3D Foot Pose Estimation, as no related open-source project is available, the assessment relies on qualitative analysis. Figures 14 and 15 depicting the impact of 3D Pose Stabilization on the final result are provided to support the evaluation. 

_Evaluation metrics_ : For the 2D Pose Estimation problem, mean average precision (mAP) is used with an Object Keypoint Similarity (OKS) [9] threshold in the range from 0.5 to 0.95. OKS is defined as: 



- _s_ is the area of the bounding box divides the total image area. 

- _ki_ per-keypoint constant that controls falloff. 

Here, to adapt Eq.(17) to the custom dataset, the value of the sigma factor is adjusted. This sigma factor corresponds to the spread in the Gaussian area of each point in the heatmap, which controls the strict evaluation. With the modified sigma value, the strictness of keypoint matching scores is under control. A higher sigma factor leads to a more soft evaluation, making the matching of keypoints less sensitive to minor differences. Conversely, lower sigma values enforce stricter matching criteria, demanding a closer alignment between predicted and ground truth keypoints. In our self-built dataset, the sigma factor is fine-tuned to suit the specific characteristics of keypoints. For instance, well-defined and easily identifiable foot keypoints like the big toe and little toe were assigned lower sigma values, resulting in precise matching requirements. Conversely, keypoints that were less distinguishable or challenging to identify, such as the heel and dorsum, were assigned higher sigma values, allowing for more flexibility in the matching process. The custom sigma factor for each is defined with the aforementioned order 3.2 aforementioned as: 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, and 0.1. 

### **Quantitative results** 

where 

- _di_ is Euclidean distance between ground-truth keypoint and predicted keypoints. 

This section compares four networks: Openpose [30], proposed Single MobilePose, Single MobilePose without PAF Stage 2 (Single MobilePose No2), and Single MobilePose without PAF Stage 2 and 3 (Single MobilePose No3). First, 

123 

**65** Page 14 of 22 

Complex & Intelligent Systems (2026) 12 :65 

**Table 2** Comparison of computation cost and speed 

|Model|FLOPS (G|)<br>Params (M)|GPU time (ms)|CPU time (ms)|TF lite (1/2/3) (ms)|
|---|---|---|---|---|---|
|OpenPose [30]|5.840|36_._79|16.01|143_._47|800.12 / 518.13 / 350.87|
|Single MobilePose|1.426|5_._53|17.16|34_._49|30.85 / 17.58 / 13.72|
|Single MobilePose No2|1.143|4_._39|13.71|30_._03|24.61 / 14.27 / 10.84|
|Single MobilePose No3|0.796|2_._80|10.07|21_._41|18.05 / 10.37 / 7.88|
|**Fig. 12** Desktops and mobile<br>devices FPS comparison||lm SingleMobilePose<br><br><br><br><br><br><br>|||63.64 026<br>|



the comparison explains the computation cost and speed of the model, which includes factors such as the number of calculations, model parameters, and inference time. Furthermore, the comparison of accuracy between different networks is also conducted. 

#### **Computation cost and speed evaluation** 

_Number of parameters and amount of calculation_ : Four models are compared in terms of calculation and parameters as showninTable 2.AllversionsoftheproposedSingleMobilePose have much fewer FLOPS (1.426G, 1.143G, 0.796G) and parameters (5.53M, 4.39M, 2.80M) compared to OpenPose [30] with FLOPS 5.840G and parameters 36.79M. 

_Inference time on GPU and CPU_ : The inference time of the models on different devices is recorded in Table 2. In general, all proposed models work efficiently on both GPU and CPU devices with low inference time. The highest inference time is 17.16 ms on GPU and 34.49 on CPU. 

_Inference time on CPU with TensorFlow Lite_ : After converting all models to the TensorFlow Lite version, the inference time with different numbers of threads are compared as a parameter, which is 1, 2, and 3 consequently. The number of threads depends on the CPU kernel supporting multithreading running. The result in Table 2 has shown the 

efficiency when converting models from TensorFlow format to TFLite format to improve execution speed, especially with the number of threads using as 3, the speed is greatly improved. 

_Performance on mobile devices with TensorFlow Lite_ : The inference time of converted TensorFlow Lite models on desktops and mobile devices are measured and compared as illustrated on Fig. 12. Whereas small devices like Iphone 11 and Iphone 11 Pro achieve around 30 frames per second (FPS), which is about half of desktops with GPU FPS, Ipad Pro 11 M1 reaches around 60 FPS on average and outperforms desktops using CPU only. 

#### **Keypoint localization evaluation** 

Tables 3 and 4 describe the performance comparison among three versions of proposed Single MobilePose and OpenPose [30] on the validation and test dataset, while Fig. 13 illustrates the PCK accuracy with the difference pixels of predicted keypoints compared to the ground truth as the threshold.ResultsshowthatOpenPose[30]withmuchhigher parameters gets slightly better results than the proposed models. In addition, among the three proposed Single MobilePose models, the second model achieves the same results as the first model while having fewer parameters and faster processing speed. The reason is that during training of the original 

123 

Page 15 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 

**Table 3** Comparison of foot keypoints localization on validation dataset 

|Model|mAP|AP50|AP75|AP90|
|---|---|---|---|---|
|OpenPose [30]|0.725|0.894|0.754|0.599|
|Single MobilePose|0.704|0.884|0.750|0.501|
|Single MobilePose No2|0.700|0.890|0.751|0.484|
|Single MobilePose No3|0.683|0.890|0.739|0.468|



**Table 4** Comparison of foot keypoints localization on test dataset 

|Model|mAP|AP50|AP75|AP90|
|---|---|---|---|---|
|OpenPose [30]|0.737|0.869|0.868|0.556|
|Single MobilePose|0.708|0.866|0.780|0.603|
|Single MobilePose No2|0.710|0.866|0.781|0.587|
|Single MobilePose No3|0.703|0.866|0.781|0.502|



with horizontal axes refer to numbers of frames and vertical axes indicate values of Rx, Ry, Rz, Tx, Tz, and Tz. The algorithm required some first loops to converge, however, once stabilized, the algorithm makes the values of R and T vectors much more stable compared to when there is no stabilization module. This approach has one drawback, which is its inability to completely handle vibration noise in the case of a completely stationary foot, adversely affecting the user experience. 

By implementing the innovative IOU approach, we can effectively maintain stability in the module. This is achieved by identifying the specific moment when we desire the value of R and T to remain unchanged. In Fig. 15, which shares the same legends with Fig. 14, the previous list of noise values is substituted with a seamless straight line, representing a complete stationary corresponding value. 

#### **User study results** 

Single MobilePose model, the output loss at PAF Stage 2 and PAF Stage 3 has a similar value and there is no significant improvement of the pafmap at this loop. Therefore, cutting the second Stage block will not have a big effect on the overall result of the model. 

#### **3D stabilization result** 

Figure 14 describes the effect of the alpha–beta filter algorithm on a sample video with values of two vectors R, T, 

A user study is conducted for quantitative comparison of our mobile-implemented method with two popular AR mobile applications: Adidas and Amazon by the selected volunteer after experiencing three applications respectively. Volunteers are recruited through an open call and screened to ensure they fall within the age range of 18 to 50 and represent a variety of occupations, thereby supporting the reliability and generalizability of the evaluation. This study emphasizes the contribution of the proposed method in terms of the fit and 

**Fig. 13** Comparison of foot keypoints localization with PCK score on test dataset 



<!-- Start of picture text -->
g<br>2 e-SingleMobilePose_ 3PAF_INEAT<br>=<br>—*-SingleMobilePose_IPAF_1HEAT.<br>=*-SingleMobilePose 2PAF_IHEAT<br>ons —*OpenPoseVGG<br>Threshold (pixels)<br><!-- End of picture text -->

123 

**65** Page 16 of 22 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
Rx Ry |— rzorigin<br>0,095 0.05 7 —— R1_Kalman<br>0.090<br>0.00<br>0.085<br>0.080 ~0.05<br>0.075 -0.10<br>0.070<br>-0.15<br>0.065 |—— RO_Origin<br>0.060 RO_Kalman 0.20<br>0 50 100 150 200 250 300 350 oO 50 100 150 200 250 300 350<br>Frame number Frame number<br>Rz TX |— to origin<br>— 0 Kalman<br>0.44<br>0.2<br>0.42<br>0.40 0.0<br>0.38<br>0.2<br>0.36<br>034 — R2_Origin -0.4<br>— R2_Kalman<br>0 50 100 150 200 250 300 350 o 50 100 150 200 250 300 350<br>Frame number Frame number<br>TyTz2<br>2<br>L<br>1<br>0 —— TLoriginTLkalman 9 —— 72_0rigin12_kalman<br>=<br>-1<br>-2<br>-3 aa3<br>oO 50 100 150 200 250 300 350 oO 50 100 150 200 250 300 350<br>Frame number Frame number<br><!-- End of picture text -->

**Fig. 14** Compare origin and used Kalman vectors 

smoothness of the shoe try-on process. Volunteers are asked to remain their foot poses and the camera position through the whole experiment. Then, a set of survey questions are given to 50 volunteers, 20 of them are working in fashion industry, the rest are selected randomly. The survey questions ask the users to rank the fit of AR shoe models and the smoothness in real-time. The survey results are then scaled into scores of zero to five where zero is the worst and five is the best. The final score for each application is computed by taking the average of all question scores provided by each partici- 

pant. Our proposed method, Amazon, and Adidas achieves the average score of 2.88, 2.38, and 2.27, respectively. Our proposed method outperforms two other applications in this user study. The normalized result histogram chart is illustrated in Fig. 16. 

### **Qualitative results** 

Several examples of our proposed method’s results are in Fig. 17. These results effectively demonstrate the robust 

123 

Page 17 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
Rx |— ro-origin Ry |—ariorigin<br>0.095 7—— Ro_Kalmaniou 0.05 7 —— R1_Kalmaniou<br>0.090<br>0.00<br>0.085<br>0.080 0.05<br>0.075 0,10<br>0.070<br>-0.15<br>0.065<br>0,060 -0.20<br>0 50 100 150 200 250 300 350 oO 50 100 150 200 250 300 350<br>Frame number Frame number<br>Rz |— 2 origin TX |— 1o_origin<br>— R2 Kalmaniou — 10_Kalmaniou<br>0.44 -<br>0.42<br>0.0<br>0.40<br>0.2<br>0.38<br>-0.4<br>0.36<br>0 50 100 150 200 250 300 350 oO 50 100 150 200 250 300 350<br>Frame number Frame number<br>TyTz2<br>2<br>,, |<br>0 —— TLT1kalmaniouOrigin ° —— 72T2_Originkalmaniou<br>-1<br>=i,<br>2<br>-2<br>-3<br>0 50 100 150 200 250 300 350 0 50 100 150 200 250 300 350<br>Frame number Frame number<br><!-- End of picture text -->

**Fig. 15** Compare origin and used Kalman IOU vectors 

capability of the suggested approach to adapt to various foot shapes and diverse scenarios. These scenarios include two viewing angles (first and third view), indoor and outdoor settings, and different clothing styles like short or long pants. Our proposed method is able to handle various types of shoes such as sneakers, high heels, and sandals. The results are visually appealing and realistic, with the shoes fitting the foot 

accurately and naturally. However, to compare with other two applications, only sneakers are used for the comparison. The proposed method outperforms the other two applications in terms of the fit and smoothness of the shoe try-on process, as shown in Fig. 18. 

123 

**65** Page 18 of 22 

Complex & Intelligent Systems (2026) 12 :65 

**Fig. 16** User study qualitative comparison 



<!-- Start of picture text -->
12<br>—— Ours<br>1 Amazon<br>—— Adidas<br>8<br>36<br>€<br>5<br>2<br>4<br>2<br>0<br>0 1 2 3 4 5<br>Score<br><!-- End of picture text -->

## **Discussions and future work** 

### **Discussions** 

In conclusion, this research presents an innovative approach in shoe AR try-on by introducing a self-built foot dataset; a proposed method emulating realistic occlusion in 3D shoe models from real-time 2D keypoints detection, smoothed by a stabilize mechanism and implemented in a mobile shoe AR try-on application. Generally, the proposed application is satisfactory with real-time performance on mobile devices, although there are still lag issues when used with older devices. The application gives quite quality results with the user’s sense of trying on shoes regardless of different scenario set-up like light condition, foot background, position of cameras (third view or first view) or clothes of users. 

### **Limitations** 

However, because the data is not diverse, under several circumstances which does not occur regularly in the self-built dataset, the shoe detection module is not able to identify specific keypoints. The prosthetic foot solution has not solved the problem completely when the human foot has many different shapes and sizes, leading to many situations where the results are unnatural. Regarding the tracking module, the system has now processed for smoothing results with stationary or moving feet. However, moving the foot at high speed can cause the model to fall off the screen, causing discomfort. A 

limitation of using 3D standard foot model in occlusion is illustrated on Fig. 19. Some region on the collar part of the 3D Shoe model is shown despite the fact that they should be covered by the human foot ankle from this point of view. 

### **Future work** 

Dataset enrichment is required to increase diversity and generality when training to improve the keypoint localization model. To address this, the dataset will be expanded through multiple strategies, including geometric and photometric augmentation, generation of additional foot poses using controlled motion capture or procedural pose synthesis, and rendering of synthetic foot–shoe pairs from various viewpoints using 3D simulation tools. These approaches help increase variability in foot shape, orientation, lighting, and occlusion conditions. Once our dataset achieves a considerable scale, it will soon be published. In addition, we will continue to refer to and improve the performance of the existing model to make the proposed method can be effortlessly implemented on limited computing power devices. Standard benchmarks and metrics are built to evaluate the quality of datasets and models. Further research on the joint solution with prosthetic models is made to optimize and accurately determine the position of the shadow of the ankle. Finally, the Stabilization module needs to be improved to stabilize and smooth the 3D model during rendering. 

123 

Page 19 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
etm 7 FN Oe a ae<br>= BsSRS<br>By je, |e ae<br><!-- End of picture text -->

**Fig. 17** Shoe try-on final results 

123 

**65** Page 20 of 22 

Complex & Intelligent Systems (2026) 12 :65 



<!-- Start of picture text -->
Original Image Adidas Amazon Ours<br>eT Wid ne Wg A<br>é =]<br>= ,<br>YS DA Sa<br><!-- End of picture text -->

**Fig. 18** Shoe Try-on compared to other applications 

123 

Page 21 of 22 **65** 

Complex & Intelligent Systems (2026) 12 :65 



**Fig. 19** An example of a failure case of the proposed method 

**Acknowledgements** This research was fully funded by Viettel High Technology Industries Corporation. The authors would like to thank all members of the 3DR team for their contribution. 

### **Declarations** 

**Conflict of interest** The authors have no conflict of interest to declare. 

**Open Access** This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/. 

## **References** 

1. Ge Y, Song Y, Zhang R, Ge C, Liu W, Luo P (2021) Parser-free virtual try-on via distilling appearance flows. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 8485–8493 

2. Ge C, Song Y, Ge Y, Yang H, Liu W, Luo P (2021) Disentangled cycle consistency for highly-realistic virtual try-on. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 16928–16937 

3. Han X, Hu X, Huang W, Scott MR (2019) Clothflow: a flowbased model for clothed person generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10471–10480 

4. Han X, Wu Z, Wu Z, Yu R, Davis LS (2018) Viton: an image-based virtual try-on network. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7543–7552 

5. Hsieh C-W, Chen C-Y, Chou C-L, Shuai H-H, Liu J, Cheng W- H (2019) Fashionon: semantic-guided image-based virtual try-on with detailed human and clothing information. In: Proceedings of the 27th ACM International Conference on Multimedia, pp 275– 283 

6. Wang B, Zheng H, Liang X, Chen Y, Lin L, Yang M (2018) Toward characteristic-preserving image-based virtual try-on network. In: Proceedings of the European Conference on Computer Vision (ECCV), pp 589–604 

7. Chou C-T, Lee C-H, Zhang K, Lee H-C, Hsu WH (2019) Pivtons: Pose invariant virtual try-on shoe with conditional image completion. In: Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part VI 14, Springer, pp 654–668 

8. An S, Che G, Guo J, Zhu H, Ye J, Zhou F, Zhu Z, Wei D, Liu A, Zhang W (2021) Arshoe: real-time augmented reality shoe try-on system on smartphones. In: Proceedings of the 29th ACM International Conference on Multimedia, pp 1111–1119 

9. Lin T-Y, Maire M, Belongie S, Hays J, Perona P, Ramanan D, Dollár P, Zitnick CL (2014) Microsoft coco: common objects in context. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, Springer, pp 740–755 

10. Tran D, Forsyth D (2010) Improved human parsing with a full relational model. In: Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV 11, Springer, pp 227– 240 

11. Hidalgo G, Raaj Y, Idrees H, Xiang D, Joo H, Simon T, Sheikh Y (2019) Single-network whole-body pose estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 6982–6991 

12. Andriluka M, Pishchulin L, Gehler P, Schiele B (2014) 2d human pose estimation: new benchmark and state of the art analysis. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 3686–3693 

13. Yu R, Wang X, Xie X (2019) Vtnfp: an image-based virtual try-on network with body and clothing feature preservation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10511–10520 

14. Xu Y, Zhang J, Zhang Q, Tao D (2022) Vitpose: simple vision transformer baselines for human pose estimation. arXiv preprint arXiv:2204.12484 

15. Sun K, Xiao B, Liu D, Wang J (2019) Deep high-resolution representation learning for human pose estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5693–5703 

16. Rad M, Lepetit V (2017) Bb8: a scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth. In: Proceedings of the IEEE International Conference on Computer Vision, pp 3828–3836 

17. Tekin B, Sinha SN, Fua P (2018) Real-time seamless single shot 6d object pose prediction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 292–301 

18. Zakharov S, Shugurov I, Ilic S (2019) Dpod: 6d pose object detector and refiner. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1941–1950 

19. Madsen K, Nielsen HB, Tingleff O (2004) Methods for non-linear least squares problems 

20. Lepetit V, Moreno-Noguer F, Fua P (2009) Ep n p: an accurate o (n) solution to the p n p problem. Int J Comput Vision 81:155–166 

21. Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H (2017) Mobilenets: efficient convolutional 

123 

**65** Page 22 of 22 

Complex & Intelligent Systems (2026) 12 :65 

neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 

22. Sandler M, Howard A, Zhu M, Zhmoginov A, Chen L-C (2018) Mobilenetv2: inverted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 4510–4520 

23. Howard A, Sandler M, Chu G, Chen L-C, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V et al (2019) Searching for mobilenetv3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1314–1324 

24. Shi W, Caballero J, Huszár F, Totz J, Aitken AP, Bishop R, Rueckert D, Wang Z (2016) Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 1874–1883 

25. Hu J, Shen L, Sun G (2018) Squeeze-and-excitation networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7132–7141 

26. Jandial S, Chopra A, Ayush K, Hemani M, Krishnamurthy B, Halwai A (2020) Sievenet: a unified framework for robust image-based virtual try-on.In: Proceedingsofthe IEEE/CVFWinterConference on Applications of Computer Vision, pp 2182–2190 

27. Dong H, Liang X, Shen X, Wu B, Chen B-C, Yin J (2019) Fw-gan: flow-navigated warping gan for video virtual try-on. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1161–1170 

28. Duke B, Ahmed A, Phung E, Kezele I, Aarabi P (2019) Nail polish try-on: realtime semantic segmentation of small objects for native and browser smartphone ar applications. arXiv preprint arXiv:1906.02222 

29. Kobayashi T, Sugiura Y, Saito H, Uema Y (2019) Automatic eyeglasses replacement for a 3d virtual try-on system. In: Proceedings of the 10th Augmented Human International Conference 2019, pp 1–4 

30. Cao Z, Simon T, Wei S-E, Sheikh Y (2017) Realtime multi-person 2d pose estimation using part affinity fields. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 7291–7299 

31. Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L (2009) Imagenet: a large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp 248–255. Ieee 

32. Loshchilov I, Hutter F (2017) Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 

**Publisher’s Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

123 

