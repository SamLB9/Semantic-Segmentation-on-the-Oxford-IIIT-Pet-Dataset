# Overview
Using a standardized Oxford-IIIT Pet Dataset segmentation task, this study provides a thorough comparative examination of four segmentation architectures, UNet, Autoencoder, CLIP-based, and Prompt-based. Providing insights into architectural trade-offs, robustness, and practical application. The UNet model performs poorly in crowded environments and under class imbalance, although showing good generalization and border delineation considering its architectural simplicity. Despite its theoretical potential, the autoencoder’s two-step decoding method and insufficient semantic granularity cause it to continually lag behind and display unpredictable training dynamics. The CLIP-based model, which makes use of frozen multimodal embeddings, performs best on all important metrics and is highly resilient to structural and photometric perturbations. Building on this, our prompt-guided segmentation model achieves high pixel accuracy with added user input despite lower spatial precision, extending utility to interactive contexts. Our findings highlight the advantages of pretraining, multimodal fusion, and architectural maturity in computer vision pipelines and confirm that deliberate simplification may yield workable lightweight alternatives for practical implementation.

## Introduction
Image segmentation plays a crucial role in many computer vision applications, including medical imaging, scene understanding, and object recognition. Image segmentation has been revolutionized by the advent of transformer-based architectures and large-scale pretrained vision-language models, which allow rich semantic understanding and robust generalization from limited supervision. Since the creation and proof of performance of the UNet in the ISBI Cell Tracking Challenge, image segmentation has significantly improved Ronneberger et al. (2015). The architecture of these networks has evolved to achieve human or even superhuman performance on tasks such as cancer detection and object recognition in self-driving vehicles. In this project, we will explore the evolution of model architectures on the popular PET III dataset to discover how changes in architecture affect performance in image segmentation tasks. We will systematically train and evaluate four segmentation models on the PET III dataset, UNet, Autoencoder, CLIP-based, and Prompt-based architectures, in order to provide a fair comparison and to produce a report on the comparative qualitative and quantitative performance of these models. In the following essay, we will evaluate the examined models via these steps, which are detailed in the order of the Table of Contents.

## Dataset Preprocessing and Augmentation
### Dataset Overview
----
The Oxford-IIIT Pet Dataset Parkhi et al. (2012) consists of 7,392 RGB images and their corresponding pixel-wise ground-truth segmentation masks, each belonging to one of 37 pet breeds (12 cats and 25 dogs). ”The dataset contains about 200 images for each breed (which have been split randomly into 50 for training, 50 for validation, and 100 for testing)” Parkhi et al. (2012). We will expand the TrainVal set with augmented data discussed later. These images exhibit considerable variability in scale, lighting conditions, pose, and background clutter, thereby making the dataset particularly suitable for evaluating the robustness of semantic segmentation algorithms under real-world visual variability. Each image is annotated at the pixel level with semantic labels representing three core classes: background (0), cat (1), and dog (2). For the test set, a fourth label corresponding to object boundaries (3) is included and encoded in RGB, requiring a dedicated parsing routine for compatibility with conventional segmentation pipelines. The dataset is partitioned into three non-overlapping subsets: training, validation, and test. Each subset is balanced to contain approximately 200 instances per breed following the evaluation protocol proposed in Parkhi et al. (2012). This ensures that the training set captures intra-class variation, while the validation set supports hyperparameter optimization and the test set provides an unbiased measure of generalization.

### Preprocessing Pipeline
----
#### Resizing and Padding
--
To address heterogeneity in raw image dimensions and ensure compatibility with neural network architectures requiring fixed input sizes, we employed a structured preprocessing pipeline involving resizing followed by symmetric padding. This pipeline was applied to the TrainVal set and the Test set. Let (H,W) denote the height and width of an input image. The resizing step rescales the image isotropically such that the larger dimension is mapped to D= 128, preserving the aspect ratio via the bilinear interpolation function: <img width="1214" alt="Screenshot 2025-04-15 at 11 50 45" src="https://github.com/user-attachments/assets/51f2099a-8232-4044-b027-7c561aa77bfc" />
To maintain spatial centrality and prevent bias, we padded the resized image to (D,D) using Mean Padding. Mean Padding using the per-channel mean pixel value of the image to minimize boundary artifacts.

#### Label Conversion and Encoding
--
Train/Val masks were converted where pixel values in the open interval (0,255) were mapped to either 1 (cat) or 2 (dog) based on filename casing logic implemented in is uppercase(). Boundary pixels (value 255) were merged with background (0). For test masks, an RGB-to-integer conversion was performed using convert label test(), where boundary pixels were explicitly preserved.

#### Normalization and File Storage
--
All image tensors were normalized to [0,1] and saved in lossless PNG format. For compatibility with ImageNet-pretrained backbones, normalization to zero mean and unit variance using ImageNet statistics was later applied at the model level. Class label tensors retained discrete integer values.

#### Normalization and File Storage
--
The end-to-end preprocessing steps can be summarized algorithmically as:
1. Read and decode raw image and label.
2. Resize image to max(H′,W′) = D using bilinear interpolation.
3. Resize label using nearest-neighbor interpolation.
4. Apply symmetric padding to reach target size (D,D).
5. Convert grayscale or RGB label to 3-class (TrainVal) or 4-class (Test) format.
6. Normalize image and store both image and label in PNG format.

### Data Augmentation Strategy
----
To mitigate overfitting and enhance model generalization, we implemented a diverse augmentation pipeline that synchronously transforms image and label pairs to preserve spatial alignment for the TrainVal set after resizing.

#### Geometric Transformations
--
- Horizontal Flip (p = 0.5): Applied via F.hflip() on both tensors to simulate mirrored scenarios.
- Random Rotation (p = 0.25): Angle θ ∼U(−45◦,45◦) using bilinear (image) and nearest-neighbor (mask) interpolation.
- Translation (p = 0.05): Translation up to 5% of image size along each axis, maintaining label alignment through affine warping.
- Random Cropping (p = 0.01): Parameters sampled from (s,r) ∈[0.9,1.0] × [1.0,1.0] to ensure minor cropping without skew.
- Scaling (p= 0.05): Resize factor λ∼U(0.8,1.2) sampled independently for each sample.

#### Elastic Deformation
--
Elastic warping introduces non-linear distortions using a Gaussian-smoothed random displacement field. Following Simard et al. (2003), we generate displacement fields (δx,δy) from N(0,σ2), smoothed with a Gaussian kernel Gσ, and scaled by α:<img width="1215" alt="Screenshot 2025-04-15 at 11 55 35" src="https://github.com/user-attachments/assets/b319bae6-dd7b-46d2-a3d6-6e4c36efa57f" />

#### Photometric Augmentations
--
- Color Jitter (p= 1.0): Random perturbations to brightness, contrast, saturation, and hue within ±20% ranges using torchvision’s ColorJitter.
- Gaussian Blur (p= 0.3): Applied with kernel radius r ∼U(0.5,1.5) to simulate sensor blur and motion artifacts.

#### Composite Augmentation Routine
--
Transformations were applied in a randomized order per image-label pair using the augment pair() routine. The algorithmic structure is:
1. Shuffle augmentation list.
2. For each (augmentation,p), apply with probability p.
3. Perform center crop to enforce final dimensions.
This probabilistic augmentation pipeline increased data diversity across epochs without inflating storage costs.

#### Augmentation Implementation & Impact
--
The augmentation pipeline was implemented to produce 10 additional Train/Val image mask pairs in addition to the original image mask pair. Test set remains 3,711 pairs, Train/Val set expanded from 3,681 to 40,480 pairs. The dataset was then split into class-stratified training (70%), validation (15%), and testing (15%), this split and augmented set was used to train and evaluate all the following models.

In our initial testing we observed immediate and substantial performance gains with added data diversity due to the addition of augmented data. Geometric distortions (rotation, translation) enhance the model’s invariance to camera pose, while elastic warping helped regularize shape deformations. Color jitter and blur increase tolerance to illumination variance and noise. The dynamic sampling of augmentations ensured continual exposure to novel samples, promoting better generalization. Performance gains were consistent with literature Shorten and Khoshgoftaar (2019).

## Network Architectures
### UNet-Based Model
----
An improved deep UNet architecture (Ronneberger et al. (2015)) designed specifically for patch-based segmentation of 128×128 RGB images has been implemented for the segmentation task. The model employs four iterative rounds of upsampling and downsampling in a robust encoder-decoder architecture with skip connections.

The encoder pathway starts with a DoubleConv block that processes the input image with 64 feature channels. Four downsampling stages then gradually expand and decrease respectively the feature space to 1024 channels at the bottleneck and spatial dimensions. Dropout is used to reduce overfitting in each DoubleConv block, which performs two consecutive 3×3 convolutions with batch normalization and LeakyReLU activations. This design guarantees that the network can efficiently extract contextual information at multiple scales from the input photos.

The network integrates skip connections from matching encoder layers and performs upsampling operations in the decoder pathway using bilinear interpolation. By establishing information highways that promote gradient flow during training, these connections maintain important spatial features while the network restores resolution. The channel dimensions are gradually decreased by processing each upsampled feature map through a second DoubleConv block after it has been concatenated with its encoder counterpart. The feature representation is projected to three output channels—background, cat, and dog—by a final 1x1 convolution, which corresponds to our segmentation classes. Figure 3 shows the entire architecture, including the encoder-decoder structure and the skip connections.

Although dropout regularization and LeakyReLU activations have been incorporated for more stable training dynamics, the implementation preserves the established benefits of the original UNet design. Additionally, the patch-based training strategy remains computationally moderate throughout both training and inference by making effective feature extraction on localized patches simpler, which allows the model to focus on fine-
grained information essential for accurate class boundary segmentation.

<img width="1248" alt="Screenshot 2025-04-15 at 12 00 15" src="https://github.com/user-attachments/assets/92816a36-15f0-4c83-ac2a-711f46e128b1" />

### Autoencoder-Based Model
----
The segmentation network is structured around a modular convolutional autoencoder, optimized in two stages: unsupervised reconstruction and supervised segmentation. By first training the encoder to reconstruct natural images, we enable it to learn robust low and mid-level visual features, which are subsequently transferred to a segmentation decoder trained on pixel-wise labels.

The shared encoder (Figure 2) consists of four convolutional blocks, each halving the spatial resolution and doubling the number of channels: (3 →32 →64 →128 →256). Each block follows a consistent pattern: Conv2d →ReLU →Dropout(p = 0.1). The encoder terminates in a fully connected bottleneck layer that projects the flattened feature map into a 512-dimensional latent space. This compact latent representation serves as the primary interface between the reconstruction and segmentation branches.

The autoencoder decoder symmetrically mirrors the encoder using ConvTranspose2d layers to progressively upsample spatial dimensions, restoring the image to its original (128 ×128) resolution. Critically skip connections are introduced at each decoding stage by concatenating intermediate encoder activations with corresponding decoder inputs. These connections reintroduce spatially localized features lost during downsampling, a design choice known to enhance the fidelity of reconstructed outputs in convolutional autoencoders Ronneberger et al. (2015).

In the segmentation pathway, we retain the pretrained encoder and freeze its parameters. The segmentation decoder, in contrast, is trained from scratch. It consists of four upsampling blocks, each comprising ConvTranspose2d →BatchNorm2d →ReLU →Dropout, culminating in a ConvTranspose2d layer that outputs raw logits over three semantic classes. Unlike the reconstruction decoder, skip connections are omitted in this path to enforce abstraction over memorization. 

We chose to pretrain using an autoencoding objective because unsupervised reconstruction encourages the encoder to develop spatially informed, task-agnostic representations. These representations when transferred to segmentation, act as a strong initialization especially beneficial in our low-label regime task here. In sum, our model architecture exploits a principled form of transfer learning by repurposing unsupervised image reconstruction as a pretraining task for segmentation. The use of skip connections, dropout regularization, weighted loss, and decoder-only finetuning cohere into a design that is modular and relatively efficient.

<img width="1273" alt="Screenshot 2025-04-15 at 12 02 39" src="https://github.com/user-attachments/assets/46881e04-e707-445a-942e-ce0555b3a42a" />

### CLIP-Based Features Model
----
Our CLIP features segmentation model efficiently performs semantic segmentation by utilizing multimodal representations from the pre-trained CLIP architecture, introduced by Radford et al. (2021). The proposed technique combines text-based semantic information with rich visual representations from images to improve segmentation performance, particularly when distinguishing between visually similar classes like cats and dogs. Descriptive text prompts are automatically generated from filenames to add semantic context (e.g., cat Siamese 27.png creates ”a photo of a Siamese cat”). CLIP’s built-in tokenizer processes these prompts, producing paired visual-textual data points for model input. 

The architecture leverages the pre-trained CLIP model (ViT-L/14 variant) as its visual backbone, processing each input image into visual tokens with 768-dimensional features. We discard the class token and reshape remaining tokens into a spatial grid, while the corresponding textual prompt is encoded and projected onto the visual feature space for dimensional consistency. Our segmentation head concatenates these representations along the channel dimension, passes them through a fusion convolution layer with dropout, and enhances them via a residual block. A channel attention mechanism adaptively reweights channels according to their semantic value before a 1×1 convolution generates segmentation logits, which are upsampled to match the original resolution. Figure 3 illustrates the full architecture.

The CLIP visual backbone remains frozen during training, with only the segmentation head being trained. This approach leverages CLIP’s generalization abilities while ensuring computational efficiency. As our experimental results show, incorporating text-based semantics significantly improves segmentation performance, especially in visually challenging conditions.

<img width="675" alt="Screenshot 2025-04-15 at 12 04 19" src="https://github.com/user-attachments/assets/85a1a70f-2905-499a-a5de-a6d461668443" />

### Prompt-Based Clip Features Model
The segmentation pipeline is built upon a modular encoder-decoder framework, where in a frozen CLIP ViT-L/14 visual encoder generates semantically rich image embeddings, and a lightweight pointer encoder transforms a user-defined click (encoded as a Gaussian heatmap) into a complementary spatial feature representation. These components are integrated via a segmentation head composed of convolutional fusion blocks, and residual module. The architecture is summarized schematically in Figure 4. Given a Gaussian heatmap P ∈ R^{1×224×224} centered at the user-clicked coordinate (see example in Figure ??), a pointer encoder comprising a single Conv2d → BN → ReLU block transforms P to a feature map F_p ∈ R^{768×16×16}, matching the spatial and channel dimensions of V. This enables direct fusion with CLIP’s features via concatenation:
F = [V; F_p] ∈ R^{1536×16×16}
.
The fused representation F is passed through a fusion block composed of:
- Conv2d(1536 → 256) → BN → ReLU → Dropout(p= 0.25)
- Residual Block: two stacked Conv2d(256 → 256) layers with identity skip connection
- Final classifier Conv2d(256 → C), where C is the number of segmentation classes 2
Finally, the coarse logits Z ∈ R^{C×16×16} are bilinearly upsampled to the original input resolution, yielding are obtained. Ŷ ∈ R^{C×224×224}, from which the per-pixel predictions ŷ_{i,j} = arg max_c
Ŷ_{c,i,j} are obtained.

The design integrates CLIP to leverage its zero-shot generalization and semantic richness Radford et al. (2021), while augmenting it with spatially grounded information via the pointer heatmap. Residual blocks and channel attention further enhance local refinement, aligning with best practices in encoder-decoder designs Ronneberger et al. (2015). The result is a robust segmentation pipeline grounded in deep representational learning and interactive spatial prompting.

<img width="1267" alt="Screenshot 2025-04-15 at 12 11 26" src="https://github.com/user-attachments/assets/9619b456-c9b8-4adb-b0b0-4c1cef21064e" />

## Training Strategy & Loss Functions

### Loss Functions
---- 
This section outlines the distinct loss functions used for training the segmentation pipelines. Each model leverages a tailored objective to align with its architecture and task.

#### Mean Squared Error (MSE)
--
<img width="1255" alt="Screenshot 2025-04-15 at 12 12 31" src="https://github.com/user-attachments/assets/a2ab1d4b-0d57-4675-a1bc-97f45b481d4e" />

#### Weighted Cross-Entropy (WCE)
--
<img width="1220" alt="Screenshot 2025-04-15 at 12 13 02" src="https://github.com/user-attachments/assets/39867558-bcbc-475f-a91c-79da0a1f76d7" />

#### Binary Cross-Entropy (BCE)
--
<img width="1221" alt="Screenshot 2025-04-15 at 12 13 29" src="https://github.com/user-attachments/assets/b4257ebc-40ed-4a65-9e70-ee10bf41e6ed" />

#### Cross-Entropy
--
<img width="1218" alt="Screenshot 2025-04-15 at 12 13 58" src="https://github.com/user-attachments/assets/5372f8c1-5120-4c18-b15a-1489af59e0e8" />

#### Dice
--
<img width="1229" alt="Screenshot 2025-04-15 at 12 14 33" src="https://github.com/user-attachments/assets/8c2a0bf5-6508-47af-910c-57a22aa62e48" />

#### Hybrid Segmentation Loss: Combining Cross-Entropy and Dice Metrics
--
<img width="1224" alt="Screenshot 2025-04-15 at 12 15 17" src="https://github.com/user-attachments/assets/d3bf9a6c-f48e-484d-80ed-cf0b8582b713" />

### UNet-Based Model
----
Training was conducted on a GPU-enabled system, leveraging the Adam optimizer (learn-
ing rate = 1 × 10^{−3}, β1 = 0.9, β2 = 0.999) with a Hybrid loss function defined in 4.1.6 to mitigate class imbalance. Each model was trained for 100 epochs with a batch size of 16 and early stopping after 30 epochs without validation improvement. A cosine annealing learning rate scheduler was implemented instead of step-based reduction. The dropout probability was set to 0.25 in DoubleConv blocks, and LeakyReLU activations used a negative slope of 0.1 to prevent dying ReLU problems while maintaining effective gradient flow during backpropagation.

### Autoencoder-Based Model
----
The autoencoder was trained on an unsupervised reconstruction objective using WCE loss, defined in 4.1.2, optimized via Adam (lr = 5 × 10^{−4}, β1 = 0.9, β2 = 0.999). Training was performed on a GPU-enabled system over a maximum of 100 epochs with a batch size of 64, leveraging mixed precision through torch.cuda.amp for improved throughput and reduced memory consumption. Early stopping (patience = 10) was used to halt training when validation loss plateaued. Following pretraining, we discard the decoder and reuse the encoder weights for segmentation. The encoder’s output feature map (B,256,8,8) is input to a newly initialized decoder that is trained to predict per-pixel class logits. Since the encoder remains frozen, the segmentation performance reflects the transferability of features learned during reconstruction. Training was conducted using Adam with a learning rate of 1 ×10−3 over 200 epochs, with early stopping applied (patience = 10) to the mean Intersection-over-Union (mIoU) on the validation set.

### CLIP-Based Features Model
----
Training was conducted on a GPU-enabled system using a pretrained CLIP model (ViT-L/14) as a fixed feature extractor. The CLIP weights were frozen while only the segmentation head was optimized. The dataset comprises images, corresponding masks, and associated text prompts. The segmentation head was trained using the Adam optimizer (learning rate = 1 × 10^{−4}) and optimized with a Cross-Entropy loss defined in 4.1.4. A Cosine Annealing learning rate scheduler gradually reduced the learning rate over 100 epochs, with a batch size of 16. Early stopping based on validation loss with a patience of 20 was implemented to prevent overfitting, and mixed precision training with gradient scaling was employed to improve computational efficiency and stabilize the training process.

### Prompt-Based CLIP Features Model
----
The training pipeline for the prompt-based CLIP features model adhered to a supervised learning framework described above in Section 4.4. To introduce spatial context from the user prompt, we generated Gaussian pointer heatmaps centered on randomly sampled foreground pixels, following the method outlined in our dataset construction script. Each training instance thus consisted of an image, a one-hot pointer heatmap, and a binary segmentation mask identifying the clicked object. Training was conducted using the Adam optimizer (initial learning rate 2 × 10^{−4}), cosine annealing schedule, and BCE with ignore index=255 to exclude the boundary pixels class for loss calculations. Training proceeded for up to 2,000 epochs with early stopping triggered after 10 epochs of stagnant validation loss.

## Results and Evaluation
### Quantitative Metrics & Results
We evaluate segmentation models using standard metrics tailored to the multi-class pixel-wise classification task, where each input image x ∈ R^{3×128×128} is mapped to a mask y ∈ {0,1,2}^{128×128} denoting class labels: background (0), cat (1), and dog (2). Let ŷ denote the model’s predicted segmentation mask, and Y_c, Ŷ_c denote the set of pixels assigned to class c in the ground truth and prediction, respectively.
- Intersection over Union (IoU): Measures the overlap between predicted and ground truth regions for class c: <img width="1235" alt="Screenshot 2025-04-15 at 12 20 59" src="https://github.com/user-attachments/assets/43b3db01-8d97-4b07-b0ba-23e186812cb3" /> where |·|denotes the number of pixels. Higher IoU indicates better spatial alignment.
- Mean Intersection over Union (mIoU): The mIoU is computed as the average of the per-class IoU values: <img width="1225" alt="Screenshot 2025-04-15 at 12 21 57" src="https://github.com/user-attachments/assets/ec91cc19-473e-4c55-a8c8-912e8ae562af" /> where C is the total number of classes. This metric provides a single scalar evaluation of overall segmentation performance.
- Dice Coefficient: A similarity measure for class c that emphasizes agreement, particularly in imbalanced settings: <img width="1221" alt="Screenshot 2025-04-15 at 12 22 34" src="https://github.com/user-attachments/assets/079056fc-dee4-4a7d-9241-a08f2de2ba25" />
- Pixel Accuracy: The proportion of correctly classified pixels across all classes: <img width="1184" alt="Screenshot 2025-04-15 at 12 23 22" src="https://github.com/user-attachments/assets/13928ec4-b77d-44d6-a822-b479e55177f5" /> where H= W = 128 in our case, and ⊮[·] is the indicator function.

<img width="1284" alt="Screenshot 2025-04-15 at 12 23 54" src="https://github.com/user-attachments/assets/b45d4dde-9cae-49e2-91e6-ac89b77ecbe3" />

Table 1 presents a comparative analysis of the four segmentation models developed with the goal of outperforming the baseline model that offers only a reference IoU of 0.33. The UNet model achieves an overall IoU of 0.79, supported by an impressive background IoU of 0.93, and shows moderate performance in object segmentation with cat and dog IoU scores of 0.70 and 0.73, respectively; its Dice coefficient reaches 0.876 and records a pixel accuracy of 93.5%. In contrast, the autoencoder model underperforms relative to the other models, with an overall IoU of 0.46, a background IoU of 0.84, and considerably lower object IoU scores of 0.19 for the cat and 0.36 for the dog; these limitations are further reflected in its Dice coefficient of 0.59 and pixel accuracy of 81.4%. The class imbalance in the dataset, where there are approximately half as many cat images as dog images, cause the performance difference between cat and dog segmentation in both the UNet and autoencoder models. This impact negatively the capacity of the models to effectively segment the underrepresented class. The CLIP-based model stands out as the top performer by registering the highest overall IoU of 0.89, a background IoU of 0.96, and strong object-specific scores of 0.87 for cats and 0.85 for dogs, along with a Dice coefficient of 0.943 and the highest pixel accuracy of 96.7%, all achieved with a low loss of 0.136. By using the pre-trained CLIP ViT-L/14 model and processing textual prompts in addition to image inputs—a feature that the UNet and autoencoder models do not have—the CLIP-based model is able to withstand the class imbalance and take advantage of semantic understanding across modalities rather than depending only on visual patterns in the unbalanced training data. The prompt-based model provides competitive results as well, falling slightly behind the CLIP-based model in several measures with an overall IoU of 0.83, a Dice coefficient of 0.90, and a pixel accuracy of 95.2%. 

In conclusion, while the prompt-based and UNet models offer competitive alternatives, the CLIP-based model offers superior segmentation performance across all assessed metrics and seems more resilient to class imbalance. Nevertheless, the autoencoder consistently performs worse, especially for the underrepresented cat class, highlighting the difficulties it encounters when learning to segment detailed objects from imbalanced data.


### Qualitative Results
----
The qualitative analysis of our four segmentation models, illustrates in Figure 5, reveals distinct performance patterns in the pet segmentation task.

<img width="1149" alt="Screenshot 2025-04-15 at 12 25 40" src="https://github.com/user-attachments/assets/be3c766d-9e5f-4efd-8b51-f9ec02395faf" />

Firstly, the UNet-based model demonstrated reasonable capability in capturing boundary information in well-defined regions, but struggled with cluttered scenes and distinguishing between dogs and cats. Edge definition is generally acceptable, although confusion occurs in complex scenarios such as cluttered images.

The Autoencoder-based model performed significantly worse than all other approaches, exhibiting a critical flaw in its predictions. This model wrongly predicts both cat and dog masks in each image, whereas it should only predict a cat or dog mask for each image. This fundamental misclassification, combined with fragmented and inconsistent masks, renders its predictions largely unusable.

The CLIP-based features model emerged as significantly more robust, particularly in complex situations. Its integration of text prompts explicitly specifying object classes (”cat” or ”dog”) provided a substantial advantage in class disambiguation. The fusion of textual and visual features yielded rich semantic information, enabling more consistent segmentation. While some challenges remain in defining complex animal shapes, these issues are considerably less severe compared to other models.

The Prompt-based CLIP features model exhibited strong capability in delineating animals with complex textures and edges, leveraging textual guidance to enhance class differentiation. However, performance degraded with small objects and poorly placed prompts, although overall performance remained strong.

A clear progression in segmentation quality can be seen in Figure 5 from the especially poor performing Autoencoder to the UNet and finally to the CLIP-based model. This progression correlates with increasing complexity and semantic understanding capabilities.

Small object segmentation remains challenging across all architectures, although the degradation is most severe in the Autoencoder model due to its fundamental misclassification issues and spatial resolution loss.

Our analysis suggests several targeted improvements:
- The UNet model would benefit from diversifying the dataset, addressing class imbalance, or enhancing the model’s ability to take contextual information into account.
- The Autoencoder model requires fundamental redesign to address its critical class confusion problem. Additionally, it might benefit from replacing the WCE loss function with a focal loss to better handle challenging regions and refined augmentation strategies to further enhance robustness.
- The CLIP-based model, while already strong, could be enhanced through architectural refinements to better capture complex object shapes.
- The Prompt-based model would benefit from higher-resolution prompt maps and multi-scale attention mechanisms to enhance fine-grained segmentation performance.

## Robustness Exploration
Robustness tests on our CLIP features segmentation model reveal its resilience under diverse challenging conditions. We evaluated performance against synthetic distortions (Gaussian noise/blur, salt and pepper noise), brightness/contrast variations, and occlusion - simulating real-world degradation scenarios. Results demonstrate that the multimodal approach combining textual and visual features maintains strong semantic representations and preserves object boundaries despite moderate distortions.

As shown in Figure 6, performance degrades gracefully with increasing distortion intensity. The model maintains relatively high segmentation accuracy under moderate noise, blur, occlusion, and brightness/contrast changes. This robustness stems from the integration of CLIP’s complementary modalities and is enhanced by our channel attention mechanism and dropout in the fusion layer, which dynamically emphasize relevant features while mitigating distortion effects.

These evaluations validate our training approach (dropout, mixed precision training with gradient scaling, early stopping, and Cosine Annealing scheduler) and confirm that our CLIP features model can deliver accurate, robust segmentation across challenging environments.

<img width="1484" alt="Screenshot 2025-04-15 at 12 28 16" src="https://github.com/user-attachments/assets/1d0da132-d617-407d-8df3-3acf79c46f5b" />

## Discussion and Conclusion
Our findings underscore several core insights into segmentation performance across architectures. First, pre-training demonstrably enhances results. The CLIP-based model, leveraging pretrained vision-language features, achieved the highest overall performance (mIoU = 0.89, Dice = 0.943, pixel accuracy = 96.7%), substantially outperforming both UNet and the Autoencoder (Table 1). Second, prompt-based segmentation proved effective with minimal input, yielding strong results (Dice = 0.90), despite the architectural simplicity and binary constraint of the task.

Model robustness varied under dataset-specific challenges. Most notably, class imbalance between dogs (27,412 samples) and cats (13,068) skewed predictions, especially for the Autoencoder, which recorded a low Cat IoU of 0.19. UNet, despite its architectural simplicity, retained strong background performance (IoU = 0.93), although object segmentation degraded in low contrast and cluttered scenes, aligning with previous findings on its sensitivity to boundary delineation.

The Autoencoder, adapted via a two-stage training pipeline, suffered from instability particularly during segmentation decoder fine-tuning, where vanishing gradients and feature confusion led to poor generalization (mIoU = 0.46, Fig. 5). While deeper training (loss plateaued at 0.53) may improve results marginally, we posit that this architecture is suboptimal for tasks with limited or imbalanced data due to its reliance on compressed latent representations and lack of direct feature reuse.

In contrast, the CLIP-based model integrated semantic priors from large-scale vision-language pretraining, enabling improved edge localization and object coherence, particularly in ambiguous cases. Prompt-based segmentation extended this approach by conditioning the output on sparse user input (e.g., clicks), achieving usable pixel accuracy (95.2%) and enabling practical applicability across tasks (Fig. 5). Although not achieving CLIP’s performance ceiling, it offers clear extendibility to interactive and multimodal systems.

In sum, this study confirms that modern CV pipelines benefit substantially from pretrained features, architectural efficiency, and multimodal augmentation. Future work should explore contrastive and focal loss techniques to mitigate class imbalance, while expanding prompt-conditioned models into “segment-anything” paradigms through richer input modalities and larger-scale training. These models are increasingly viable for real-world deployment where high accuracy, interactivity, and domain transfer are essential.
