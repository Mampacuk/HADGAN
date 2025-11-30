# ! References

# ! Loss
• Reconstruction Loss Function- L2,1 norm loss and spectral angle mapper(SAM) loss:
   To train the autoencoder for hyperspectral anomaly detection, a hybrid loss function was designed that jointly minimizes spatial and spectral reconstruction errors. The objective is to ensure both pixel-wise intensity accuracy and spectral consistency between the input and its reconstruction.

• Latent Consistency Loss:
   In addition to the reconstruction loss, a latent consistency constraint was introduced to stabilize the learned representation space and enhance generalization.
   The intuition is that when a latent code z is decoded into its reconstructed input and then re-encoded, the resulting latent vector should be close to the original z. This enforces cycle-consistency in the encoder–decoder pair.

• Latent Shrinkage Loss:
    To further regularize the latent space and enhance anomaly discrimination, a shrinkage loss term was incorporated. The goal of this loss is to encourage sparsity and prevent the encoder from generating large, noisy activations. It penalizes the magnitude of the latent activations, driving them towards zero.

    When encoder converts input x into a latent vector z=E(x), we want:
    Normal data → small, compact, consistent latent vectors
    Anomalous data → larger deviations in latent space

# ! Model architecture
The proposed model follows an autoencoder-based framework, consisting of an Encoder and a Decoder, both built using stacked fully connected (FC) layers with optional batch normalization and LeakyReLU activation.
To enforce distributional alignment and improve the realism of reconstructed and latent representations, two discriminators were introduced — a Latent Discriminator and an Image Discriminator.
Both are simple multilayer perceptrons (MLPs) trained adversarially against the encoder–decoder network.

• FCBlock:
    FCBlock serves as the fundamental building block for both the encoder and decoder.
    Each block performs:
        A fully connected (Dense) transformation
        Batch Normalization (to stabilize training and control internal covariate shift)
        LeakyReLU activation (to allow small gradients for negative inputs and avoid dead neurons)
        
    This modular design allows easy stacking of multiple such layers to control model depth.

• Encoder: 
    The Encoder maps high-dimensional spectral input x∈R to a low-dimensional latent representation z∈Rd
    Two sequential FCBlocks progressively extract nonlinear features.
    The final Dense layer (latent) compresses the information into a latent vector z.
    A Dropout layer is applied during training in latent space to prevent overfitting and improve robustness.

• Decoder:
    The Decoder reconstructs the original input spectrum from its latent code z
    Two FCBlocks with nonlinear transformations expand the latent vector back to spectral space.
    The final output layer (Dense(B)) produces a reconstruction x^ of the same dimensionality as the input x.

• Latent Disriminator:
    The Latent Discriminator operates in the latent space.
    Its objective is to distinguish between latent vectors z=E(x) produced by the encoder and samples drawn from a target prior distribution.
    It encourages the encoder to produce latent representations that follow a smooth, continuous prior distribution, improving generalization and anomaly separability.

    The final output is a scalar (real vs. fake score).

• Image Discriminator:
    The Image Discriminator operates in the input (spectral) space.
    It aims to distinguish between real input samples x and reconstructed samples x^=D(E(x))
    This adversarial supervision pushes the decoder to generate more realistic and spectrally consistent reconstructions, reducing blur and over-smoothing in the spectral domain.

The discriminators are trained adversarially alongside the autoencoder:
    The discriminators learn to classify real vs. generated samples correctly.
    The encoder–decoder learns to fool the discriminators by producing latent codes and reconstructions indistinguishable from real data.

This results in:
    Sharper and more realistic reconstructions,
    Well-regularized latent space, and
    Improved anomaly detection performance by reducing distributional mismatch between real and reconstructed features.

# ! Training Methodology
The goal:
    Learn a latent representation z that captures normal data structure and reconstructs input well — while making reconstructions and latent codes indistinguishable from real samples to the discriminators.
• Each training iteration consists of:
    1. k-step Discriminator Update
        In every iteration, both discriminators are updated for k steps before the generator update.
        This alternating training approach (k-step discriminator updates followed by one generator update) improves convergence and prevents mode collapse — a common issue in adversarial models and ensures that the disriminators provide accurate and informative gradients for the subsequent generator update.

    2. Generator (Encoder–Decoder) Update
        Once the discriminators are updated, the encoder–decoder pair is optimized with two goals:
        Reconstruction fidelity – to accurately reproduce input samples.
        Adversarial realism – to generate outputs that can fool both discriminators.

# ! Training Procedure
The HADGAN model was trained using a structured and patch-based training pipeline designed for    hyperspectral image (HSI) data. The process ensures efficient utilization of memory, robust learning across spatial variations, and stable adversarial optimization.

• Patch-based Dataset 
    The hyperspectral cube was divided into overlapping spatial patches of size (Hk,Wk,Bk) to enable localized learning while preserving spectral–spatial correlations.

    During initial exploration of the hyperspectral dataset, it was observed that the captured scene was slightly tilted, leading to several spatial regions containing minimal or no valid spectral information (represented as black or zero-value areas).
    Including such patches during training could bias the model, as reconstructing regions with no meaningful data is a trivial task.
    
    To ensure the model focuses on learning the spectral–spatial structure of meaningful regions rather than empty areas, a patch filtering criterion was applied. Only patches containing at least 80% valid pixels (i.e., less than 20% blacked or zero-value data) were included in the training set. This selective sampling allowed the model to learn from spectrally informative regions

• Model's generalization
    To enhance the model’s generalization capability and prevent overfitting to specific spatial layouts, data augmentation was employed on these valid patches. Augmentation operations such as random flips, rotations etc were applied so that the model encountered slightly varied versions of the same region during training.
    This encouraged the model to focus on spectral consistency and reconstruction quality rather than memorizing fixed spatial patterns, thereby improving its ability to detect anomalies under diverse scene orientations.

# ! Inferencing
• Patch-wise Reconstruction
    The input hyperspectral cube was divided into smaller overlapping or non-overlapping spatial patches, depending on the overlap parameter.
    Each patch was independently passed through the trained encoder–decoder network to obtain its reconstructed counterpart.
    Although an overlap of 0% was used during the final evaluation (for faster inference and reduced redundancy), the framework supports configurable overlaps.
    In cases where overlaps are introduced, a mean-based aggregation strategy can be applied to average the reconstruction of shared pixels, resulting in smoother boundary transitions between adjacent patches.

• Residual Map Computation
    Once the reconstruction was complete, a residual map was generated by computing the absolute difference between the original and reconstructed hyperspectral data across all spectral bands.
    This residual map captures localized deviations from the learned normal data distribution — regions with higher residuals indicate pixels that the model could not accurately reconstruct, thereby hinting at possible anomalies.
    A validity mask was applied to ensure only spatial regions containing real data were evaluated, discarding zero or blacked-out regions from contributing to the anomaly score.

• Post-Processing and Masking Methodology
    Region Selection:
    A polygon-based valid mask was applied to focus processing within the scene’s region of interest and exclude non-imaged borders.

    Informative Band Selection:
    PCA was applied on the residual cube, and top-energy components were used to identify the most discriminative spectral bands for spatial analysis, reducing redundancy and noise sensitivity.

    Spatial Detection:
    Spatial anomalies were enhanced using a guided-filter-based detector combining edge and variance information.
    This helped suppress homogeneous regions while emphasizing localized discontinuities that typically indicate manmade structures.

    Spectral Detection:
    A fast RX-style detector was implemented using PCA and Ledoit–Wolf covariance estimation for global background modeling, coupled with a local RX approximation to account for contextual variability.

    Spectral–Spatial Fusion:
    The spatial and spectral anomaly maps were fused through multiplicative combination, ensuring that only pixels exhibiting both spectral and spatial distinctiveness were retained as anomalies.

    Masking of Natural Regions:
    Finally, vegetation and water areas were excluded using a manual geospatial mask derived from pre-computed satellite layers to eliminate residual natural artifacts.

# !

Components-
AE, Discriminative network, spectral-spatial combination AD -> HADGAN
Simpler AE, SuperGraph -> RGAE
Adaptive weighted loss function -> Auto-AD

HADGAN AE + SuperGraph + L21 - DN

# ! What happens when you train GANs (or any neural net) patch-by-patch?
This often improves robustness because the model can’t just memorize one patch distribution — it keeps adapting to new local variations in the data.

# ! Why this could help robustness in HSI anomaly detection
Hyperspectral images are very heterogeneous:
Some patches may be homogeneous vegetation, others contain man-made structures, others anomalies.
By fine-tuning iteratively on these patches, the discriminator keeps learning “rules” that generalize across local distributions instead of overfitting to global statistics.
The generator, in turn, must keep adapting to fool a discriminator that is being shown diverse examples sequentially.
So instead of the model “locking in” to one particular distribution too early, it keeps being nudged in many small, diverse directions. This can help with anomaly detection where robustness matters.

# ! A common compromise is:
Not strictly “one patch at a time”, but small random batches of patches (say 4–16 at once). This still gives robustness + efficiency.

# ! Hyperparameters-
epochs
Window size - Hk, Wk, Bk=B
overlap in training and inferencing
data augmentation
k steps training
GAN loss function

# ! Imp Points
Postprocessing
Preprocessing
Man made anomalies
Tilted hsi image but straight GT
The GT is not tilted maybe and we made a tilted anomaly map so training our model on the actual image and then checking for f1, etc on that image's binary map and GT may give us correct parameters
Post processing changes to be made- 
        Anomaly Scoring Enhancement (RX Detector)
        The final scoring needs the most improvement. The Mahalanobis (RX) score in spectral_detector_from_residual is calculated using EmpiricalCovariance over all residual pixels. If your MOCK dataset has a dense anomaly cluster, this estimate is contaminated.

        Improvement: Use Robust Covariance Estimation by excluding high-residual outliers (put this logic inside a copy of spectral_detector_from_residual):

        Temporarily compute initial Mahalanobis distance.

        Filter out the top 5% (or 10%) of pixels based on this distance.

        Refit the EmpiricalCovariance using only the remaining "background" pixels to get a cleaner precision matrix.

        Compute the final RX score using this robust precision matrix. This significantly boosts performance on real-world HSI data.   
 
So now I am training my HADGAN model on prisma hsi images and I am feeding them patch wise with each patch augmented so that it gets almost a different type of patch each time
But there is an issue my prisma hsi image is tilted and when I am sending patches onto my model some contain the region which should not be there like the main part of the image is in the tilt square but my patches are also covering the region which should not be there

