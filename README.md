# ! Possible changes

1) Make skip connections in the autoencoder
2) Use the adaptive weighted loss function of the Auto-AD paper in the HADGAN
3) Try to use a mix of the modules in HADGAN and RGAE with some other loss functions
4) Supergraph (KNN) would probably fail in higher dimension 
5) Adaptive weighted loss function and L21 norm are trying to achieve same thing
6) Train Discriminator for some K steps and then train Autoencoder (Generator) for 1 step in this   way autoencoder generalize better

Components-
AE, Discriminative network, spectral-spatial combination AD -> HADGAN
Simpler AE, SuperGraph -> RGAE
Adaptive weighted loss function -> Auto-AD

HADGAN AE + SuperGraph + L21 - DN
