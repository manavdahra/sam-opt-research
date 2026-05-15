# Geometry-Aware Sharpness Minimization: Invariance and Generalization in SAM Variants

## Motivation

Recent advances in deep learning optimization highlight the importance of sharpness-aware methods
that seek flat minima to improve generalization. One such method, Sharpness-Aware Minimization
(SAM) [Foret et al., 2021], has shown promising results. However, it is sensitive to the choice of
model parameterization, as its perturbation radius is defined in the Euclidean metric.
Adaptive SAM (ASAM) [Kwon et al., 2021] and Monge SAM (M-SAM) [Jacobsen and Arvanitidis,
2025] try to address these limitations by introducing scale invariance and geometry-aware metrics,
respectively. However, existing evaluations of M-SAM are limited to a single fine-tuning task on
ResNet-18 and a multi-modal alignment task; a systematic comparison of all three optimizers under
controlled reparametrization across multiple architectures has not been done.
Understanding how these methods behave under reparametrization is crucial for developing more
robust optimization techniques, and this gap motivates our empirical work on benchmarking SAM,
ASAM, and M-SAM on ResNet-18 and ViT-B/16 architectures with a focus on reparametrization
invariance and generalization performance.


## Running the experiments

Vast.ai is being used for renting GPUs to run the experiments.
Following are the steps to run the experiments:
1. Run `uv sync` to install the required dependencies. Make sure you have `uv` installed on your system.

2. Rent a GPU on Vast.ai with pytorch installed. Preferably, use a GPU with at least 16GB of VRAM to accommodate the models and training process.
2. Clone the repository and navigate to the project directory.
3. Install the required dependencies using pip:
   ```
   uv sync
   ```
4. Run the training script with the desired parameters. For example, to train a ResNet-18 model on the CIFAR-10 dataset using SAM, you can use the following command:
   ```
   python train.py --model resnet18 --dataset cifar10 --optimizer sam
   ```


### Important notes from the discussion with Bradley:

- Reparameterization of the model ($\alpha$ and $\frac{1}{\alpha}$ in consecutive layers) may not be a good idea because of batch normalization. I don't remember why. Bradley suggested to instead transform the input data by $\alpha$ and $\frac{1}{\alpha}$ before feeding it to the model. This way, the model itself remains unchanged and we can still achieve the desired effect of reparameterization without interfering with batch normalization. 
- Ideally we should check how the research paper implemented the reparameterization and try to replicate that as closely as possible. If they transformed the input data, we should do the same. If they reparameterized the model, we should also do that, but we need to be careful with batch normalization layers.
- SAM approaches require two forward-backward passes per iteration, which can be computationally expensive. We should consider the trade-off between the potential performance gains from SAM and the increased computational cost. It might be worth experimenting with a smaller subset of the data or a simpler model to see if SAM provides significant benefits before fully committing to it.
