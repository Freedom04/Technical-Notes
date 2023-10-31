# Implementing scMVP

[TOC]

## 1 Create and manage a virtualenv

Pipenv is a Python virtualenv management tool that supports many systems and nicely bridges the gaps between pip, python (using system python, pyenv, or asdf), and virtualenv. Therefore, I decided to use `pipenv` to create and manage a virtualenv for my projects.

The packages that have been installed are as follows:

```bash
# CUDA 11.7
pipenv install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pipenv install pandas
pipenv install scipy
pipenv install scikit-learn==0.22.2
pipenv install scanpy
```

## 2 Dataset

### 2.1 sciCAR dataset

For the sci-CAR dataset, only co-assay cells were used for further analysis, and cells with fewer than 200 peaks or genes and peaks or genes with fewer than 10 cells were removed from further analysis.

### 2.2 Paired-seq dataset

For Paired-seq dataset, cells with fewer than 200 peaks or genes, peaks, or genes with fewer than 10 cells or peaks with more than 336 cells were removed from further analysis.

### 2.3 SNARE-seq dataset

For SNARE-seq dataset, cells with fewer than 200 peaks or genes and peaks or genes with fewer than 10 cells were removed from further analysis.

## 3 The Model

scMVP consists of a two-channel encoder network and a two-channel decoder to integrate the information from scRNA-seq and scATAC-seq, and the input dimension of each channel is determined by the gene and peak number.

![scMVP](resources/scMVP.png)

### 3.1 A two-channel encoder

scMVP uses a **mask attention channel** for the RNA branch and a **self-attention channel** for the ATAC branch to identify the cell type-associated information and capture the intra-omics distal correlation. The output two channels are combined together to form a shared linear layer (256 dimensions).

<img src="resources/two_channel_encoder_network.png" alt="two_channel_encoder_network" style="zoom: 45%;" />

#### 3.1.1 The RNA branch of the encoder

RNA branch of the encoder sequentially concatenates a 128-dimensional hidden layer, a layer normalization layer, a batch normalization layer, and an output Relu activation layer, which is weighted by a mask attention tensor generated from the first 128-dimensional hidden layer.

<img src="resources/RNA_branch_of_encoder.png" alt="RNA_branch_of_encoder" style="zoom: 45%;" />

#### 3.1.2 The ATAC branch of the encoder

The ATAC branch of the encoder sequentially concatenates a 128-dimensional hidden layer, a batch normalization layer, a Relu activation layer, and a multi-heads self-attention layer, which is designed as 8 self-attention heads and each head takes a 16-dimension feature in this study, and a layer normalization.

<img src="resources/ATAC_branch_of_encoder.png" alt="ATAC_branch_of_encoder" style="zoom: 45%;" />

### 3.2 The attention module

The attention module receives the $p(c|z)$ for all K components as input, by a linear layer (128 dimensions), and then weights the last layer of each decoder channel with a SoftMax activation function.

### 3.3 A two-channel decoder

A two-channel decoder is employed to determine the distribution parameters of NB and ZIP for the reconstruction of scRNA-seq and scATAC-seq, which utilize a similar network structure with the network except for **an attention module**.

<img src="resources/two_channel_decoder_network.png" alt="two_channel_decoder_network" style="zoom:45%;" />
$$
p(c)=Cat(\pi)=\prod_{k=1}^K{\pi _k^{c_k}}, \ \pi = [\pi_1,\pi_2,...,\pi_K]
$$

$$
p(z|c)=N(z|\mu_c,\sigma_cI)=\frac{1}{\sqrt{2\pi}\sigma}e^{(-\frac{(z-\mu_c)^2}{2\sigma^2_c})}
$$

Here, $c$ represents one of the $K$ components(clusters) of Gaussian mixture distribution, which is extracted from a categorical distribution with probability $\pi_c$, then the common embedding latent variable $z$ is derived from the component $c$ with a probability $p(z|c)=N(z|\mu_c,\sigma_cI)$, which means the latent variable $z$ associated cells belongs to a specific cluster (cell type) $c$.
$$
\alpha_x,\beta_x=Decoder_x(z)
$$

$$
p\left(\mu_x \mid \alpha_x, \beta_x\right)=Gamma\left(\alpha_x, \beta_x\right)=\frac{\beta_x^{\alpha_x} \bar{x}^{\alpha_x-1} e^{-\beta_x \bar{x}}}{\Gamma\left(\alpha_x\right)}
$$

$$
p\left(\mathrm{x} \mid \mu_x\right)=Poisson\left(\mu_x\right)=\frac{\mu_x^{\mathrm{x}}}{\mathrm{x} !} e^{-\mu_x}
$$

Then, a two-channel decode network is used to generate the parameters of the NB and ZIP distribution to reconstruct the original observed $x$ (RNA) and TF-IDF transformed $y$ (ATAC) from the common latent variable $z$. As a result, the RNA counts can be imputed with the mean of the Poisson distribution.
$$
\mu_y, \tau_y=Decoder_y(z)
$$

$$
p\left(\bar{y} \mid \mu_y\right)={Poisson}\left(\mu_y\right)=\frac{\mu_y^{\bar{y}}}{\bar{y} !} e^{-\mu_y}
$$

$$
\mathrm{p}\left(\omega_y \mid \tau_y\right)={Bernoulli}\left(\tau_y\right)=\tau_y{ }^{\omega_y}\left(1-\tau_y\right)^{1-\omega_y}
$$

$$
\begin{equation}
	\begin{aligned}
	\mathrm{p}(\mathrm{y} \mid \bar{y}, \omega_y) =&\ \left[p\left(\bar{y} \mid \mu_y\right) * \mathrm{p}\left(\omega_y=1 \mid \tau_y\right)\right]_{y>0}  \\
	& +\left[\mathrm{p}\left(\omega_y=0 \mid \tau_y\right)+p\left(\bar{y} \mid \mu_y\right) * \mathrm{p}\left(\omega_y=1 \mid \tau_y\right)\right]_{y=0}
	\end{aligned}
\end{equation}
$$

Similarly, the ZIP distribution is decomposed as a Poisson distribution and a Bernoulli distribution, and the mean $μ_y$ of the Poisson distribution is worked here as the imputation of scATAC-seq data

