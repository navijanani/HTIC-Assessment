# Question 1 — Imaging Science

## Overview

I implemented an illumination-correction pipeline for grayscale images with uneven lighting.

The observed image can be modeled as:

$$
I(x,y)=R(x,y)L(x,y)
$$

where:

* $I(x,y)$ is the observed image.
* $R(x,y)$ is the surface reflectance or texture.
* $L(x,y)$ is the illumination.

The challenge is that only the observed image is available. Both reflectance and illumination are unknown.

## Approach

Histogram equalization can improve contrast, but it cannot distinguish between actual surface variation and lighting variation. Instead, I separate the illumination and reflectance components based on their spatial frequencies.

I first transform the image into the log domain:

$$
\log I(x,y)=\log R(x,y)+\log L(x,y)
$$

Using:

$$
i(x,y)=\log I(x,y)
$$

$$
r(x,y)=\log R(x,y)
$$

$$
l(x,y)=\log L(x,y)
$$

the model becomes:

$$
i(x,y)=r(x,y)+l(x,y)
$$

This converts the original multiplicative relationship into an additive one.

I assume illumination changes gradually across the image, while texture contains faster spatial variations. Therefore, the low-frequency component can be used to estimate illumination:

$$
\hat l(x,y)=i_{low}(x,y)
$$

The estimated log-reflectance is then:

$$
\hat r(x,y)=i(x,y)-\hat l(x,y)
$$

Finally, I reconstruct the reflectance:

$$
\hat R(x,y)=\exp(\hat r(x,y))
$$

and normalize the result for visualization.

## Implementation

The processing pipeline is:

```text
Input grayscale image
        ↓
Floating-point conversion
        ↓
Log transformation
        ↓
Low-frequency estimation
        ↓
Illumination separation
        ↓
Reflectance reconstruction
        ↓
Normalization
        ↓
Corrected image
```

The implementation keeps the frequency-separation stage independent so that the illumination estimation method can be adjusted without changing the rest of the pipeline.

## Color Extension

The same model can be extended to RGB images:

$$
I_c(x,y)=R_c(x,y)L_c(x,y)
$$

where $c$ represents a color channel.

Correcting each channel independently can alter the original colors. Therefore, the illumination should be estimated while preserving relationships between channels such as:

$$
\frac{R}{G},\qquad \frac{G}{B},\qquad \frac{R}{B}
$$

This allows illumination variation to be reduced without unnecessarily changing the perceived surface color.

## Limitations

The method assumes that illumination varies more slowly than surface texture. Large texture patterns can therefore be mistaken for illumination.

Sharp shadows and sudden lighting changes can also violate this assumption.

Because only a single image is available, reflectance and illumination cannot be uniquely recovered. The reconstructed reflectance should therefore be treated as an estimate rather than an exact physical measurement.

## Result

The pipeline separates slowly varying illumination from higher-frequency surface information and reconstructs an image with more uniform lighting while retaining the visible texture.

---

# Question 2 — Computer Vision

## Overview

I implemented a camera-calibration pipeline to estimate radial lens distortion from a single photograph of a planar rectangular grid.

The implementation handles noisy detections and outliers using RANSAC and robust parameter optimization.

The overall pipeline is:

```text
Grid image
    ↓
Feature detection
    ↓
Camera + distortion model
    ↓
RANSAC
    ↓
Parameter optimization
    ↓
Undistortion
    ↓
Grid reprojection
    ↓
Residual analysis
```

## Camera Model

A point on the planar grid is represented as:

$$
P_i=(X_i,Y_i,0)
$$

and its detected image position as:

$$
p_i=(u_i,v_i)
$$

The camera transforms the world point using rotation and translation:

$$
P_c=RP_w+t
$$

The resulting point is projected into normalized image coordinates.

For pixel coordinates $(u,v)$:

$$
x=\frac{u-c_x}{f_x}
$$

$$
y=\frac{v-c_y}{f_y}
$$

where $f_x$ and $f_y$ are the focal lengths and $(c_x,c_y)$ is the principal point.

The radial distance from the optical axis is:

$$
r^2=x^2+y^2
$$

I use a polynomial radial distortion model:

$$
D(r)=1+k_1r^2+k_2r^4
$$

The distorted coordinates become:

$$
x_d=xD(r)
$$

$$
y_d=yD(r)
$$

## Feature Detection and Outlier Handling

The grid points detected from the input image can contain errors caused by noise, occlusion, lighting variation, perspective distortion, or incorrect detections.

I use RANSAC before the final optimization to prevent these points from strongly affecting the estimated model.

The process repeatedly:

1. Selects a subset of detected points.
2. Estimates a candidate model.
3. Reprojects the grid.
4. Calculates reprojection errors.
5. Classifies points as inliers or outliers.
6. Keeps the model with the strongest inlier support.

The final parameter optimization is performed using the remaining inliers.

## Optimization

For each detected point, I calculate the difference between its observed and predicted positions:

$$
e_i=
\begin{bmatrix}
u_i-u_i^{pred} \
v_i-v_i^{pred}
\end{bmatrix}
$$

The residual magnitude is:

$$
\lVert e_i\rVert_2
$$

The optimization minimizes a robust objective:

$$
E(\theta)=\sum_i\rho\left(\lVert e_i\rVert_2^2\right)
$$

where $\theta$ contains the camera and distortion parameters and $\rho$ is a robust loss function.

The parameters can include:

* Focal length
* Principal point
* Camera rotation
* Camera translation
* Radial distortion coefficients

## Validation

After optimization, I use the estimated parameters to undistort the image.

I then reproject the planar grid back into the original image and compare the predicted positions against the detected positions:

$$
e_i=p_i^{detected}-p_i^{predicted}
$$

This provides a direct measurement of how well the estimated camera model explains the observations.

## Results

The final reprojection measurements are:

| Measurement              |       Result |
| ------------------------ | -----------: |
| Mean residual X          | -0.000000 px |
| Mean residual Y          | -0.000000 px |
| Mean absolute residual X |  0.245648 px |
| Mean absolute residual Y |  0.306147 px |
| Mean residual magnitude  |  0.425650 px |
| RMSE reprojection error  |  0.503742 px |

The mean residuals are close to zero, indicating little systematic bias.

The final RMSE is approximately:

$$
0.504\text{ px}
$$

which indicates that the estimated model reproduces the detected grid positions with a small average positional error.

## Limitations

The accuracy of the calibration depends heavily on the quality of the detected grid points.

Potential sources of error include:

* Incorrect feature detection
* Image noise
* Severe occlusion
* Strong perspective distortion
* Insufficient reliable grid points
* Differences between the selected radial model and the actual lens distortion

More complex lenses may require additional distortion parameters.

---

# Question 3 — Deep Learning

## Overview

I implemented a character-level sequence-to-sequence model for transliterating Romanized Hindi into Devanagari.

For example:

```text
Romanized: ghar
Devanagari: घर
```

Instead of treating each word as a single token, the model processes and generates individual characters.

The architecture is:

```text
Romanized characters
        ↓
Character embedding
        ↓
Encoder RNN
        ↓
Encoder state
        ↓
Decoder RNN
        ↓
Output projection
        ↓
Devanagari characters
```

## Data Processing

I use Romanized–Devanagari word pairs from the Aksharantar dataset released by AI4Bharat.

Before training, the pipeline:

1. Loads and validates the word pairs.
2. Builds source and target character vocabularies.
3. Maps characters to integer IDs.
4. Adds sequence-control tokens.
5. Converts words into character sequences.
6. Pads sequences when required.
7. Creates batches for training.

This keeps the preprocessing pipeline separate from the model architecture.

## Model Architecture

### Character Embedding

Each input character ID is converted into a learned dense representation.

For an embedding dimension $E$, every character is represented by an $E$-dimensional vector.

### Encoder

The encoder processes the input characters sequentially:

$$
x_1,x_2,\ldots,x_T
$$

At each step, the hidden state is updated:

$$
h_t=f(x_t,h_{t-1})
$$

The final encoder state contains information about the complete Romanized input and is passed to the decoder.

### Decoder

The decoder starts from the encoder state and generates the Devanagari sequence one character at a time.

At every decoding step, it uses its current hidden state and the previous output character to predict the next character.

Generation continues until the end-of-sequence token is produced.

### Output Layer

The decoder hidden state is projected to the target vocabulary.

For a target vocabulary containing $V$ characters, the output layer generates $V$ scores at each decoding step.

During inference, these scores are used to select the predicted output character.

## Configuration

I designed the implementation so that the model architecture can be changed without rewriting the training pipeline.

The configurable parameters include:

* Input embedding dimension
* Encoder hidden dimension
* Decoder hidden dimension
* RNN cell type
* Number of encoder layers
* Number of decoder layers

This makes it easier to compare different architectures and training configurations.

## Training

During training, the model predicts the target Devanagari sequence and compares it against the expected sequence.

```text
Romanized input
       ↓
Encoder
       ↓
Decoder
       ↓
Predicted sequence
       ↓
Loss calculation
       ↓
Backpropagation
       ↓
Parameter update
```

Validation data is kept separate from the training data and is used to monitor generalization while experimenting with different configurations.

## Evaluation

After training, the model is evaluated using held-out word pairs that were not used for parameter updates.

For each sample, the model receives a Romanized word and generates its corresponding Devanagari sequence.

The evaluation pipeline can measure both character-level performance and complete-word accuracy depending on the metrics enabled for the experiment.

This also makes it possible to compare multiple model configurations before selecting the final model.

## Computational Complexity

Assume:

* Embedding size = $E$
* Hidden size = $H$
* Sequence length = $T$
* Vocabulary size = $V$
* One encoder layer
* One decoder layer

For a standard RNN:

$$
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
$$

The main multiplication cost for one RNN step is approximately:

$$
EH+H^2
$$

For the encoder:

$$
T(EH+H^2)
$$

For the decoder:

$$
T(EH+H^2)
$$

The output projection requires approximately:

$$
THV
$$

Therefore, the total approximate computation is:

$$
\boxed{2T(EH+H^2)+THV}
$$

This counts the main matrix multiplications and excludes smaller operations such as activations and some bias calculations.

## Parameter Count

For a source vocabulary of size $V$, the input embedding requires:

$$
VE
$$

parameters.

A standard encoder RNN requires:

$$
EH+H^2+H
$$

parameters.

The decoder requires another:

$$
EH+H^2+H
$$

parameters.

The output projection requires:

$$
HV+V
$$

parameters.

The total is therefore:

$$
P=VE+(EH+H^2+H)+(EH+H^2+H)+(HV+V)
$$

which simplifies to:

$$
\boxed{P=VE+2EH+2H^2+2H+HV+V}
$$

## Limitations

The model depends on the transliteration patterns represented in the training dataset. Rare spellings or unseen character combinations can therefore be harder to predict.

A basic encoder-decoder architecture also compresses the input into its hidden-state representation, which can become limiting for longer sequences.

Romanized Hindi can additionally contain ambiguous spellings where multiple Devanagari outputs may be reasonable.

## Conclusion

I built the transliteration system as a configurable character-level encoder-decoder pipeline rather than tying the implementation to one fixed model configuration.

The model handles preprocessing, character embeddings, sequence encoding, decoding, training, inference, and evaluation while allowing the core RNN architecture and dimensions to be changed for experimentation.

I also derived the computational complexity and parameter count for the standard single-layer RNN configuration.