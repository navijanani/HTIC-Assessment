## Question 1 — Imaging Science

### 1. Introduction

The aim of this question is to correct uneven illumination in a grayscale image of a textured surface. The input image has some bright areas and some shadowed areas because the lighting is not uniform. The main goal is to recover the actual texture of the surface as if it was captured under uniform lighting.

The observed image is modeled as:

$$
I(x,y)=R(x,y)\times L(x,y)
$$

where (I(x,y)) is the observed image, (R(x,y)) is the reflectance or actual texture of the surface, and (L(x,y)) is the illumination.

The main difficulty is that only (I(x,y)) is available. Both (R(x,y)) and (L(x,y)) are unknown.

---

### 2. Why Histogram Equalization Is Not Enough

Histogram equalization changes the intensity distribution of an image. It can improve the overall contrast, but it does not understand whether a change in brightness is caused by the surface or by the lighting.

For example, if one part of the surface looks dark, histogram equalization cannot know whether that part is naturally dark or is in shadow.

Therefore, histogram equalization cannot directly separate the reflectance from the illumination.

For this problem, the spatial information of the image is important. We need a method that can estimate the slowly changing illumination and separate it from the texture.

---

### 3. Proposed Method

I use a log-domain approach to separate the reflectance and illumination.

Starting from:

$$
I(x,y)=R(x,y)L(x,y)
$$

taking the logarithm on both sides gives:

$$
\log I(x,y)=\log R(x,y)+\log L(x,y)
$$

Let:

$$
i(x,y)=\log I(x,y)
$$

$$
r(x,y)=\log R(x,y)
$$

$$
l(x,y)=\log L(x,y)
$$

Then:

$$
i(x,y)=r(x,y)+l(x,y)
$$

This changes the multiplication between reflectance and illumination into an addition, which makes their separation easier.

I assume that the illumination changes slowly across the image, while the surface texture has more rapid changes. Therefore, the low-frequency part of the log image can be used as an estimate of illumination.

The high-frequency part can then be used as an estimate of the reflectance.

---

### 4. Frequency Separation

The first step is to convert the input image into the log domain.

The log image can be considered as a combination of two parts:

$$
i(x,y)=i_{low}(x,y)+i_{high}(x,y)
$$

where:

* (i_{low}) represents slowly changing intensity information and is used to estimate illumination.
* (i_{high}) represents faster intensity changes and is used to estimate reflectance.

The illumination estimate is therefore:

$$
\hat l(x,y)=i_{low}(x,y)
$$

The estimated log reflectance is:

$$
\hat r(x,y)=i(x,y)-\hat l(x,y)
$$

Finally, the estimated reflectance is recovered using the exponential operation:

$$
\hat R(x,y)=\exp(\hat r(x,y))
$$

The recovered image is then normalized for display.

---

### 5. Algorithm Steps

The complete grayscale processing procedure is:

1. Read the input grayscale image.
2. Convert the image values to floating-point representation.
3. Convert the image into the log domain.
4. Estimate the low-frequency component manually.
5. Use the low-frequency component as the illumination estimate.
6. Subtract the illumination estimate from the log image.
7. Obtain the high-frequency component as the estimated log reflectance.
8. Apply the exponential operation to recover the reflectance.
9. Normalize the recovered image for visualization.
10. Compare the original image and the corrected reflectance image.

---

### 6. Python Implementation

The implementation is written in Python. The code follows the mathematical steps described above.

The main stages of the implementation are:

```text
Input image
     ↓
Convert to floating point
     ↓
Log transformation
     ↓
Manual low-frequency estimation
     ↓
High-frequency separation
     ↓
Reflectance reconstruction
     ↓
Normalization
     ↓
Output image
```

The complete Python implementation is provided in the GitHub repository under the Question 1 folder.

---

### 7. Experimental Results

For the experiment, I used a single grayscale photograph of a textured surface with non-uniform illumination, as required by the question.

The following intermediate outputs were generated:

**Figure 1: Original grayscale image**

[Insert your original input image here]

The original image contains visible brightness variation caused by uneven illumination.

**Figure 2: Log-domain image**

[Insert your log-domain image here]

The log transformation changes the multiplicative image formation model into an additive model.

**Figure 3: Estimated low-frequency illumination**

[Insert your low-frequency image here]

This component represents the slowly varying brightness pattern caused mainly by illumination.

**Figure 4: Estimated high-frequency component**

[Insert your high-frequency image here]

This component contains the faster spatial variations that are associated with the surface texture.

**Figure 5: Recovered reflectance**

[Insert your final reflectance image here]

The final image represents the estimated surface reflectance after reducing the effect of uneven illumination.

---

### 8. Color Image Extension

The same idea can be extended to a color image. In a color image, illumination can affect the red, green, and blue channels differently.

For each channel, the image can be represented as:

[
I_c(x,y)=R_c(x,y)L_c(x,y)
]

where (c) represents the red, green, or blue channel.

A simple independent correction of each channel may change the original color relationships. Therefore, the illumination correction should be performed jointly so that the true color ratios are preserved as much as possible.

The important color relationships include:

[
\frac{R}{G},\quad \frac{G}{B},\quad \frac{R}{B}
]

The color version of the algorithm therefore estimates the illumination while trying to avoid unnecessary changes to these ratios.

The final corrected color image should have more uniform illumination while keeping the original surface colors visually consistent.

---

### 9. Assumptions

The method is based on the following assumptions:

1. The illumination changes more slowly than the surface texture.
2. The image contains enough texture information to separate high- and low-frequency components.
3. The image can be reasonably represented by the model (I=R\times L).
4. The illumination and reflectance cannot be uniquely recovered from one image without assumptions.
5. Very large texture patterns or sharp shadows may not always satisfy the frequency-separation assumption.

---

### 10. Limitations

The main limitation is that the method depends on the assumption that illumination is smooth and texture changes more rapidly.

If the surface contains very large texture structures, those structures may be treated as part of the illumination. Similarly, strong shadows or sudden lighting changes may not be completely separated from the reflectance.

Therefore, the recovered reflectance is an **estimate** rather than the exact physical reflectance.

---

### 11. Conclusion

In this question, I treated the observed image as the product of surface reflectance and illumination.

Using the logarithmic transformation, the multiplicative model becomes an additive model. I then used the difference between low- and high-frequency information to estimate the illumination and reflectance components.

The estimated reflectance provides a corrected representation of the surface texture with reduced effect from uneven illumination. The same concept can also be extended to color images by considering the RGB channels jointly and preserving their color relationships.
# Question 2 — Computer Vision

## 1. Introduction

The objective of this question is to estimate the radial distortion parameters of a camera using a single photograph of a planar rectangular grid.

The image may contain perspective distortion, radial distortion, noise, lighting variation, and some missing or incorrect grid points. Therefore, the method needs to be robust to outliers.

The main steps used in this work are:

```text
Input grid image
       ↓
Detect grid features
       ↓
Define camera and distortion model
       ↓
RANSAC for outlier removal
       ↓
Robust parameter optimization
       ↓
Estimate distortion parameters
       ↓
Undistort image
       ↓
Reproject grid
       ↓
Calculate residuals and reprojection error
```

This follows the requirements given in the assignment. 

---

## 2. Problem Formulation

A planar grid provides known points on a flat plane. Let a grid point on the plane be represented as:

$$
P_i=(X_i,Y_i,0)
$$

The camera maps this 3D point to an image point:

$$
p_i=(u_i,v_i)
$$

The observed image points are affected by the camera parameters and radial distortion.

The goal is to estimate the parameters that make the projected grid points as close as possible to the detected image points.

---

## 3. Radial Distortion Model

For this implementation, I use a polynomial radial distortion model.

First, the image coordinates are converted into normalized camera coordinates:

$$
x=\frac{u-c_x}{f_x}
$$

$$
y=\frac{v-c_y}{f_y}
$$

where:

* (f_x,f_y) are the focal lengths,
* (c_x,c_y) are the principal point.

The distance from the optical axis is:

$$
r^2=x^2+y^2
$$

The radial distortion factor is modeled using radial coefficients:

$$
D(r)=1+k_1r^2+k_2r^4
$$

The distorted normalized coordinates are then:

$$
x_d=xD(r)
$$

$$
y_d=yD(r)
$$

The exact parameters used in the final implementation are reported together with the experiment results.

---

## 4. Camera Projection

The planar grid is represented in a world coordinate system.

The camera transformation is represented using rotation and translation:

$$
P_c=RP_w+t
$$

where:

* (R) is the camera rotation,
* (t) is the camera translation.

The camera projection converts the 3D point into normalized image coordinates. The radial distortion model is then applied before converting the point into pixel coordinates.

This gives the model-predicted position of each grid point.

---

## 5. Grid Feature Detection

The first practical step is to detect the grid points from the input photograph.

The detected points are matched with their corresponding planar grid coordinates.

The detected points can contain errors because the input image may have:

* noise,
* partial occlusion,
* perspective distortion,
* lighting variation,
* incorrectly detected features.

Therefore, the detected points cannot all be assumed to be correct.

$$Insert detected-grid image here$$

**Figure 1: Detected grid features**

---

## 6. Robust Cost Function

For each detected point, I calculate the difference between the detected position and the position predicted by the camera model.

For point (i), the residual is:

$$
e_i=
\begin{bmatrix}
u_i-u_i^{pred}\
v_i-v_i^{pred}
\end{bmatrix}
$$

The residual magnitude is:

$$
|e_i|_2
$$

The optimization tries to find camera and distortion parameters that minimize these errors.

A robust cost function is useful because large errors may come from incorrect feature detections or occluded points.

The general objective can be written as:

$$
E(\theta)=\sum_i \rho(|e_i|_2^2)
$$

where (\theta) represents the camera and distortion parameters and (\rho) is a robust loss function.

---

## 7. RANSAC Outlier Removal

RANSAC is used to reduce the effect of incorrect detected points.

The basic process is:

1. Select a subset of detected points.
2. Estimate the model from this subset.
3. Calculate the error for all detected points.
4. Classify points with small errors as inliers.
5. Classify points with large errors as outliers.
6. Repeat the process and select the model with the best support.
7. Use the inliers for the final parameter refinement.

This is useful for the given problem because the assignment allows partial occlusion and noisy or incorrect feature detections. 

$$Insert RANSAC inlier/outlier visualization here$$

**Figure 2: RANSAC inliers and outliers**

---

## 8. Parameter Optimization

After removing major outliers, the camera and distortion parameters are refined using the remaining inlier points.

The parameters can include:

* focal length,
* principal point,
* camera rotation,
* camera translation,
* radial distortion coefficients.

The optimization minimizes the reprojection error between the detected grid points and the points predicted by the camera model.

The complete optimization process is:

```text
Initial parameters
       ↓
Project grid points
       ↓
Apply radial distortion
       ↓
Calculate residuals
       ↓
Robust loss
       ↓
Update parameters
       ↓
Repeat until convergence
```

---

## 9. Estimated Parameters

The final estimated camera and distortion parameters from the experiment are reported below.

**Camera parameters:**

$$Insert your actual values here$$

**Radial distortion parameters:**

$$Insert your actual (k_1,k_2,\ldots) values here$$

The values should be taken directly from the final execution of the implementation.

---

## 10. Image Undistortion

After estimating the distortion parameters, the distortion is removed from the image.

The purpose of this step is to obtain an image where the grid geometry is closer to its ideal planar representation.

$$Insert original distorted image here$$

**Figure 3: Original distorted grid image**

$$Insert undistorted image here$$

**Figure 4: Undistorted grid image**

The corrected image can then be used to visualize the estimated grid on an undistorted plane.

---

## 11. Undistorted Grid

The estimated grid points are transformed using the camera model to obtain their positions on the undistorted image.

$$Insert undistorted grid result here$$

**Figure 5: Grid on the undistorted image**

The result should show that the grid geometry becomes more regular after removing the estimated radial distortion.

---

## 12. Reprojection

To check whether the estimated parameters are correct, the grid is projected back into the original distorted image.

For every grid point, two positions are compared:

$$
p_i^{detected}
$$

and

$$
p_i^{predicted}
$$

The residual is:

$$
e_i=p_i^{detected}-p_i^{predicted}
$$

A small residual indicates that the estimated camera and distortion parameters describe the observed image well.

$$Insert reprojected-grid image here$$

**Figure 6: Reprojected grid on the original image**

---

## 13. Reprojection Error

The reprojection errors obtained from the experiment were:

| Measurement              |       Result |
| ------------------------ | -----------: |
| Mean residual X          | -0.000000 px |
| Mean residual Y          | -0.000000 px |
| Mean absolute residual X |  0.245648 px |
| Mean absolute residual Y |  0.306147 px |
| Mean residual magnitude  |  0.425650 px |
| RMSE reprojection error  |  0.503742 px |

The mean residual values are close to zero, which indicates that there is little systematic bias in the final projection.

The RMSE value of approximately **0.504 pixels** indicates that the estimated model reproduces the detected grid positions with a small average positional error.

$$Insert residual visualization here$$

**Figure 7: Reprojection residuals**

---

## 14. Discussion

The complete pipeline combines feature detection, outlier removal, parameter optimization, undistortion, and reprojection.

RANSAC helps reduce the effect of incorrect detections before the final optimization. The final reprojection step provides an independent way to check whether the estimated parameters are consistent with the observed grid.

The low residual values indicate that the fitted model provides a good representation of the detected grid points for this image.

---

## 15. Limitations

The method depends on the quality of the detected grid features.

Possible sources of error include:

* incorrect feature detection,
* strong image noise,
* severe occlusion,
* very strong perspective,
* inaccurate assumptions about the distortion model,
* insufficient number of reliable grid points.

The chosen radial distortion model is also an approximation of the real camera optics. A more complex camera model may be required for images with stronger or more complicated lens distortion.

---

## 16. Conclusion

In this question, I estimated camera radial distortion using a planar grid image.

The process started by detecting grid features and defining a camera projection and radial distortion model. RANSAC was then used to reduce the effect of outliers, followed by robust optimization of the camera and distortion parameters.

The estimated parameters were used to undistort the image and reconstruct the grid. Finally, the grid was reprojected into the original distorted image and the residual errors were calculated.

The final RMSE reprojection error obtained in the experiment was approximately **0.504 pixels**, showing that the estimated model provides a close fit to the detected grid points.

# Question 3 — Deep Learning

## 1. Introduction

The objective of this question is to build a character-level sequence-to-sequence model for transliteration.

The input is a Romanized Hindi word and the output is the corresponding Devanagari word.

For example:

```text
Input:  ghar
Output: घर
```

The task is different from normal word-level translation because the model works with a sequence of **characters**.

The assignment requires an RNN-based sequence-to-sequence model with:

1. Character embedding layer
2. Encoder RNN
3. Decoder RNN
4. Output layer for generating one Devanagari character at a time

The model should also allow the embedding size, hidden size, RNN cell type, and number of encoder/decoder layers to be changed. 

---

## 2. Dataset

For this experiment, I used the sample of the **Aksharantar dataset released by AI4Bharat**, as specified in the assignment.

The dataset contains pairs of Romanized text and the corresponding Devanagari text.

An example pair is:

```text
ajanabee → अजनबी
```

The dataset is divided into training, validation, and test data.

The raw dataset files are kept separately from the source code. The code reads the dataset from the expected local data directory.

---

## 3. Data Preprocessing

Before training the model, the text needs to be converted into a form that the neural network can process.

The main preprocessing steps are:

1. Read the Romanized and Devanagari word pairs.
2. Check and clean the input data.
3. Build a character vocabulary for the input.
4. Build a character vocabulary for the output.
5. Assign an integer ID to each character.
6. Add special tokens required by the sequence-to-sequence model.
7. Convert each word into a sequence of integer IDs.
8. Pad sequences where required.
9. Create batches for training.

The model therefore receives numerical character sequences instead of raw strings.

---

## 4. Model Architecture

The model follows an encoder-decoder sequence-to-sequence architecture.

```text
Romanized input
       ↓
Character Embedding
       ↓
Encoder RNN
       ↓
Encoder hidden state
       ↓
Decoder RNN
       ↓
Output layer
       ↓
Devanagari characters
```

The encoder reads the input characters sequentially.

The final encoder state contains information about the input sequence and is passed to the decoder.

The decoder then generates the output characters one at a time.

---

## 5. Character Embedding

The first layer converts each input character ID into a dense vector.

If the embedding dimension is (E), each character is represented using an (E)-dimensional vector.

This allows the model to learn a useful numerical representation of the input characters instead of treating each character as only an unrelated integer.

---

## 6. Encoder

The encoder is an RNN-based network.

It processes the input sequence one character at a time:

$$
x_1,x_2,\ldots,x_T
$$

At each time step, the encoder updates its hidden state:

$$
h_t=f(x_t,h_{t-1})
$$

The final hidden state contains information about the complete input sequence.

This state is then provided to the decoder.

---

## 7. Decoder

The decoder generates the Devanagari output sequence one character at a time.

For each time step, the decoder receives its previous hidden state and the previous output character.

The general process is:

```text
Encoder hidden state
        ↓
Decoder
        ↓
Character 1
        ↓
Character 2
        ↓
Character 3
        ↓
...
```

The decoder continues until the end-of-sequence token is generated.

---

## 8. Output Layer

The decoder hidden state is passed through an output layer to produce scores for every character in the target vocabulary.

If the target vocabulary contains (V) characters, the output layer produces (V) scores at every decoding step.

The character with the highest predicted probability can be selected during inference.

---

## 9. Configurable Model

One important requirement of the assignment is that the implementation should be flexible.

The following settings can be changed:

* input embedding dimension,
* encoder hidden dimension,
* decoder hidden dimension,
* RNN cell type,
* number of encoder layers,
* number of decoder layers.

The model configuration is kept separate from the main training logic so that different experiments can be performed without rewriting the complete model.

---

## 10. Training

The model is trained using the training portion of the dataset.

During training, the decoder learns to predict the correct next Devanagari character.

The training process is:

```text
Input Romanized word
        ↓
Encoder
        ↓
Encoder state
        ↓
Decoder
        ↓
Predicted Devanagari sequence
        ↓
Compare with target sequence
        ↓
Calculate loss
        ↓
Backpropagation
        ↓
Update model parameters
```

The validation set is used to monitor model performance during training.

The exact optimizer, learning rate, batch size, number of epochs, and other hyperparameters should be reported from the final experiment.

---

## 11. Evaluation

After training, the model is evaluated on data that was not used for parameter updates.

The input is a Romanized word and the model generates the corresponding Devanagari sequence.

For example:

```text
Input:  ghar
Target: घर
Prediction: $$actual model prediction$$
```

The final evaluation results should be reported using the metric(s) actually calculated by the implementation.

### Final Model Result

| Metric          |                 Result |
| --------------- | ---------------------: |
| Training loss   | $$insert actual result$$ |
| Validation loss | $$insert actual result$$ |
| Test accuracy   | $$insert actual result$$ |
| Word accuracy   | $$insert if calculated$$ |

The values above should be replaced with the results from the final run.

---

## 12. Accuracy Improvement

I also evaluated different model settings to improve the transliteration performance.

The experiments should be recorded in the following format:

| Experiment   | Change                |   Result |
| ------------ | --------------------- | -------: |
| Baseline     | Initial configuration | $$result$$ |
| Experiment 1 | $$actual change$$       | $$result$$ |
| Experiment 2 | $$actual change$$       | $$result$$ |
| Final model  | Best configuration    | $$result$$ |

This comparison helps show how the model was improved instead of reporting only the final result.

Possible factors considered in the experiments include model size, embedding dimension, hidden-state dimension, number of layers, RNN cell type, training settings, and data preprocessing.

Only changes that were actually tested should be included in the final table.

---

# 13. Computational Complexity

The assignment asks for the total number of computations using the following assumptions:

* input embedding size = (E)
* hidden size = (H)
* input sequence length = (T)
* output sequence length = (T)
* vocabulary size = (V)
* one encoder layer
* one decoder layer

For a standard RNN, the hidden-state computation can be represented as:

$$
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
$$

The main matrix operations involve the input-to-hidden and hidden-to-hidden transformations.

For one RNN time step, the approximate multiplication count is:

$$
EH+H^2
$$

Therefore, for the encoder over (T) time steps:

$$
T(EH+H^2)
$$

For the decoder, if the decoder input is represented using the same embedding dimension:

$$
T(EH+H^2)
$$

The output projection from hidden state to the target vocabulary requires approximately:

$$
THV
$$

operations.

Therefore, the approximate total computation is:

$$
\boxed{
2T(EH+H^2)+THV
}
$$

This expression counts the main matrix multiplications. The exact count can differ depending on whether additions, bias operations, activation functions, and other implementation details are included.

---

# 14. Number of Parameters

The parameter count can be divided into different parts.

### 14.1 Input Embedding

If the source vocabulary size is (V) and the embedding size is (E):

$$
VE
$$

parameters are required.

### 14.2 Encoder RNN

For a standard RNN:

$$
EH+H^2+H
$$

parameters are required, where the last (H) represents the bias.

### 14.3 Decoder RNN

Similarly:

$$
EH+H^2+H
$$

parameters are required.

### 14.4 Output Layer

The decoder output layer maps the hidden state to the target vocabulary:

$$
HV+V
$$

parameters are required.

### 14.5 Total

Therefore, under these assumptions:

$$
P=VE+(EH+H^2+H)+(EH+H^2+H)+(HV+V)
$$

which can be simplified to:

$$
\boxed{
P=VE+2EH+2H^2+2H+HV+V
}
$$

This calculation assumes a standard RNN, separate source embedding and output projection, and includes biases.

---

# 15. Software Structure

The implementation is organized into separate files so that the model and data processing can be changed independently.

The main structure is:

```text
Question3/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── vocabulary.py
    ├── model.py
    ├── train.py
    └── inference.py
```

The raw dataset is not included in the GitHub repository. The README explains how the dataset should be placed locally before running the code.

---

# 16. Results and Example Predictions

The final report should include examples of the model predictions.

| Romanized Input | Expected Output | Model Output        |
| --------------- | --------------- | ------------------- |
| ghar            | घर              | $$actual prediction$$ |
| $$input$$         | $$target$$        | $$prediction$$        |
| $$input$$         | $$target$$        | $$prediction$$        |

These examples help show whether the model is producing complete and correct Devanagari words.

A training/validation loss graph can also be included if it was generated during training.

**Figure: Training and validation loss**

$$Insert your actual graph here$$

---

# 17. Limitations

The sequence-to-sequence model has some limitations.

First, the model learns from the examples available in the dataset. Words or spelling patterns that are poorly represented in the training data may be difficult to predict.

Second, a basic encoder-decoder model compresses the input sequence into hidden-state information. This can become difficult for longer sequences.

Third, transliteration can sometimes have multiple possible spellings or mappings between Romanized and native-script characters.

The final performance therefore depends on the dataset, vocabulary, model capacity, and training configuration.

---

# 18. Conclusion

In this question, I implemented a character-level RNN-based sequence-to-sequence model for Romanized-to-Devanagari transliteration.

The model uses character embeddings, an encoder RNN, a decoder RNN, and an output layer to generate the target word character by character.

The implementation was designed to allow important model settings such as embedding size, hidden size, RNN cell type, and number of layers to be changed.

I also evaluated the model using held-out data and compared different configurations to understand their effect on performance.

Finally, I derived the computational complexity and parameter count of the network using the assumptions given in the assignment.
