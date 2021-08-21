# Shape Parameters Estimation

## Examples
Provide some examples to illustrate the some of the idea of nuisance parameters estimation
* Huber Penalty

   Generate data with outlier in y and huber penalty provide and more robust results with automatically finding the optimal parameter.

* Quantile Penalty

   Generate data with large percentage positive (or negative) noise in y. And quantile penalty is more flexiable than l2 norm and provide a bette result.
* Quantile Huber Penalty

   Combine the noise we use in huber and quantile penalty makes y contain outlier and "one-side" sign noise. Quantile huber penalty handle this job very well.

* Quantile Regularizer

   This example concerns about sparse recovery in compress-sensing. When the sparse signal have large percentage positive (or negative) components, quantile regularizer provide more freedom to obtain better result.
