# CVBF Two-Sample Testing

This repository contains code and illustrative examples for the methodology described in the following paper:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "**Use of Cross-validation Bayes Factors to Test Equality of Two Densities**"

Nonparametric hypothesis testing is an important branch of statistics with broad applications. Hart and Choi (2017) propose a nonparametric, Bayesian test to compare the fit of parametric and nonparametric models, termed cross-validation Bayes factor (CVBF). The CVBF is an objective Bayesian procedure where the nonparametric model for densities is a kernel estimate. However, the standard version of kernel estimates cannot be used directly in Bayesian analysis because they become models only after being computed from the data. This issue is resolved by computing a kernel estimate from a subset of the data, and then using the estimate as a model for the remaining data. In this paper, we propose a CVBF approach for testing the equality of two densities, a well-studied subject in the frequentist literature. Our numerical experiments reveal that this new procedure outperforms the Holmes et al. (2015)'s Polya Tree method under the null scenario.

## Installation

The `R` package `BSCRN` provides functions that implement our procedure. The package is available from [Github](https://github.com/naveedmerchant/BayesScreening) with:

``` r
# install.packages("devtools")
devtools::install_github("naveedmerchant/BayesScreening")

```

## Usage

To demonstrate the usage of this package, we apply our test to a set of simulated data.

### Testing whether two data sets share the same distribution

Testing whether two data sets share the same distribution can be challenging if no parametric distribution is specified for the data sets. This package provides functions that implement the CVBF procedure for this problem, allowing the test to be performed without specifying a distribution. The test returns log Bayes factors. A positive log Bayes factor, significantly different from 0, suggests that the two distributions are different. Conversely, a negative log Bayes factor, substantially different from 0, indicates that the two data sets come from the same distribution. If the log Bayes factor is close to 0, the result is inconclusive.

#### Example

We generate samples $X$ and $Y$ according to the null and alternative scenarios described in the paper:

1. Null case: Both $X$ and $Y$ are generated from the standard normal distribution $\phi$. 
2. Alternative case: $X$ is generated from the standard normal distribution $\phi(x)$, while $Y$ is generated from a mixture of two normal distributions: $0.5\phi(x) + 0.5\phi(x/2)/2$. This case represents is from the 'Scale change' scenario discussed in the paper.

``` r
set.seed(100)

numsplits <- 200
trainsize <- seq(from = 20, by = 20, to = 100)

dataset1 <- rnorm(200)
dataset2 <- rnorm(200)

logBFmat.null = matrix(nrow = numsplits, ncol = length(trainsize))
for(j in 1:numsplits){
  for(k in 1:length(trainsize)){
    logBFmat.null[j,k] = CVBFtestrsplit(dataset1 = dataset1, dataset2 = dataset2, trainsize1 = trainsize[k], trainsize2 = trainsize[k])$logBF
  }
}

dataset3 <- NULL
unifdraw = runif(200)
for(j in 1:200){ dataset3[j] = ifelse(unifdraw[j] > 0.5, rnorm(1), rnorm(1, mean = 0, sd = sqrt(4))) }

logBFmat.alt = matrix(nrow = numsplits, ncol = length(trainsize))
for(j in 1:numsplits){
  for(k in 1:length(trainsize)){
    logBFmat.alt[j,k] = CVBFtestrsplit(dataset1 = dataset1, dataset2 = dataset3, trainsize1 = trainsize[k], trainsize2 = trainsize[k])$logBF
  }
}

par(mfrow=c(1,2))
boxplot(logBFmat.null, main="Null case", xlab='trainsize', ylab='log of CVBF',xaxt='n')
axis(1, at=1:length(trainsize), labels=trainsize)
abline(h=0,col='red',lty=2)
boxplot(logBFmat.alt, main="Alternative case", xlab='trainsize', ylab='log of CVBF',xaxt='n')
axis(1, at=1:length(trainsize), labels=trainsize)
abline(h=0,col='red',lty=2)

```

### Posterior predictive checking of the test

To construct a Bayesian nonparametric test, one should adopt a Bayesian nonparametric model for the underlying distribution of the dataset. By examining the posterior predictive distribution of this model, we can gain insights into what the procedure infers about the data's underlying distribution.

#### Example

We generate a sample from the standard normal distribution and observe how our cross-validatory method estimates the underlying density.

``` r
set.seed(100)

dataset = rnorm(500)
XT1 = dataset[1:250]; XV1 = dataset[251:500]

predbwvec = PredCVBFIndepMHbw(ndraw = 200, maxIter = 1000, XT1 = XT1, XV1 = XV1)
predpostsamp = PredCVBFDens(predbwvec$predbwsamp, XT1 = XT1)

par(mfrow=c(1,1))
xgrid <- seq(from = min(dataset), to = max(dataset), length.out = 100)
plot(xgrid , predpostsamp(xgrid),type='l', xlab = "x", ylab = "Density", main = "Posterior predictive of CVBF vs True density")
hist(XV1, breaks = 23, add = T, freq = FALSE,col = 'grey80', border = 'grey70')
lines(xgrid, predpostsamp(xgrid),lwd=2)
lines(xgrid, dnorm(xgrid), col = "blue",lwd=2,lty=2)
legend("topright", legend=c("CVBF", "True"), col=c('black','blue'), lty=c(1,2),lwd = c(2,2), bty="n")

```

### Application to Higgs boson data

We provide codes to reproduce the results of the data analysis for the Higgs boson dataset (Section 7). The dataset can be downloaded from the [UCI Machine Learning repo](https://archive.ics.uci.edu/dataset/280/higgs).

* `HiggsBosonCol23andCol29withRpackage.Rmd` applies CVBF to a subset of the 23rd and 29th column of the dataset (Figure 9).
* `HiggsBosonRfilewithRpackage.Rmd` uses the same dataset to compute predictive posteriors of the densities, providing a comparison between our method and Polya Trees (Figure 8,10).
    
## References
* Holmes, C.C. et al. (2015). Two-sample Bayesian Nonparametric Hypothesis Testing. Bayesian Analysis, 10(2):297-320. (https://projecteuclid.org/euclid.ba/1422884976)
* Hart, J.D. and Choi, T. (2017). Nonparametric Goodness of Fit via Cross-Validation Bayes Factors. Bayesian Analysis, 12(3):653-677. (https://projecteuclid.org/journals/bayesian-analysis/volume-12/issue-3/Nonparametric-Goodness-of-Fit-via-Cross-Validation-Bayes-Factors/10.1214/16-BA1018.full)
