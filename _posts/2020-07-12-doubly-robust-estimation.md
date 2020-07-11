---
layout: tutorial
tags: [Econ]
comments: true
title: 12 - Doubly Robust Estimation
subtitle: Merging Propensity Score and Regression
date: 2020-07-12
true-dt: 2020-07-12
author: "Matheus Facure"
---

## Don't Put All your Eggs in One Basket

We've learned how to use linear regression and propensity score weighting to estimate \\(E[Y\|Y=1] - E[Y\|Y=0] \| X\\). But which one should we use and when? When in doubt, just use both! Doubly Robust Estimation is a way of combining propensity score and linear regression in a way you don't have to rely on any single one of them. 

To see how this works, let's consider the mindset experiment. It is a randomised study conducted in U.S. public high schools which aims at finding the impact of a growth mindset. The way it works is that students receive from the school a seminary to instil in them a growth mindset. Then, they follow up the students on college years to measure how well they performed academically. This measurement was compiled into an achievement score and standardised. The real data on this study is not publicly available in order to preserve students' privacy. However, we have a simulated dataset with the same statistical properties provided by [Athey and Wager](https://arxiv.org/pdf/1902.07409.pdf), so we will use that instead.


```python
data = pd.read_csv("./data/learning_mindset.csv")
data.sample(5, random_state=5)
```


<div>
<table class="table table-bordered table-striped table-hover" border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schoolid</th>
      <th>intervention</th>
      <th>achievement_score</th>
      <th>...</th>
      <th>school_ethnic_minority</th>
      <th>school_poverty</th>
      <th>school_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>259</th>
      <td>73</td>
      <td>1</td>
      <td>1.480828</td>
      <td>...</td>
      <td>-0.515202</td>
      <td>-0.169849</td>
      <td>0.173954</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>76</td>
      <td>0</td>
      <td>-0.987277</td>
      <td>...</td>
      <td>-1.310927</td>
      <td>0.224077</td>
      <td>-0.426757</td>
    </tr>
    <tr>
      <th>9963</th>
      <td>4</td>
      <td>0</td>
      <td>-0.152340</td>
      <td>...</td>
      <td>0.875012</td>
      <td>-0.724801</td>
      <td>0.761781</td>
    </tr>
    <tr>
      <th>4488</th>
      <td>67</td>
      <td>0</td>
      <td>0.358336</td>
      <td>...</td>
      <td>0.315755</td>
      <td>0.054586</td>
      <td>1.862187</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>16</td>
      <td>1</td>
      <td>1.360920</td>
      <td>...</td>
      <td>-0.033161</td>
      <td>-0.982274</td>
      <td>1.591641</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 13 columns</p>
</div>



Although the study was randomised, it doesn't seem to be the case that this data is free from confounding. One possible reason for this is that the treatment variable is measured by the student's receipt of the seminar. So, although the opportunity to participate was random, participation is not. We are here dealing with a case of non-compliance here. One evidence of this is how the student's success expectation is correlated with the participation in the seminar. Students with higher self-reported participation are more likely to have joined the growth mindset seminar.


```python
data.groupby("success_expect")["intervention"].mean()
```


    success_expect
    1    0.271739
    2    0.265957
    3    0.294118
    4    0.271617
    5    0.311070
    6    0.354287
    7    0.362319
    Name: intervention, dtype: float64



Since we will use linear methods (linear regression and logistic regression), we need to convert the categorical variables to dummies.


```python
categ = ["ethnicity", "gender", "school_urbanicity"]
cont = ["school_mindset", "school_achievement", "school_ethnic_minority", "school_poverty", "school_size"]

data_with_categ = pd.concat([
    data.drop(columns=categ), # dataset without the categorical features
    pd.get_dummies(data[categ], columns=categ, drop_first=False)# dataset without categorical converted to dummies
], axis=1)

print(data_with_categ.shape)
```

    (10391, 32)


We are now ready to understand how doubly robust estimation works.

## Doubly Robust Estimation

![img](/img/econ/doubly-robust/double.png)

Instead of deriving the estimator, I'll first show it to you and only then tell why it is awesome.

$$
\hat{ATE} = \frac{1}{N}\sum \bigg( \dfrac{T_i(Y_i - \hat{\mu_1}(X_i))}{\hat{P}(X_i)} + \hat{\mu_1}(X_i) \bigg) - \frac{1}{N}\sum \bigg( \dfrac{(1-T_i)(Y_i - \hat{\mu_0}(X_i))}{1-\hat{P}(X_i)} + \hat{\mu_0}(X_i) \bigg)
$$

where \\(\hat{P}(x)\\) is an estimation of the propensity score (using logistic regression, for example), \\(\hat{\mu_1}(x)\\) is an estimation of \\(E[Y\|X, T=1]\\) (using linear regression, for example), and \\(\hat{\mu_1}(x)\\) is an estimation of \\(E[Y\|X, T=0]\\). As you might have already guessed, the first part of the doubly robust estimator estimates \\(E[Y_1]\\) and the second part estimates \\(E[Y_0]\\). Let's examine the first part, as all the intuition will also apply to the second part by analogy.

But first, let's see it in practice.


```python
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )
```


```python
T = 'intervention'
Y = 'achievement_score'
X = data_with_categ.columns.drop(['schoolid', T, Y])

doubly_robust(data_with_categ, X, T, Y)
```


    0.3882218480614833


Doubly robust estimator is saying that we should expect individuals who attended the mindset seminar to be 0.388 standard deviations above their untreated fellows, in terms of achievements. Once again, we can use bootstrap to construct confidence intervals.


```python
from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(doubly_robust)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                          for _ in range(bootstrap_sample))
ates = np.array(ates)
```


```python
print(f"ATE 95% CI:", (ates.mean() - 1.96*ates.std(), ates.mean() + 1.96*ates.std()))
```

    ATE 95% CI: (0.3530692065036196, 0.42163132128223224)



![png](/img/econ/doubly-robust/output_13_0.png)


Now, let's examine why the doubly robust estimator is so great. It is called doubly robust because it only requires one of the models, \\(\hat{P}(x)\\) or \\(\hat{\mu}(x)\\), to be correctly specified. To see this, take the first part that estimates \\(E[Y_1]\\) and take a good look at it.

$$
\hat{E}[Y_1] = \frac{1}{N}\sum \bigg( \dfrac{T_i(Y_i - \hat{\mu_1}(X_i))}{\hat{P}(X_i)} + \hat{\mu_1}(X_i) \bigg)
$$

First, assume that \\(\hat{\mu_1}(x)\\) is correct. If the propensity score model is wrong, we wouldn't need to worry. Because if \\(\hat{\mu_1}(x)\\) is correct, then \\(E[T_i(Y_i - \hat{\mu_1}(X_i))]=0\\). That is because the multiplication by \\(T_i\\) selects only the treated and the residual of \\(\hat{\mu_1}\\) on the treated have, by definition, mean zero. This causes the whole thing to reduce to \\(\hat{\mu_1}(X_i)\\), which is correctly estimated \\(E[Y_1]\\) by assumption. So, you see that by being correct, \\(\hat{\mu_1}(X_i)\\) wipes out the relevance of the propensity score model. We can apply the same reasoning to understand the estimator of \\(E[Y_0]\\). 

But don't take my word for it. Let the code show you the way! In the following estimator, I've replaced the logistic regression that estimates the propensity score by a random uniform variable that goes from 0.1 to 0.9 (I don't want very small weights to blow up my propensity score variance). Since this is random, there is no way it is a good propensity score model, but we will see that the doubly robust estimator still manages to produce an estimation that is very close to when the propensity score was estimated with logistic regression.


```python
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_wrong_ps(df, X, T, Y):
    # wrong PS model
    np.random.seed(654)
    ps = np.random.uniform(0.1, 0.9, df.shape[0])
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )
```


```python
doubly_robust_wrong_ps(data_with_categ, X, T, Y)
```


    0.3797180320806519


If we use bootstrap, we can see that the variance is slightly higher than when the propensity score was estimated with a logistic regression.


```python
from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
parallel_fn = delayed(doubly_robust_wrong_ps)
wrong_ps = Parallel(n_jobs=4)(parallel_fn(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                              for _ in range(bootstrap_sample))
wrong_ps = np.array(wrong_ps)
```


```python
print(f"ATE 95% CI:", (wrong_ps.mean() - 1.96*wrong_ps.std(), wrong_ps.mean() + 1.96*wrong_ps.std()))
```

    ATE 95% CI: (0.3398327292509242, 0.43263125686601944)


This covers the case that the propensity model is wrong but the outcome model is correct. What about the other situation? Let's again take a good look at the first part of the estimator, but let's rearrange some terms

$$
\hat{E}[Y_1] = \frac{1}{N}\sum \bigg( \dfrac{T_i(Y_i - \hat{\mu_1}(X_i))}{\hat{P}(X_i)} + \hat{\mu_1}(X_i) \bigg)
$$

$$
\hat{E}[Y_1] = \frac{1}{N}\sum \bigg( \dfrac{T_iY_i}{\hat{P}(X_i)} - \dfrac{T_i\hat{\mu_1}(X_i)}{\hat{P}(X_i)} + \hat{\mu_1}(X_i) \bigg)
$$

$$
\hat{E}[Y_1] = \frac{1}{N}\sum \bigg( \dfrac{T_iY_i}{\hat{P}(X_i)} - \bigg(\dfrac{T_i}{\hat{P}(X_i)} - 1\bigg) \hat{\mu_1}(X_i) \bigg)
$$

$$
\hat{E}[Y_1] = \frac{1}{N}\sum \bigg( \dfrac{T_iY_i}{\hat{P}(X_i)} - \bigg(\dfrac{T_i - \hat{P}(X_i)}{\hat{P}(X_i)}\bigg) \hat{\mu_1}(X_i) \bigg)
$$

Now, assume that the propensity score \\(\hat{P}(X_i)\\) is correctly specified. In this case, \\(E[T_i - \hat{P}(X_i)]=0\\), which wipes out the part dependent on \\(\hat{\mu_1}(X_i)\\). This makes the doubly robust estimator reduce to the propensity score weighting estimator \\(\frac{T_iY_i}{\hat{P}(X_i)}\\), which is correct by assumption. So even if the \\(\hat{\mu_1}(X_i)\\) is wrong, the estimator will still be correct, provided that the propensity score is correctly specified.

Once again, if you believe more in code than in formulas, here it is the practical verification. In the code below, we've replaced both regression models with a random normal variable. So, no doubt that \\(\hat{\mu}(X_i)\\) is not correctly specified. Still, we will see that doubly robust estimation still manages to recover the same \\(\hat{ATE}\\) of about 0.38 that we've seen before.


```python
from sklearn.linear_model import LogisticRegression, LinearRegression

def doubly_robust_wrong_model(df, X, T, Y):
    np.random.seed(654)
    ps = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    
    # wrong mu(x) model
    mu0 = np.random.normal(0, 1, df.shape[0])
    mu1 = np.random.normal(0, 1, df.shape[0])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )
```


```python
doubly_robust_wrong_model(data_with_categ, X, T, Y)
```


    0.3745648055762825



One again, we can use bootstrap and see that the variance is just slightly higher.


```python
from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
parallel_fn = delayed(doubly_robust_wrong_model)
wrong_mux = Parallel(n_jobs=4)(parallel_fn(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                               for _ in range(bootstrap_sample))
wrong_mux = np.array(wrong_mux)
```


```python
print(f"ATE 95% CI:", (wrong_mux.mean() - 1.96*wrong_mux.std(), wrong_mux.mean() + 1.96*wrong_mux.std()))
```

    ATE 95% CI: (0.34243822810645, 0.4346749772942791)


In practice, what ends up happening is that neither the propensity score nor the outcome model are 100% correct. They are both wrong, but in different ways. Doubly robust estimation can combine those two wrong models to make them less wrong. 

## Keys Ideas

Here, we saw a simple way of combining linear regression with the propensity score to produce a doubly robust estimator. This estimator bears that name because it only requires one of the models to be correct. If the propensity score model is correct, we will be able to identify the causal effect even if the outcome model is wrong. On the flip side, if the outcome model is correct, we will also be able to identify the causal effect even if the propensity score model is wrong.

## References

I like to think of this entire series as a tribute to Joshua Angrist, Alberto Abadie and Christopher Walters for their amazing Econometrics class. Most of the ideas here are taken from their classes at the American Economic Association. Watching them is what is keeping me sane during this tough year of 2020.
* [Cross-Section Econometrics](https://www.aeaweb.org/conference/cont-ed/2017-webcasts)
* [Mastering Mostly Harmless Econometrics](https://www.aeaweb.org/conference/cont-ed/2020-webcasts)

I'll also like to reference the amazing books from Angrist. They have shown me that Econometrics, or 'Metrics as they call it, is not only extremely useful but also profoundly fun.

* [Mostly Harmless Econometrics](https://www.mostlyharmlesseconometrics.com/)
* [Mastering 'Metrics](https://www.masteringmetrics.com/)

My final reference is Miguel Hernan and Jamie Robins' book. It has been my trustworthy companion in the most thorny causal questions I had to answer.

* [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

The data that we used was taken from the article [Estimating Treatment Effects with Causal Forests: An Application](https://arxiv.org/pdf/1902.07409.pdf), by Susan Athey and Stefan Wager. 

The code for this and other tutorials about causality can by found at my [GitHub page](https://github.com/matheusfacure/python-causality-handbook). If you like it, don't forget to leave me a star! 

![img](/img/econ/poetry.png)