---
layout: tutorial
tags: [Econ]
comments: true
title: 7 - Beyond Confounders
subtitle: Good and bad controls
date: 2020-07-06
true-dt: 2020-07-06
author: "Matheus Facure"
---

## Good Controls

We've seen how adding additional controls to our regression model can help identify causal effect. If the control is a confounder, adding it to the model is not just nice to have, but is a requirement. When the unwary see this, a natural response is to throw whatever he can measure into the model. In today's world of big data, this could easily be more than 1000 variables. As it turns out, this is not only unnecessary, but can be detrimental to causal identification. Now, turn our attention to controls that are not confounders. First, let's take a look at the good ones. Then, we will delve into the harmful controls.

As a motivating example, let's suppose you are a data scientist in the collections team of a fintech. Your next task is to figure out the impact of sending an email asking people to negotiate their debt. Your response variable is the amount of payments from the late customers.

To answer this question, your team selects 5000 random customers from your late customers base to do a random test. For every customer, you flip a coin, if its heads, the customer receives the email; otherwise, it is left as a control. With this, you hope to find out how much extra money the email brings to the team.


```python
data = pd.read_csv("./data/collections_email.csv")
data.head()
```


<div>
<table class="table table-bordered table-striped table-hover" border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>payments</th>
      <th>email</th>
      <th>opened</th>
      <th>agreement</th>
      <th>credit_limit</th>
      <th>risk_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>740</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2348.495260</td>
      <td>0.666752</td>
    </tr>
    <tr>
      <th>1</th>
      <td>580</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>334.111969</td>
      <td>0.207395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>600</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1360.660722</td>
      <td>0.550479</td>
    </tr>
    <tr>
      <th>3</th>
      <td>770</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1531.828576</td>
      <td>0.560488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>660</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>979.855647</td>
      <td>0.455140</td>
    </tr>
  </tbody>
</table>
</div>



Since the data is random, you know that a simple difference in means estimates the Average Treatment Effect. In other words, nothing can have caused the treatment but the randomisation, so the potential outcomes are independent of the treatment: \\((Y_0, Y_1)\perp T\\). 

$$
ATE = E[Y_i|T_i=1] - E[Y_i|T_i=1]
$$

Since you are smart and want to place a confidence interval around your estimate, you use a linear regression.


```python
print("Difference in means:",
      data.query("email==1")["payments"].mean() - data.query("email==0")["payments"].mean())

model = smf.ols('payments ~ email', data=data).fit()
model.summary().tables[1]
```

    Difference in means: -0.6202804021329484


<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  669.9764</td> <td>    2.061</td> <td>  325.116</td> <td> 0.000</td> <td>  665.937</td> <td>  674.016</td>
</tr>
<tr>
  <th>email</th>     <td>   -0.6203</td> <td>    2.941</td> <td>   -0.211</td> <td> 0.833</td> <td>   -6.387</td> <td>    5.146</td>
</tr>
</table>


Sadly, the estimated ATE is -0.62, which is pretty weird. How can sending an email make late customers pay less than average? Still, the P-value is so high that this probably doesn't mean anything. What you should do now? Go back to your team with a tail between your legs and say that the test is inconclusive and you need more data? Not so fast.

Notice how your data has some other interesting columns. For example, `credit_limit` represents the customer's credit line prior to he or she getting late. `risk_score` corresponds to the estimated risk of the customer prior to the delivery of the email. It makes sense to think that credit limit and risk are probably very good predictors of payments. What is still not clear is how is that useful. 

First, let's understand why we can fail to find statistical significance in a treatment even when it is there. It could be that, like in this case, the treatment has very little impact on the outcome. If you think about it, what makes people pay their debt is, in the majority of it, factors outside the control of the collections department. They relate to finding a new job, managing one's finances, income and so on. In statistical terms, we can say that **the variability of payments is explained much more by other factors other than by the email**. 

To get a visual understanding of it, we can plot the payments against the treatment variable email. I've also plotted the fitted line of the model above in red. To help visualization, I've added a little bit of noise to the email variable so that it doesn't get smashed at the zero or one. 


![png](/img/econ/beyond-confounders/output_6_0.png)


We can see how wildly payments vary in a single treatment group. Visually, it looks like it is going from a little bit under 400 to 1000 in both groups. If the impact of the email in the the order of say 5.00 or 10.00 R$, it is no wonder it will be hard to find inside all the variability.

Fortunately, regression can help us lower this variability. The trick is to use additional controls. **If a variable is a good predictor of the outcome, it will explain away lot of its variance**. If risk and credit limit are good predictors of payment, we can control from them to make it easier to find the impact of the email on payments. If we remember how regression works, this has an intuitive explanation. Adding extra variables to a regression means keeping them constant while looking at the treatment. So, the reasoning goes, if we look at similar levels of risk and credit limit, the variance of the responde variable payments should be smaller. Or, in other words, if risk and credit line predict payments very well, customers with similar risk and credit line should also have similar payment levels, hence with less variance.

![img](/img/econ/beyond-confounders/y-pred.png)

To demonstrate this, let's resort to the partialling out way of breaking regression into 2 steps. First, we will regress the treatment, email, on and the outcome, payments, on the additional controls, credit limit and risk score. Second, we will regress the residual of the treatment on the residuals of payments, both obtained in step 1. (This is purely pedagogical, in practice you won't need to go through all the hassle).


```python
model_email = smf.ols('email ~ credit_limit + risk_score', data=data).fit()
model_payments = smf.ols('payments ~ credit_limit + risk_score', data=data).fit()

residuals = pd.DataFrame(dict(res_payments=model_payments.resid, res_email=model_email.resid))

model_treatment = smf.ols('res_payments ~ res_email', data=residuals).fit()
```

What this does, is, in essence, lower the variance of the dependent variable. By regressing payments on credit limit and risk and obtaining the residuals for this model, we are creating a new dependent variable with much less variability than the original one. The last model also uncovers the `ATE` with valid standard error estimate.

Just out of curiosity, we can also check that the model that predict email should not be able to lower the variance of it. That's because email is, by design, random, so nothing can predict it.


```python
print("Payments Variance", np.var(data["payments"]))
print("Payments Residual Variance", np.var(residuals["res_payments"]))

print("Email Variance", np.var(data["email"]))
print("Email Residual Variance", np.var(residuals["res_email"]))

model_treatment.summary().tables[1]
```

    Payments Variance 10807.612416
    Payments Residual Variance 5652.453558466207
    Email Variance 0.24991536
    Email Residual Variance 0.24918421069820032


<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> 4.143e-13</td> <td>    1.063</td> <td>  3.9e-13</td> <td> 1.000</td> <td>   -2.084</td> <td>    2.084</td>
</tr>
<tr>
  <th>res_email</th> <td>    4.4304</td> <td>    2.129</td> <td>    2.080</td> <td> 0.038</td> <td>    0.256</td> <td>    8.605</td>
</tr>
</table>



Notice how the variance of payments went from 10807 to 5652. We've decreased it by almost half once we control for risk and credit limits. Also notice that we didn't manage to reduce the variability of the treatment email. This makes sense, since risk and credit line does not predict email (nothing does, by definition of randomness).

Now, we see something much more reasonable. This new estimate tells us that we should expect customers that received the email to pay, on average, 4.4 reais more than those in the control group. This estimate is now statistically different from zero. We can also visualize how the variance is now lower within each control group.


![png](/img/econ/beyond-confounders/output_12_0.png)


As I've said, we did this for pedagogical reasons. In practice, you can simply add the controls to the regression model together with the treatment and the estimates will be exactly the same.


```python
model_2 = smf.ols('payments ~ email + credit_limit + risk_score', data=data).fit()
model_2.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  490.8653</td> <td>    9.715</td> <td>   50.527</td> <td> 0.000</td> <td>  471.820</td> <td>  509.911</td>
</tr>
<tr>
  <th>email</th>        <td>    4.4304</td> <td>    2.130</td> <td>    2.080</td> <td> 0.038</td> <td>    0.255</td> <td>    8.606</td>
</tr>
<tr>
  <th>credit_limit</th> <td>    0.1511</td> <td>    0.008</td> <td>   18.833</td> <td> 0.000</td> <td>    0.135</td> <td>    0.167</td>
</tr>
<tr>
  <th>risk_score</th>   <td>   -8.0516</td> <td>   38.424</td> <td>   -0.210</td> <td> 0.834</td> <td>  -83.379</td> <td>   67.276</td>
</tr>
</table>



To wrap it up, anytime we have a control that is a good predictor of the outcome, even if it is not a confounder, adding it to our model is a good idea. It helps lowering the variance of our treatment effect estimates. Here is a picture of what this situation looks like with causal graphs.


![svg](/img/econ/beyond-confounders/output_16_0.svg)



## Mostly Harmful Controls

As a second motivating example, let's consider a drug test scenario with 2 hospitals. Both of them are conducting randomised trials on a new drug to treat a certain illness. The outcome of interest is days hospitalised. If the treatment is effective, it will lower the amount of days the patient stays in the hospital. For one of the hospitals, the policy regarding the random treatment is to give it to 90% of its patients while 10% get a placebo. The other hospital has a different policy: it gives the drug to a random 10% of its patients and 90% get a placebo. You are also told that the hospital that gives 90% of the true drug and 10% of placebo usually gets more severe cases of the illness to treat. 


```python
hospital = pd.read_csv("./data/hospital_treatment.csv")
hospital.head()
```

<div>
<table class="table table-bordered table-striped table-hover" border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hospital</th>
      <th>treatment</th>
      <th>severity</th>
      <th>days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>29.686618</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>20.050340</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>20.302399</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>10.603118</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>8.332793</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



Since you are dealing with randomized data, your first instinct is to simply run a regression of the outcome on the treatment.


```python
hosp_1 = smf.ols('days ~ treatment', data=hospital).fit()
hosp_1.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   33.2667</td> <td>    2.662</td> <td>   12.498</td> <td> 0.000</td> <td>   27.968</td> <td>   38.566</td>
</tr>
<tr>
  <th>treatment</th> <td>   14.1533</td> <td>    3.367</td> <td>    4.204</td> <td> 0.000</td> <td>    7.451</td> <td>   20.856</td>
</tr>
</table>



But you find some counterintuitive results. How can the treatment be increasing the number of days in the hospital? The answer lies in the fact that we are running 2 different experiments. Severity is positively linked with more days at the hospital and since the hospital with more severe cases also gives more of the drug, the drug becomes positively correlated with more days at the hospital. When we look at both hospital together, we have that \\(E[Y_0\|T=0]>E[Y_0\|T=1]\\), that is, the potential outcome of the untreated is, on average, higher than that of the treated because there are more untreated in the hospital with less severe cases. In other words, severity acts as a confounder, determining the hospital the patient goes and, hence, the probability of receiving the drug. 

There are 2 ways of fixing that. The first one, which defeats the purpose of using data from both hospitals, is to simply look at the ATE in each hospital individually.


```python
hosp_2 = smf.ols('days ~ treatment', data=hospital.query("hospital==0")).fit()
hosp_2.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   30.4074</td> <td>    2.868</td> <td>   10.602</td> <td> 0.000</td> <td>   24.523</td> <td>   36.292</td>
</tr>
<tr>
  <th>treatment</th> <td>  -11.4074</td> <td>   10.921</td> <td>   -1.045</td> <td> 0.306</td> <td>  -33.816</td> <td>   11.001</td>
</tr>
</table>


```python
hosp_3 = smf.ols('days ~ treatment', data=hospital.query("hospital==1")).fit()
hosp_3.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   59.0000</td> <td>    6.747</td> <td>    8.745</td> <td> 0.000</td> <td>   45.442</td> <td>   72.558</td>
</tr>
<tr>
  <th>treatment</th> <td>  -10.3958</td> <td>    6.955</td> <td>   -1.495</td> <td> 0.141</td> <td>  -24.371</td> <td>    3.580</td>
</tr>
</table>



In this case, we did get an intuitive result of the ATE. It looks like now the drug is in fact lowering the amount of days at the hospital. However, since we are looking at each hospital individually, there are not enough data points. As a consequence, we are unable to find statistically significant results.

The other approach, which leverages the power of regression, is to control for severity by including it in the model.


```python
hosp_4 = smf.ols('days ~ treatment + severity', data=hospital).fit()
hosp_4.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   11.6641</td> <td>    2.000</td> <td>    5.832</td> <td> 0.000</td> <td>    7.681</td> <td>   15.647</td>
</tr>
<tr>
  <th>treatment</th> <td>   -7.5912</td> <td>    2.269</td> <td>   -3.345</td> <td> 0.001</td> <td>  -12.110</td> <td>   -3.073</td>
</tr>
<tr>
  <th>severity</th>  <td>    2.2741</td> <td>    0.154</td> <td>   14.793</td> <td> 0.000</td> <td>    1.968</td> <td>    2.580</td>
</tr>
</table>



The question that arises next is, should we also include hospital in the model? After all, we know that hospitals cause the treatment right? Well, that is true, but once we've controlled for severity, hospital is no longer correlated with the outcome number of days hospitalised. And we know that to be a confounder a variable has to cause both the treatment and the outcome. In this case, we have a variable that only causes the treatment.

But maybe controlling for it lowers the variance, right? Well, not true again. In order for a control to lower the variance, it has to be a good predictor of the outcome, not of the treatment, which is the case here.

Still, we might want to control it right? It can't hurt... Or can it?


```python
hosp_5 = smf.ols('days ~ treatment + severity + hospital', data=hospital).fit()
hosp_5.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   11.0111</td> <td>    2.118</td> <td>    5.198</td> <td> 0.000</td> <td>    6.792</td> <td>   15.230</td>
</tr>
<tr>
  <th>treatment</th> <td>   -5.0945</td> <td>    3.492</td> <td>   -1.459</td> <td> 0.149</td> <td>  -12.049</td> <td>    1.861</td>
</tr>
<tr>
  <th>severity</th>  <td>    2.3865</td> <td>    0.195</td> <td>   12.251</td> <td> 0.000</td> <td>    1.999</td> <td>    2.774</td>
</tr>
<tr>
  <th>hospital</th>  <td>   -4.1535</td> <td>    4.413</td> <td>   -0.941</td> <td> 0.350</td> <td>  -12.943</td> <td>    4.636</td>
</tr>
</table>



Surprisingly, it can hurt! 

![img](/img/econ/beyond-confounders/shocked.png)

Adding hospital on top of severity as a control induced MORE variance to our ATE estimator. How can that be? The answer lies in the formula for the standard error of the regression coefficient.

$$
\hat{\sigma}^2 = \dfrac{1}{n-2} \sum( y_i - \hat{y}_i )^2
$$

$$
\text{Var}(\hat{\beta}_2) = \dfrac{\sigma^2}{\sum(x_i - \bar{x})^2}
$$

From this formula, we can see that the standard error is inversely proportional to the variance of the variable \\(X\\). This means that, if \\(X\\) doesn't change much, it will be hard to estimate its effect on the outcome. This also makes intuitive sense. Take it to the extreme and pretend you want to estimate the effect of a drug, so you conduct a test with 10000 individuals but only 1 of them get the treatment. This will make finding the ATE very hard, we will have to rely on comparing a single individual with everyone else. Another way to say this is that we need lots of variability in the treatment to make it easier to find its impact. 

As to why including hospitals in the model increases the error of our estimate, it is because it is a good predictor of the treatment and not of the outcome (once we control for severity). So, by predicting the treatment, it effectively makes it so that it's variance is lower! Once again, we can resort to partitioning our regression above into it's 2 steps to see this.


```python
model_treatment = smf.ols('treatment ~ severity + hospital', data=hospital).fit()
model_days = smf.ols('days ~ severity + hospital', data=hospital).fit()

residuals = pd.DataFrame(dict(res_days=model_days.resid, res_treatment=model_treatment.resid))

model_treatment = smf.ols('res_days ~ res_treatment', data=residuals).fit()

model_treatment.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td> 2.498e-14</td> <td>    0.827</td> <td> 3.02e-14</td> <td> 1.000</td> <td>   -1.646</td> <td>    1.646</td>
</tr>
<tr>
  <th>res_treatment</th> <td>   -5.0945</td> <td>    3.447</td> <td>   -1.478</td> <td> 0.143</td> <td>  -11.957</td> <td>    1.768</td>
</tr>
</table>




```python
print("Treatment Variance", np.var(hospital["treatment"]))
print("Treatment Residual Variance", np.var(residuals["res_treatment"]))
```

    Treatment Variance 0.234375
    Treatment Residual Variance 0.05752909187211906


Also, don't take my word for it! You can check that the SE formula above is true:


```python
sigma_hat = sum(model_treatment.resid**2)/(len(model_treatment.resid)-2)
var = sigma_hat/sum((residuals["res_treatment"] - residuals["res_treatment"].mean())**2)
print("SE of the Coeficient:", np.sqrt(var))
```

    SE of the Coeficient: 3.4469737674869028


So the bottom line is that we should add controls that are both correlated with the treatment and the outcome (confounder), like the severity in the model above. We should also add controls that are good predictors of the outcome, even if they are not confounders, because they lower the variance of our estimates. However, we should **NOT** add controls that are just good predictors of the treatment, because they will increase the variance of our estimates.

Here is a picture of what this situation looks like with causal graphs.


![svg](/img/econ/beyond-confounders/output_34_0.svg)

## Bad Controls - Selection Bias

Let's go back to the collections email example. Remember that the email was randomly assigned to customers. We've already explained what `credit_limit` and `risk_score` is. Now, let's look at the remaining variables: `opened`. This is a dummy variable for the customer opening the email or not and `agreement` is another dummy marking if the customers contacted the collections department to negotiate their debt, after having received the email. Which of the following models do you think is more appropriate? The first is a model with the treatment variable plus `credit_limit` and `risk_score`; the second adds `opened` and `agreement` dummies.


```python
email_1 = smf.ols('payments ~ email + credit_limit + risk_score', data=data).fit()
email_1.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  490.8653</td> <td>    9.715</td> <td>   50.527</td> <td> 0.000</td> <td>  471.820</td> <td>  509.911</td>
</tr>
<tr>
  <th>email</th>        <td>    4.4304</td> <td>    2.130</td> <td>    2.080</td> <td> 0.038</td> <td>    0.255</td> <td>    8.606</td>
</tr>
<tr>
  <th>credit_limit</th> <td>    0.1511</td> <td>    0.008</td> <td>   18.833</td> <td> 0.000</td> <td>    0.135</td> <td>    0.167</td>
</tr>
<tr>
  <th>risk_score</th>   <td>   -8.0516</td> <td>   38.424</td> <td>   -0.210</td> <td> 0.834</td> <td>  -83.379</td> <td>   67.276</td>
</tr>
</table>


```python
email_2 = smf.ols('payments ~ email + credit_limit + risk_score + opened + agreement', data=data).fit()
email_2.summary().tables[1]
```

<table class="table table-bordered table-striped table-hover" border="1">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  488.4416</td> <td>    9.716</td> <td>   50.272</td> <td> 0.000</td> <td>  469.394</td> <td>  507.489</td>
</tr>
<tr>
  <th>email</th>        <td>   -1.6095</td> <td>    2.724</td> <td>   -0.591</td> <td> 0.555</td> <td>   -6.949</td> <td>    3.730</td>
</tr>
<tr>
  <th>credit_limit</th> <td>    0.1507</td> <td>    0.008</td> <td>   18.809</td> <td> 0.000</td> <td>    0.135</td> <td>    0.166</td>
</tr>
<tr>
  <th>risk_score</th>   <td>   -2.0929</td> <td>   38.375</td> <td>   -0.055</td> <td> 0.957</td> <td>  -77.325</td> <td>   73.139</td>
</tr>
<tr>
  <th>opened</th>       <td>    3.9808</td> <td>    3.914</td> <td>    1.017</td> <td> 0.309</td> <td>   -3.692</td> <td>   11.654</td>
</tr>
<tr>
  <th>agreement</th>    <td>   11.7093</td> <td>    4.166</td> <td>    2.811</td> <td> 0.005</td> <td>    3.542</td> <td>   19.876</td>
</tr>
</table>


While the first model finds statistically significant results for the email, the second one does not. But maybe the second one is the right model and there's no effect for the email. After all, this model controls for more factors, so it should be more robust right? By now you probably know that this is not the case. What is left is to figure out what is.

We know that we MUST add confounder variables, variables that cause both the treatment and the outcome. We also know that it is a good idea to add controls that predict the outcome very well. This is not required, but it's nice to have. We also know that it is a bad idea to add control that predicts only the treatment. Agin, this is not a deadly sin, but is nice to avoid. So what kind of controls are `opened` and `agreement`? Turns out, they are neither of the above.

If you think about it `opened` and `agreement` are surely correlated with the email, after all, you can't open the email if you didn't receive it and we've also said that the agreement only considers renegotiation that happened after the email has been sent. But **they don't cause email! Instead, they are caused by it!** 

Whenever I need to understand what kind of variables I'm dealing with, I always like to think about their causal graph. Let's do these here


![svg](/img/econ/beyond-confounders/output_39_0.svg)



So, we know nothing causes email, because it's random by design. And we know (or at least we have strong reasons to believe) that credit limit and risk cause payments. We also think that email causes payments. As for opened, we think that it does cause payments. Intuitively, people that opened the collection email are more willing to negotiate and pay their debt. We also think that opened causes agreements for the same reasons as it causes payments. Moreover, we know opened is caused by email and we have reasons to believe people with different risk and credit limits have different open rates for the emails, so credit limit and risk also causes opened. As for agreement, we also think that it is caused by opened. If we think about the payments response variable, we can think of is as the result of a funnel:

$$
email -> opened -> agreement -> payment 
$$

We also think that different levels of risk and line have different propensity of doing an agreement, so we will mark them also causing agreement. As for email and agreement, we could make an argument that some people just read the subject and make more agreement, so it could also cause agreement without passing through opened.

What we notice with this graph is that opened and agreement are both in the causal path from email to payments. So, if we control for them with regression, we would be saying "this is the effect of email while keeping opened and agreement fixed". However, both are part of the causal effect of the email, so we don't want to hold them fixed. Instead, we could argue that email increases payments precisely because it boosts the agreement rate. If we fix those variable, we are removing some of the true effect from the email variable. 

With potential outcome notation, we can say that, due to randomization \\(E[Y_0\|T=0] = E[Y_0\|T=1]\\). However, even with randomization, when we control for agreement, treatment and control are no longer comparable. In fact, with some intuitive thinking, we can even guess how they are different:


$$
E[Y_0|T=0, Agreement=0] > E[Y_0|T=1, Agreement=0]
$$

$$
E[Y_0|T=0, Agreement=1] > E[Y_0|T=1, Agreement=1]
$$

The first equation makes it explicit that we think those without the email and the agreement are better than those with the email and without the agreement. That is because, if the treatment has a positive effect, those that didn't make and agreement **even with after having received the email** are probably worse in terms of payments compared to those that also didn't do the agreement but also didn't get the extra incentive of the email. As for the second equation, those that did the agreement even without having received the treatment are probably better than those that did the agreement but had the extra incentive of the email. 

This might be very confusing the first time you read it (it was for me), but make sure you understand it. Read it again if necessary. Then, a similar kind of reasoning can be done with the opened variable. Try to make it yourself.

This sort of bias is so pervasive it has its own name. While confounding is the bias from failing to control for a common cause, **selection bias is when we control for a common effect or a variable in between the path from cause to effect.** As a rule of thumb, always include confounders and variables that are good predictors of \\(Y\\) in your model. Always exclude variables that are good predictors of only \\(T\\), mediators between the treatment and outcome or common causes of the treatment and outcome.

![img](/img/econ/beyond-confounders/selection.png)

Selection bias is so pervasive that not even randomization can fix it, or better yet, it is often introduced by the ill advised even in random data. Spotting and avoiding selection bias requires more practice than skill. Often, they appear underneath some supposedly clever idea, making it even harder to uncover. Here are some examples of selection biased I've encountered:

1. Adding a dummy for paying the entire debt when trying to estimate the effect of a collections strategy on payments.
2. Controlling for white vs blue collar jobs when trying to estimate the effect of schooling on earnings
3. Controlling for conversion when estimating the impact of interest rates on loan duration
4. Controlling for marital happiness when estimating the impact of children on extramarital affairs
5. Breaking up payments modeling E[Payments] into one binary model that predict if payment will happen and another model that predict how much payment will happen given that some will: E[Payments\|Payments>0]*P(Payments>0)
    
What is notable about all these ideas is how reasonable they sound. Selection bias often does. Let this be a warning. As a matter of fact, I myself have fallen into the traps above many many times before I learned they were bad. One in particular, the last one, deserves further explanation because it looks so clever and catches lots of data scientists off guard. It's so pervasive that it has its own name: **The Bad COP**!

### Bad COP

The situation goes like this. You have a continuous variable that you want to predict but its distribution is over represented at zero. For instance, if you want to model consumer spending, you will have something like a gamma distribution, but with lots of zeros.

![png](/img/econ/beyond-confounders/output_41_0.png)


When a data scientist sees this, the first idea that pops into his head is to break up modeling into 2 steps. The first is the participation, that is, the probability that \\(Y > 0\\). In our spend example, this would be modeling if the customer decided to spend or not. The second part models \\(Y\\) for those that decided to participate. It is the Conditional-on-Positives effect. In our case, this would be how much the customer spends after he or she decided they would spend anything. If we would like to estimate the effect of the treatment \\(T\\) on expenditures, it would look something like this under the two model approach

$$
E[Y_i|T_i] = E[Y_i|Y_i>0, T_i]P(Y_i>0|T_i)
$$


There is nothing wrong with the participation model \\(P(Y_i>0\|T_i)\\). In fact, if \\(T\\) is randomly assigned, it will capture the increase in probability of spending due to the treatment. The issue is with the COP part. **It will be biased even under random assignment**:

$$
\begin{align} 
E[Y_i|Y_i>0, T_i]&=E[Y_i|Y_i>0, T_i=1]-E[Y_i|Y_i>0, T_i=0] \\
&=E[Y_{i1}|Y_{i1}>0]-E[Y_{i0}|Y_{i0}>0] \\
&=\underbrace{E[Y_{i1} - Y_{i0}|Y_{i1}>0]}_{Causal \ Effect} + \underbrace{\{ E[Y_{i0}|Y_{i1}>0] - E[Y_{i0}|Y_{i0}>0] \}}_{Selection \ Bias}
\end{align}
$$ 

When we break up the COP effect, we get first the causal effect on the participant subpopulation. In our example, this would be the causal effect on those that decide to spend something. Second, we get a bias term which is the difference in \\(Y_0\\) for those that decide to participate with the treatment (\\(E[Y_{i0}\|Y_{i1}>0]\\)) and those that that participate even without the treatment (\\(E[Y_{i0}\|Y_{i0}>0]\\)). In our case, this bias is probably negative, since those that spend with the treatment, have they not gottend the treatment, would probably spend less than those that spend even without the treatment \\(E[Y_{i0}\|Y_{i1}>0] < E[Y_{i0}\|Y_{i0}>0]\\).

![img](/img/econ/beyond-confounders/cop.png)

To wrap up selection bias, we need to always remind ourselves to never control for a variable that is either in between the treatment and the outcome or is a common cause of the outcome and the treated. In graphical language, here is what bad control looks like:

<img class="img-responsive center-block" src="/img/econ/beyond-confounders/output_43_0.svg" style="width: 20%;" alt=""/>

## Key Ideas

In this section, we've looked at variables that are not confounders and if we should add them or not to our model for causal identification. We've seen that variables that are good predictors of the outcome \\(y\\) should be added to the model even if they don't predict \\(T\\) (are not confounders). This is because predicting \\(Y\\) lowers its variance and makes it more likely that we will see statistically significant results when estimating the causal effect. Next, we've seen that it is a bad idea to add variables that predict the treatment but not the outcome. Those variables reduce the variability of the treatment, making it harder for us to find the causal effect. Finally, we've looked at selection bias. This is bias that arises when we control for variables in the causal path from the treatment to the outcome or variables that are common causes of the treatment and the outcome.


## References

I like to think of this entire series as a tribute to Joshua Angrist, Alberto Abadie and Christopher Walters for their amazing Econometrics class. Most of the ideas here are taken from their classes at the American Economic Association. Watching them is what is keeping me sane during this tough year of 2020.
* [Cross-Section Econometrics](https://www.aeaweb.org/conference/cont-ed/2017-webcasts)
* [Mastering Mostly Harmless Econometrics](https://www.aeaweb.org/conference/cont-ed/2020-webcasts)

I'll also like to reference the amazing books from Angrist. They have shown me that Econometrics, or 'Metrics as they call it, is not only extremely useful but also profoundly fun.

* [Mostly Harmless Econometrics](https://www.mostlyharmlesseconometrics.com/)
* [Mastering 'Metrics](https://www.masteringmetrics.com/)

My final reference is Miguel Hernan and Jamie Robins' book. It has been my trustworthy companion in the most thorny causal questions I had to answer.

* [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

The code for this and other tutorials about causality can by found at my [GitHub page](https://github.com/matheusfacure/python-causality-handbook). If you like it, don't forget to leave me a star! 

![img](/img/econ/poetry.png)