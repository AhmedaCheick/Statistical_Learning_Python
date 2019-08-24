# Chapter 3: Linear Regression 
---

This chapter is about linear regression, a very simple, yet powerful, approach to supervised learning for predicting a quantitative respone. In this chapter, we review some key ideas underlying the linear regression model, as well as the <strong>least squares</strong> approach, that is commonly used to <strong>fit</strong> a model. 

## 3.1 Simple Linear Regression 
SLR lives up to its name: it's a straightforward approach for predicting a response <strong>$Y$</strong> on the basis of a <strong>single</strong> predictor $X$. We assume there is approximately a linear relationship between the two. Mathematically it's written as: <br>
 $$
\begin{equation}
  Y \approx \beta_0 + \beta_1X
\label{eq:SLR}
\tag{3.1}
\end{equation}
$$
"$\approx$" is read as approximately. We use it because of the error associated to predicting $Y$. <br> We can describe (3.1) by saying that we are regressing $Y$ onto $X$. For example, $X$ may represent TV ads and $Y$ represents sales. <br>
 $$
\begin{equation}
  sales \approx \beta_0 + \beta_1TV
\label{eq:sales}
\
\end{equation}
$$ <br>

In (3.1), $\beta_0$ and $\beta_1$ are two <strong>unkown</strong> constants that represent the *intercept* and *slope* terms in the linear model. They're known as *coefficients* or *parameters*. <br>
We use our **training data** to produce $\hat{\beta_0}$ and $\hat{\beta_1}$ for the **model** coefficients. We use this model to predict future sales based on a particular TV ad by computing: <br>

$$
\begin{equation}
 \widehat{y}  = \hat{\beta_0} + \hat{\beta_1}x
\label{eq:model1}
\tag{3.2}
\end{equation}
$$ 
<br>
Where $\widehat{y}$ indicates a prediction of $Y$ on the basis of $X$ = $x$. The *hat* symbol, $\widehat{}$,  denotes the estimated value for an unknown parameter or coefficient, or the predicted value of the response.

 
### 3.1.1 Estimating the Coefficients
In practice $\beta_0$ and $\beta_1$ are unknown. We therefore must use the data to estimate those two. <br>
Let n observation pairs:
$$(x_1, y_1), (x_2, y_2)...,(x_n,y_n)$$ <br>
The advertising data has n = 200. In other words, for every market in the 200 markets we have a TV budget and the product sales. Our goal here is to obtain the estimate coefficients $\hat{\beta_0}$ and $\hat{\beta_1}$ for a line which is the **closest** to all the 200 data points. Take a look at what that line would look like: 

<p align="center">
  <img width="400" height="300" src=images/ad_data.png>
</p>
<br>
So that for every $i$-*th* observation in the dataset:
$$
\begin{equation}
 y_i  \approx \hat{\beta_0} + \hat{\beta_1}x_i
\label{eq:ith}
\
\end{equation}
$$ 

By far the most common approach to measure closeness invovles minimizing the *least squares* criterion. Other approaches are discussed in Chapter 6. <br>
The fit is found by minimizing the sum of squared errors. Every grey line in the figure is an error and a 'compromise' is reached by averaging their squares. 
Ok, time for some fun! <br>
let $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}x_i$ the prediction of $Y$ based on $i$th value of $X$. Then $e_i = y_i - \hat{y_i}$ which is the *residual* - or in other words, the difference between the actual value and the predicted one usiong our linear model. The *residual sum of squares* (RSS) is then <br>
$$RSS = e^2_1 + e^2_2 + ... + e^2_n$$
Orrrr
$$
RSS = (y_1 - \hat{\beta_0} - \hat{\beta_1}x_1)^2 + (y_2 - \hat{\beta_0} - \hat{\beta_1}x_2)^2 + ... + (y_n - \hat{\beta_0} - \hat{\beta_1}x_n)^2
$$
The *least squares* chooses $\hat{\beta_0}$ and $\hat{\beta_1}$ which minimizes the $RSS$. Using calculus one can prove: <br>

$$\hat{\beta_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
,$$ $\tag{3.4}$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x},$$


### 3.1.2 Assessing the Accuracy of the Coefficient Estimates
In chapter 2 we assumed a true relationship between $X$ and $Y$ in the form $Y = f(X) + \epsilon$ for some unkown $f$, where $\epsilon$ is mean-zero random error term. If we were to approximate $f$ by a linear function then the relationship is
$$Y = \beta_0 + \beta_1X + \epsilon \tag{3.5}$$
Equation (3.5) is the *population regression line* which is the best linear approx to the true relationship. 
The term $\epsilon$ is a catch-all for what we miss with this simple model. Since the true relationship is probably not linear, and there may be other variables that cause variation in $Y$. $\epsilon$ is typically independent of $X$. <br>
At first the difference between the population regression line and the least squares line may be subtle and confusing. In practice we don't have access to the first and we are interested in estimating the latter. <br>
Suppose we are interested in computing the mean $\mu$ of some random variable $Y$. Using the n observations we have access to we can estimate $\mu$. A resonable estimate is $\widehat{\mu}$ = $\bar{y}$ where $\bar{y}$ is the sample mean. The sample will give us a good estimate of the population mean. In the same way, $\hat{\beta_0}$ and $\hat{\beta_1}$ will give us a good estimate to the coefficients $\beta_0$ and $\beta_1$ in the population regression line. <br>
The analogy between linear regression and estimation of the mean is an apt one based on the concept of *bias*. On **avergae** we expect $\widehat{\mu}$ to equal $\mu$. That is to say, it can on occasions underestimate or overestimate the population mean. But a huge number of sample means would *exactly* equal the population mean. We say $\widehat{\mu}$ is an *unbiased* estimator of $\mu$. This property of unbiasedness holds also true for our least squares coefficient estimates. If we could average estimates obtained over a huge number of datasets, then the average would be spot on!
<br>
So, how accurately does a single $\widehat{\mu}$ estimate $\mu$? In general, this is answered by computing the *standard error* of $\widehat{\mu}$. The well-known formula 
$$Var(\widehat{\mu}) = SE(\widehat{\mu})^2 = \frac{\sigma^2}{n}
,\tag{3.7}$$

The standard error reflects the amount by which an estimator varies under repeated sampling. Similarely we can compute:

$$SE(\hat{\beta_0})^2 = {\sigma}^2[\frac{1}{n}+\frac{\bar{x}^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}],$$

$\tag{3.8}$


$$SE(\hat{\beta_1})^2 = \frac{\sigma^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2},$$

$\sigma^2$ is unkown, but can be estimated from the data. The estimate of $\sigma$ is known as the *residual standard error* and is given be the formula
$$RSE = \sqrt{RSS/(n-2)}$$
Strictly speaking, when $\sigma^2$ is estimated we should write $\widehat{SE}(\hat{\beta_1})^2$ to indicate an estimation has taken place but for simplicity we will drop the hat here. 
Standard errors can be used to compute **confidence intervals**. A 95% confidence interval is defined as a range of values such that with 0.95 probability, the range will contain the true value unknown value of the parameter. It has lower and upper limits computed from the sample data. <br> 
For linear regression, the 95% confidence intervals for the values $\beta_1$ and $\beta_0$ take the following forms

$$\hat{\beta_1} \pm 2 \cdot SE(\hat{\beta_1}) \tag{3.9}$$ 

Also can be written
$$[\hat{\beta_1} - 2 \cdot SE(\hat{\beta_1}),  \hat{\beta_1} + 2 \cdot SE(\hat{\beta_1})] \tag{3.10}$$

Similarly
$$\hat{\beta_0} \pm 2 \cdot SE(\hat{\beta_0}) \tag{3.11}$$

Standard errors can also be used to compute *hypothesis tests*.

$H_0$: There is no relationship between $X$ and $Y$ $\tag{3.12}$
*Versus* <br>
$H_\alpha$: There is some relationship

Mathematically this is written
$$H_0: \beta_1 = 0$$
$$H_\alpha: \beta_1 \ne 0$$

To test our hypothesis, we need to determine whether $\hat{\beta_1}$ is sufficiently far from zero that we can be confident that $\beta_1$ is non-zero. Well, how far is enough? This of course depends on the accuracy of $\hat{\beta_1}$!! That is - it depends on $SE(\hat{\beta_1})$. If that standard error is small, a small value of $\hat{\beta_1}$ may provide strong evidence that $\beta_1 \ne 0$. If it's a large number, then $\hat{\beta_1}$ must be large in absolute value to draw the same conclusion. In practice, we compute a *t-statistic*, given by

$$ t = \frac{\hat{\beta_1}-0}{SE(\hat{\beta_1})}
,$$ $\tag{3.14}$

which measures the number of standard deviations that $\hat{\beta_1}$ is away from 0. The test statistic follows a t distribution with (n-2) degrees of freedom, where n is the total number of observations. <br>
The null hypothesis is accepted if the calculated value is such that:
$$-t_{\alpha/2,n-2}<t<t_{\alpha/2,n-2}$$
where $-t_{\alpha/2,n-2}$ and $t_{\alpha/2,n-2}$ are the critical values for the two-sided hypothesis. The t-distribution has a bell shape for values of n greater than approximately 30. <br>$t_{\alpha/2,n-2}$ is the percentile of the t distribution corresponding to a cumulative probability of (1-$\alpha$/2) and $\alpha$ is the significance level. We call this probability the **p-value**. A small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance. Typically the cuttoffs for rejecting the hypothesis are 5 and 1%. When n = 30, these correspond to t-statistics (3.14) of around 2 and 2.75, respectively.



### 3.1.3 Assessing the Accuracy of the Model

It is natural to want to quantify the extent to which the model fits the data. <br>
The quality of a linear regression fit is typically assessed using two related quantities: the *residual standard error* $(RSE)$ and the **$R^2$** statistic.

**Residual Standar Error**
<br>
Even if we knew the true regression line we would not be able to perfectly predict $Y$ from $X$. This is due to the error term $\epsilon$ in equation (3.5), that is associated with every observation. <br>
The **$RSE$** is an estimate of the standard deviation of $\epsilon$. Roughly speaking, it is the average amount that the response will deviate from the true regression line. It is computed using the formula

$$RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum_{i=1}^{n}(y_i - \hat{y_i})^2} \tag{3.15}$$

In the ad example the RSE is 3.26. In other words, actual sales in each market deviate from the true regression line by approximately 3,260 units, on average. <br>
The RSE is considered a measure of the *lack of fit* of the model (3.5).
<br>
<br>
**$R^2$ Statistic**<br>
The RSE provides an absolute measure of lack of fit of the model to the data. But since it is measured in the units of Y , it is not always clear what constitutes a good RSE. The $R^2$ statistic provides an alternative measure of fit. It takes the form of a proportion—the proportion of variance explained—and so it always takes on a value between 0 and 1, and is independent of the scale of Y.

$$R^2 = \frac{TSS}{TSS-RSS} = 1 - \frac{RSS}{TSS} \tag{3.17}$$

Where TSS is the *total sum of squares* $\sum(y_i-\bar{y})^2$. TSS measures the total variance in the response $Y$, and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, $RSS$ measures the amount of variability that is left unexplained after performing the regression. Hence, $TSS - RSS$ measures the amount of variability in the response that is explained (or removed) by performing the regression, and $R^2$ measures the proportion of variability in $Y$ that can be explained using $X$. an $R^2$ that is close to 1 indicates that a large proportion of the variability in the response hs been explained by the regression. It's still challenging to determine what's a *good*  $R^2$ value, and in general, it depends on the application. a value close to 1 might be acceptable in physics and in marketing a value below 0.1 might be realistic! For instance, we may know in the first case that the data come from a linear model, and in the latter the model is just a rough approximation to the data, and residual errors due to other unmeasured factors are often very large. 
<br>
The $R^2$ statistic is a measure of the linear relationship between X and Y . Recall that correlation, defined as

$$Cor(X,Y) = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}} \tag{3.18}$$

is also the measure of the linear relationship between $X$ and $Y$. <br>
$r=Cor(X,Y)$, it can be proven that in a simple linear regression $r^2=R^2$. This is because correlation quantifies the association between only a single pair of variables.


## 3.2 Multiple Linear Regression 
In practice, we often have multiple predictors rather than one. The approach of fitting a seperate simple linear regression model for each is not satisfactory. Each will ignore the others in forming estimates for the coefficients, so we turn to multiple linear regression. We can do this by giving each predictor a separate slope coefficient in a single model. Suppose that we have p distinct predictors. Then the equation takes the form

 $$
\begin{equation}
  Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + .. + \beta_pX_p + \epsilon
\tag{3.19}
\end{equation}
$$


We interpret $\beta_j$ as the average effect on Y of a one unit increase in $X_j$, **holding all other predictors fixed**.
This then becomes in our advertising example as 

$$sales = \beta_0 + \beta_1.TV+ \beta_2.radio+\beta_3.newspaper+\epsilon \tag{3.20}$$



### 3.2.1 Estimating the Regression Coefficients 


Just as we did in the simple linear regression, we need to estimate $\beta_0, \beta_1,..,\beta_p$ <br>

$$\widehat{y}= \hat{\beta_0} + \hat{\beta_1}x_1 + \hat{\beta_2}x_2 + .. + \hat{\beta_p}x_p \tag{3.21}$$
Using the least squares approach we choose  $\beta_0, \beta_1,..,\beta_p$ to minimize the sum of squared residuals
$$RSS = \sum_{i=1}^{n}(y_i - \hat{y_i})^2
=\sum_{i=1}^{n}(y_i-\hat{\beta_0} - \hat{\beta_1}x_{i1} - \hat{\beta_2}x_{i2} - .. - \hat{\beta_p}x_{ip}) \tag{3.22}$$

<p align="center">
  <img width="500" height="300" src=images/multi.png>
</p>



**note**: This a three-dimensional setting, with two predictors $X_1$ $X_2$ and one response $Y$. The surface (plane) is chosen to minimize the sum of the vertical distances between each observation (shown in red) and the plane. <br>

The values $\hat{\beta_0}$, $\hat{\beta_1}$, .., $\hat{\beta_p}$ that minimize (3.22) are the multiple least squares regression coefficient estimates. They are most easily represented using matrix algebra. 

<br>
Table 3.4 displays the multiple regression coefficient estimates when TV, radio, and newspaper advertising budgets are used to predict product sales using the Advertising data.

<p align="center">
  <img width="700" height="90" src=images/table3.4.png>
</p>

An additional 1,000 dollars on radio ads leads to an increase in sales by approximately 189 units. However, **newspaper** coefficient estimate is close to zero with a p-value that is not significant, this isn't the case when we run a simple linear regression against sales. 
<br>

So what happened?! This difference stems from the fact that in the simple regression case, the slope term represents the average effect of a 1,000 dollars increase in newspaper advertising, ignoring other predictors such as TV and radio. In contrast, in the multiple regression setting, the coefficient for newspaper represents the average effect of increasing newspaper spending by 1,000 dollars while holding TV and radio fixed. Now, consider this table 3.5 showing the correlation matrix for the three predictors.

<p align="center">
  <img width="800" height="140" src=images/table3.5.png>
</p>

This reveals a tendency to spend more on newspaper advertising in markets where more is spent on radio advertising (correlation).
Hence, in a simple linear regression which only examines **sales** versus **newspaper**, we will observe that higher values of newspaper tend to be associated with higher values of sales, even though newspaper advertising does not actually affect sales. In other words, newspaper gets "credit" for the effect of **radio** on sales.
<br> 
<br>
Running a regression of shark attacks versus ice cream sales for data collected at a given beach community over a period of time would show a positive relationship, similar to that seen between sales and newspaper. Of course no one (yet) has suggested that ice creams should be banned at beaches to reduce shark attacks. In reality, higher temperatures cause more people to visit the beach, which in turn results in more ice cream sales and more shark attacks. A multiple regression of attacks versus ice cream sales and temperature reveals that, as intuition implies, the former predictor is no longer significant after adjusting for temperature.

### 3.2.2 Some Important Questions
1. Is at least one of the predictors $X_1$ , $X_2$ , . . . , $X_p$ useful in predicting the response?
2. Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?
3. How well does the model fit the data?
4. Given a set of predictor values, what response value should we predict,
and how accurate is our prediction?

## 3.3 Other Considerations in the Regression Model 
### 3.3.1 Qualitative Predictors 
### 3.3.2 Extensions of the Linear Model 
### 3.3.3 Potential Problems 
## 3.5 Comparison of Linear Regression with K-Nearest Neighbors  