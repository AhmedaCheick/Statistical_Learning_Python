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
$$(x_1, y1_1), (x_2, y_2)...,(x_n,y_n)$$ <br>
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
The term $\epsilon$ is a catch-all for what we miss with this simple model. Since the true relationship is probably not linear, and there may be other variables that cause variation in $Y$. $\epsilon$ is typically independent of $X$.
### 3.1.3 Assessing the Accuracy of the Model

## 3.2 Multiple Linear Regression 
### 3.2.1 Estimating the Regression Coefficients 
### 3.2.2 Some Important Questions

## 3.3 Other Considerations in the Regression Model 
### 3.3.1 Qualitative Predictors 
### 3.3.2 Extensions of the Linear Model 
### 3.3.3 Potential Problems 
## 3.5 Comparison of Linear Regression with K-Nearest Neighbors 