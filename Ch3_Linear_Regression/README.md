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
 y  = \hat{\beta_0} + \hat{\beta_1}x
\label{eq:model1}
\tag{3.2}
\end{equation}
$$

 
### 3.1.1 Estimating the Coefficients
### 3.1.2 Assessing the Accuracy of the Coefficient Estimates
### 3.1.3 Assessing the Accuracy of the Model

## 3.2 Multiple Linear Regression 
### 3.2.1 Estimating the Regression Coefficients 
### 3.2.2 Some Important Questions

## 3.3 Other Considerations in the Regression Model 
### 3.3.1 Qualitative Predictors 
### 3.3.2 Extensions of the Linear Model 
### 3.3.3 Potential Problems 
## 3.5 Comparison of Linear Regression with K-Nearest Neighbors 