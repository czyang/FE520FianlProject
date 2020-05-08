# FE520FianlProject
FE520 Fianl Project

## Topic
Build a regression package (Using Class)
1. A linear regression
2. non linear regression
3. Output statistic value for regression (p value, t-test, F-test, R square, AIC...)

## Usage

### Unit test
There have two way to run unit test :
1. Using IDE(Pycharm) open project and directly run the `tests/*.py`
2. Using command line python:
```cmd
python -m unittest tests/test_linear.py
```

## FE520 Group 1 Project Proposal
Regression Package Implementation and Test
Group Member:
	Chengzhi Yang
	Yangyang Liu
	Xinzhe Li

1. Topic: Regression Package Implementation and Test

2. Motivation and insights expect from data analysis:
In this project, we are focusing on implementing regression class and statistic tool building. As regression is a powerful tool in data analysis, we apply it to solve real problems.

We find a housing data set. House attributes include floors, bedroom number, living space, location, year of built, housing price and so on. We have already divided it into train set and test set. Our model will be tested. And it will show us how these house attributes impact on housing price. 

3.Related Work:
Pattern Recognition and Machine Learning. ChristopherBishop, Springer, 2006 as algorithm implementation guide.

4.Methodology:
There are three ways to find linear regression fitting functions. Gradient Descent, Newton’s Law and Normal Equation. The class will mainly rely on Gradient Descent. And we will try other methods if time is sufficient.

5.Project Plan:
The entire project will be divided into three parts, including linear regression class implementation, no-linear regression class implementation and statistic tools. Those three parts will be distributed to our group members.Classes and tools will be tested against data set. Essential meta data and results will be represented graphically.


### Feedback
This is easy. please refer https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html I expect you can calculate all thing similar with summary table.
Dan Wang, Apr 18 at 2:45pm

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 4.020e+06
Date:                Fri, 13 Mar 2020   Prob (F-statistic):          2.83e-239
Time:                        13:54:01   Log-Likelihood:                -146.51
No. Observations:                 100   AIC:                             299.0
Df Residuals:                      97   BIC:                             306.8
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.3423      0.313      4.292      0.000       0.722       1.963
x1            -0.0402      0.145     -0.278      0.781      -0.327       0.247
x2            10.0103      0.014    715.745      0.000       9.982      10.038
==============================================================================
Omnibus:                        2.042   Durbin-Watson:                   2.274
Prob(Omnibus):                  0.360   Jarque-Bera (JB):                1.875
Skew:                           0.234   Prob(JB):                        0.392
Kurtosis:                       2.519   Cond. No.                         144.
==============================================================================
```

### Structure
```
.
├── regression
│   ├── __init__.py
│   ├── linear.py
│   ├── nonlinear.py
│   ├── stats.py
│   └── summary.py
├── tests
│   ├── __init__.py
│   ├── test_linear.py
│   ├── test_nonlineartest.py
│   ├── test_stats.py
│   └── test_summary.py
├── project.md
├── README.md
├── test.csv
└── train.csv
```

### Reference
https://cs.stanford.edu/~ermon/cs325/slides/ml_nonlin_reg.pdf

https://github.com/statsmodels/statsmodels

https://rickwierenga.com/blog/ml-fundamentals/polynomial-regression.html

https://docs.python-guide.org/writing/structure/
