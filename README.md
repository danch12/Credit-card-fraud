# Credit-card-fraud


## Packages used

* Pandas
* Scikit-learn
* Matplotlib
* Numpy
* itertools
* scipy
* Light GBM


## Introduction

I wanted to do a small project looking at credit card fraud. I thought it would be quite an interesting project and could also have real business impact. The dataset I used was from an old kaggle [competition](https://www.kaggle.com/c/ieee-fraud-detection/overview) that had finished a while ago. It had a load of samples and features so thought it would lead to some interesting results and also I wanted to look into learning how to use Light GBM which needs quite a lot of data. The bulk of this readme is going to be dedicated to the differences between ada boost and gradient descent algorithms as this project helped solidify this concept to me and therefore I thought it would be beneficial to lay it out in a project. 

## Feature Engineering

As most of the features were at least semi veiled I took a wide aproach to feature engineering and tried to think of as many feature interactions as possible for fear of missing some. Specifically I was looking at the Santander kaggle competition for some inspiration and saw that many were using counts of unique values in order to get an edge so I also implemented this into my dataset to see if the same was true.

## EDA

I mention this in the notebook but I read an interesting [article](https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb) on EDA and imbalanced datasets, specifically resampling the dataset before doing a correlation heatmap so variables correlated with the target variable are easier to see. From doing this we can see that there are quite a few variables that are somewhat correlated with the target variable. The other interesting part of the EDA was the 'C' variables interaction with TransactionDT, it seems as though there is some seasonal aspect to the data but I am unsure what exactly it is.

## Comparison of gradient boosting vs ada boosting

As alluded to in the introduction I wanted to talk about how ada boost and graident boost work, then how light GBM is different to the usual gradient boost algorithm.

### How ada boost works

In essence ada boost uses a load of weak learners (usually decision trees with few levels) to slowly step towards the correct prediction. The errors in previous stumps how the future stumps are made. 

First you start with some data and give each sample in the dataset a weight which indicates how important it is that we classify this sample correctly. At the start all samples get the same weight - 1/ number of samples. However after the first stump these weights change to indicate how future stumps should be created. To make the first stump you choose the variable that does the best job of classifying the samples(in this case I am talking about ada boost with max depth of 1, this would be different for other depths) this is done using the Gini index. 

Gini index measures the degree or probability of a particular variable being wrongly classified when it is randomly chosen. The degree of Gini index varies between 0 and 1 where 0 denotes that all elements belong to a certain class and 1 denotes that all elements are randomly distributed across various classes. Can be calculated by subtracting the sum of squared probabilities from 1. Probabilities meaning the probability that an object is classified to a particular class.

We determine how much say a stump gets in the final classification depending on how well it predicted the classes. The total error for a stump is the sum of the weights associated with the incorrectly classified samples. We then use the total error to determine the amount of say this stump has in the final classification with the following formula-

![equation](http://www.sciweavers.org/upload/Tex2Img_1578306086/render.png)

When a stump does a good job and the amount of error is close to 0 then the amount of say will be high. When a stump does a bad job and the total error is around 0.5 it will give very little amount of say. When a stump's error is close to 1 the amount of say will be a large negative value.

To modify the weights of an incorrectly classified sample we use this formula-


![equation](http://www.sciweavers.org/upload/Tex2Img_1578252395/render.png)

The above formula will increase the sample weights which is a good thing as we want the incorrectly classified samples to be more important in the next stump.

to modify the weights of correctly classified samples we do the below formula -


![equation](http://www.sciweavers.org/upload/Tex2Img_1578252612/render.png)

this will decrease the importance of the sample going forward as we want to prioritise classifying the incorrectly classified samples.

After changing the weights we need to normalise the values by adding each of the sample weights together and dividing each weight by this number. After doing this we can use these new sample weights for the next stump.

After doing the above for a number of stumps to classify a sample you would take the amount of say of each of the stumps that predicted the sample to be in class 0 and the amount of say of each of the stumps that predicted the sample to be in class 1 and sum both up. Whichever number is higher is the final class prediction.

### How gradient descent models work

Gradient descent algorithms starts by making a single leaf instead of a tree or stump like in adaboost. this leaf represents an initial guess for the samples target variable- for regression it would be the average value. The gradient boost then builds a tree like ada boost based on previous trees errors but usually this tree is larger although still restricted. The tree is scaled much like how ada boost scales but all trees are scaled by the same amount. then gradient boost builds another tree based on errors made by previous tree. Continues to do this until it has made the number of trees you asked for or the additional trees fail to see improvement. 

For regression the errors of the previous tree are the differences between observed and predicted values. This difference is called the pseudo residual. When building the next tree we use the predictor variables to predict the residuals and not the original target variable. As usually we have less leaves than residuals, when there is more than one residual per leaf we take the average. Now we go back to our original guess and run the sample down the tree and add the residual on to the original guess. Doing this directly would mean that we would have very high variance and therefore overfitting. To deal with this gradient boosting has a learning rate to scale the contribution from the new tree so it would be -


![equation](http://www.sciweavers.org/upload/Tex2Img_1578253024/render.png)

Now this prediction will not be as good compared to just using the residual but its still better than the original guess. We can then use the residuals made from this tree to create another tree and follow the same process of finding what the residuals are. Once we do this we can scaled amounts from both trees to our initial guess.


![equation](http://www.sciweavers.org/upload/Tex2Img_1578253172/render.png)

We can repeat the process of using the residuals to make a new tree but this time we use the residuals made from both trees. And so on until we get to the specified amount of trees.

When we get new measurements we can predict the outcome of new data by starting with our original guess and then adding the relevant residuals from each tree. 


For classification it’s slightly more complicated as the initial prediction is the log of the odds. To calculate the log of the odds in a binary case you do -

![equation](http://www.sciweavers.org/upload/Tex2Img_1578253336/render.png)

to use the log of the odds for classification we turn it into a probability using the logistic function 


![equation](http://www.sciweavers.org/upload/Tex2Img_1578253433/render.png)

We can measure how bad the initial prediction is by calculating the pseudo residuals - the difference between the observed and the predicted values. Using the observed values means that we use either 1 or 0 depending on if the sample is in the positive class or not. 

Residuals = (observed - predicted)

Now we build a tree to predict the residual much like how we do for regression. However when we want to add the trees residuals onto the original prediction we have to use a transformation , the most common one being -

![equation](http://www.sciweavers.org/upload/Tex2Img_1578253623/render.png)

We can then add the residual from the tree onto the original prediction and then convert the new log odds prediction into a probability using 

![equation](http://www.sciweavers.org/upload/Tex2Img_1578253794/render.png)

We then calculate the residuals again using the new predicted probabilities for each of the samples and the observed values and repeat the process.

### How Light GBM differs

Light GBM is a gradient boosting framework that splits the tree leaf wise rather than depth wise or level wise. This means that it will be more complex than other methods and prone to overfit in small data sets. It will choose the leaf with max delta loss to grow, and when growing the same lead, leaf wise algorithms can reduce more loss than a level wise algorithm. GMB uses a GOSS method that keeps all instances with large gradients and performs random sampling on the instances with small gradients. eg we have 200K rows of data with 5K rows with a high gradient, it will keep the 5K rows and then x% amount of the remaining 195K rows chosen randomly. Assumption is that samples with small gradients have smaller training error and therefore are already well trained.

Light GBM also has an incredibly large amount of features to tweak highlighted in this [article](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc), the aforementioned article has a good run down of what all the most important features do.

## Using Light GBM in this dataset

Overall the performance of Light GBM was very impressive with a mean ROC score of 0.920 and an out of fold ROC score of 0.919. This is combined with fast training speeds makes it seem like Light GBM is the real deal.


