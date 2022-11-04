# GUIDE Regression Tree Analysis on Factors Affecting Probability on Dying from COVID-19 (GUIDE, R)

What kind of people are more suceptible to dying from COVID-19 when they get infected? With a dataset reported from 21 healthsystems with 145,944 subjects hospitalized with COVID-19 collected from February 1st, 2020 to January 31, 2022:

Goal was to find which characteristic of someone hospitalized for COVID-19 is the most significant in affecting one's proability in dying from the disease.
Fitted a standard logistic model, GUIDE random forest, GUIDE logistic regression tree, and GUIDE tree to predict the probability of one dying from COVID-19, infer about th effects of the variables and the accuracy of the prediction models.
Found predicted values from using Rpart, Ctree, Cforest, and RandomForest to compare its performance with that of GUIDE tree, and GUIDE forest.


## Constructing a GUIDE regression tree for estimating probability of death 
By using GUIDE, I created a GUIDE regression tree to visualize splitting the data into more specific nodes, starting off by finding the 1st best varaible to split upon, 2nd, and 3rd best variable to split with. Each tree created as shown below is a piecewise constant least-squares regression tree for predicting `died`. If the condition is satisfied, an observation goes to the left branch. The italics reports the sample size and below, the mean of `died`. If the terminal node has a mean above 0.041, it is colored yellow, and if the mean is below 0.041, the node is colored blue.

Consequently, by comparing all the trees splitted by the first, second, and third best variable in predicting `died`, we can tell why they are respectively ranked the way they are reported by GUIDE as the trees are more concise and shorter when it is splitting upon a variable ranked higher. 

### Splitting with first best variable: agecat 
`agecat`: age groups (0= 18-50, 1= 50-59, 2= 60-69, 3= 70-79, 4= 80-90 years old) 
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>

### Splitting with second best variable: renal 
`renal`: Renal disease (0= no, 1= yes) 
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>

### Splitting with third best variable: CHF 
`CHF`: Congestive Heart Failure (0= no, 1= yes) 
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>




## Finding importance scores for each variable 
Each variable has their own predictive power, in other words, how important they are in the model. A variable with a higher score will have a larger effect on the model in predicting the independent variable. Here, I have visualized each variable's importance scores, where: 

    - A: >= 99% confidence 
    
    - B: 95-99% confidence
    
    - C: 90-95% confidence 
    
    - D: 80-90% confidence
    
    - E: < 80% confidence 

From the graph below, we can tell each variable can have their importance scores reported with a very high confidence level, and as we know by having GUIDE report `agecat` as the best splitting variable, `agecat` shows the highest improtance score among all the other variables in this dataset. 
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>




## Comparing the accuracy of different prediction models. 
### 1. GUIDE forest 
Ran a GUIDE forest prediction on the original dataset, `s1`, through GUIDE and appended the prediction column to `s1`, call it `s2`, this column is our GUIDE Forest predicted values 
```
s1 = read.csv("s1.csv",header=T)
forest = read.table("forestpred",header=T)
s1$pred = forest$predicted
write.csv(s1,"/Users/eson7/Desktop/stat443/homework2/s2.csv", row.names = FALSE)
```
### 2. Simple linear regression
Running a simple linear regression with all the variables, excluding variable `charlson` as it is a comorbidity index (the weighted sum of comorbidities). 
 ``` 
s1_lr = read.csv("s1.csv")
summary(s1_lr)
set.seed(1)
sample = sample(c(T,F),nrow(s1_lr), replace = T, prob= c(0.7,0.3))
train = s1_lr[sample,]
test = s1_lr[!sample,]
model = glm(died ~ sex + race + cancer + MI + CHF + cerebro + dementia + CPD + RD + PUD +diabetes + hemipara + renal + metastaticcancer + aids + PVD + agecat + liver, family = "binomial",data = s1_lr)
predicted = predict(model, s1_lr, type = 'response')
write.csv(predicted,"/Users/eson7/Desktop/stat443/hw2/slm.csv", row.names =FALSE)
```
### 3. GUIDE logistic regression tree
Running a logistic regression tree on GUIDe using `s2` (the dataset with an extra column of s1's predicted values) under GUIDE to generate predicted values through a GUIDE logistic regression tree. Modified the description file with `charlson` coded as `s`(using variable `charlson` as a numerical variable only used for splitting nodes), all other variables coded as `b` (catagorical variable used both for splitting and for node modeling), `agecat` as `c` (treated as a catagorical variable). Finally, we add an addition line to the description file, `21 pred e` indicating an extra 21st column named `pred` coded as `e`(estimated proability variable for logistic regression) to store our predicted values. 


### 4. GUIDE tree 
Running a linear regression tree on GUIDE using`s1` by modifying the description file with `charlson` coded as `s`, all other variables coded as `c`, `agecat` as `n` (numerical variable used both for splitting nodes and fitting node models). 

### Plotting GUIDE tree, GUIDE logistic tree, and standard logistic model against GUIDE forest 
```
log_reg= read.csv("slm.csv")
s2.pred = read.table("s2.pred", header= T)
tree.pred = read.table("tree.fit",header = T)
s2 = read.csv("s2.csv")
forest = s2$pred
guide_logitree = s2.pred$predicted
s_logit = log_reg$x
guide_tree = tree.pred$predicted
par(mfrow=c(1,3),pty="s")
#plot 1: Standard logistic model vs Guide Forest
plot(forest,s_logit,xlab = "Guide Forest",ylab = "Standard logistic
model",pch=".",xlim=c(0,0.7),ylim=c(0,0.7), col = "red")
abline(coef = c(0,1),col="blue")
#plot 2: Guide tree vs Guide Forest
plot(forest,guide_tree,xlab = "Guide Forest",ylab = "Guide
Tree",pch=".",xlim=c(0,0.7),ylim=c(0,0.7), col = "red")
abline(coef = c(0,1),col="blue")
#plot 3: Guide Logistic tree vs Guide Forest
plot(forest,guide_logitree,xlab = "Guide Forest",ylab = "Guide Logistic
Tree",pch=".",xlim=c(0,0.7),ylim=c(0,0.7), col = "red")
abline(coef = c(0,1),col="blue")
```
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>

## Comparing prediction accuracy (Rpart, Ctree, Cforest, RandomForest) 
Rpart and Ctree are both very powerful machine learning methods to build classification and regression trees. By visualizing the predicted values Rpart, Ctree, Cforest, and RandomForest creates, 

### Visualizing RPart, CTree 
generating Rpart predicted values and generating plot 
```
library(rpart)
library(rpart.plot)
z = read.csv("covid.csv", header=T, na.strings=c("NA",""))
z$sex = factor(z$sex)
z$race = factor(z$race)
z$died = as.numeric(z$died)
z$agecat <- factor(z$agecat)
z$liver <- factor(z$liver)
rp <- rpart(died ~ ., data=z, method="anova")
rp.fit <- predict(rp, newdata=z)
rpart.plot(rp,type=2)
```
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>


generating Ctree predicted values and generating plot
```
library(partykit)
z = read.csv("covid.csv",header=T,na.strings=c("NA",""))
z$sex = factor(z$sex)
z$race = factor(z$race)
z$died = as.numeric(z$died)
z$agecat <- factor(z$agecat)
z$liver <- factor(z$liver)
cntrl = ctree_control(minbucket=1000,minsplit=2000)
ctree.model = ctree(died ~ .,data = z, control = cntrl)
ct.fit = predict(ctree.model,newdata=z)
plot(ctree.model,type="simple")
```
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>

### Graphing Rpart, Ctree, RandomForest, Cforest 
(i) Reading in data 
```
z = read.csv("s1.csv",header=T,na.strings=c("NA",""))
z$died <- as.numeric(z$died)
z$agecat <- factor(z$agecat)
z$sex <- factor(z$sex)
z$race <- factor(z$race)
z$liver = factor(z$liver)
guide.forest=read.table("guide.forest",header=T)
guide.tree=read.table("guide.tree",header=T)
rpart_new = read.table("rpart_new.fit",header=T)
ctree_new = read.table("ctree_new.fit",header=T)
```
(ii) Generating RandomForest and measuring running time 
```
library(randomForest)
rf = randomForest(died ~ ., data=z)
rf.time=system.time({rf = randomForest(died ~ ., data=z)})
rf.fit = predict(rf,newdata=z)
print(rf.time)
#time elapsed: 476.067
rforest = read.table("rf.fit",header=T)
```
(iii) Generating Cforest and measuring running time 
```
library(partykit)
cf.model= cforest(died ~ ., data=z)
cf.time = system.time({cf.model= cforest(died ~ ., data=z)})
cf.fit = predict(cf.model,newdata=z)
print(cf.time)
#time elapsed: 771.437
cforest = read.table("cf.fit",header=T)
```
(iv) Plotting 
```
par(mfrow=c(2,2),pty="m",oma = c(0,0,2,0))

#plot 1: rPart vs Guide Tree
plot(guide.tree$x,rpart_new$x,xlab = "Guide Tree",ylab = "rPart",pch=".", cex
= 6,xlim=c(0,0.25),ylim=c(0,0.25), col = "red")
abline(coef = c(0,1),col="blue")

#plot 2: Ctree vs Guide Tree
plot(guide.tree$x,ctree_new$x,xlab = "Guide Tree",ylab = "Ctree", pch=".",
cex = 6, xlim=c(0,0.25),ylim=c(0,0.25), col = "red")
abline(coef = c(0,1),col="blue")

#plot 3: RandomForest vs Guide Forest
plot(guide.forest$x,rforest$x,xlab = "Guide Forest",ylab =
"RandomForest",pch=".", cex = 2,xlim=c(0,0.7),ylim=c(0,0.7), col = "red")
abline(coef = c(0,1),col="blue")

#plot 4: Cforest vs Guide Forest
plot(guide.forest$x,cforest$x,xlab = "Guide Forest",ylab = "Cforest",pch=".",
cex = 2,xlim=c(0,0.7),ylim=c(0,0.7), col = "red")
abline(coef = c(0,1),col="blue")
```
<p align="center">
  <img width="460" height="460" src="https://github.com/soneunii/relative-humidity-on-covid-19/raw/main/Figure1.png">
</p>

### Conclusion 
- Running time: by using `system.time` in R to measure the running time, I found that the execution time for running Cforest is substantially longer than running RandomForest. 
- Both rpart and ctree focus on performing splits between the dependent variables based
on the values existing. While rpart uses information measures for selecting the current
covariate, Ctree avoides the variable selection bias of rpart. This variable selection bias
is generated by selecting variables that have many possible splits or missing values.
Ctree, on the other hand, uses significance tests to select variables. In other words,
Ctree uses significance tests in order to avoid overfitting.
- The Cforest method is more accurate than the randomForest method in predicting,
however, according to its running time, it runs more slower and seems to handle less
data for the same memory. The reason for Cforest providing more accurate predictions
is because it produces unbiased trees.
