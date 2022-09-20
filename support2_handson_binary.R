#Evidera Hands-on Session
#Working with real binary outcome data
#Disclaimer: these analyses are for teaching purposes only
#Dave Vanness - 2019

rm(list=ls())
library(Hmisc)
library(tidyverse)
library(caret)
library(RANN)
library(MLmetrics)
library(ROCR)
library(parallel)
library(doParallel)
library(rpart)
library(xgboost)
library(earth)
library(mgcv)
library(recipes)
library(e1071)
library(rpart.plot)
getHdata("support2") 
#If "getHdata" doesn't work, go to the following website: http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets
#Then, look for "SUPPORT study datasets" and download the file support2.sav
#type the command: getwd() in your console and note the location. This is your R home directory.
#Move the support2.sav file to that location. Then, uncomment the following line and execute it:
#load(file = "support2.sav")

#Clean the data - declare factors, ensure integers are stored as double-precision variables.
#Drop cases with patterns of missing data that are excessive (note: many R packages available for investigating missingness. See 'mice')
tbl.1 = select(as_tibble(support2),-c(sfdm2,sps,aps,surv2m,surv6m,hday,prg2m,prg6m,dnr,dnrday,death,slos,d.time,dzclass,charges,totcst,totmcst,avtisst,adlp,adls)) %>% filter(!is.na(meanbp)&!is.na(crea)&!is.na(wblc)&!is.na(scoma)&!is.na(race)) %>% filter(!(is.na(glucose)|is.na(bun)|is.na(urine))) 
tbl.2 = tbl.1 %>% mutate(hospdead = factor(hospdead,levels = c("0","1"),labels = c("Alive","Dead"))) %>% 
  mutate(diabetes = factor(diabetes,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(dementia = factor(dementia,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(num.co = as.double(num.co)) %>%
  mutate(edu = as.double(edu)) %>%
  mutate(scoma = as.double(scoma)) %>%
  mutate(resp = as.double(resp)) %>%
  mutate(sod = as.double(sod))
tbl.2 = tbl.1 %>% mutate(hospdead = factor(hospdead,levels = c("0","1"),labels = c("Alive","Dead"))) %>% 
  mutate(diabetes = factor(diabetes,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(dementia = factor(dementia,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(num.co = as.double(num.co)) %>%
  mutate(edu = as.double(edu)) %>%
  mutate(scoma = as.double(scoma)) %>%
  mutate(resp = as.double(resp)) %>%
  mutate(sod = as.double(sod)) %>%
  mutate(age = as.double(age)) %>%
  mutate(dzgroup = as.double(dzgroup)) %>%
  mutate(meanbp = as.double(meanbp)) %>%
  mutate(wblc = as.double(wblc)) %>%
  mutate(hrt = as.double(hrt)) %>%
  mutate(temp = as.double(temp)) %>%
  mutate(pafi = as.double(pafi)) %>%
  mutate(alb = as.double(alb)) %>%
  mutate(bili = as.double(bili)) %>%
  mutate(crea = as.double(crea)) %>%
  mutate(ph = as.double(ph)) %>%
  mutate(glucose = as.double(glucose)) %>%
  mutate(bun = as.double(bun)) %>%
  mutate(urine = as.double(urine)) %>%
  mutate(adlsc = as.double(adlsc))

#Generate partition: 75% for estimation, 25% for validation  
set.seed(314159)
i.est = createDataPartition(y = tbl.2$hospdead,times = 1,p = .75,list = FALSE) #Creates a partition of the data balancing on factor death in hospital
est = tbl.2[i.est,]
val = tbl.2[-i.est,]

#Pre-process using the "recipe" method to impute missing predictors.
set.seed(314159)
recipe.support2 = recipe(hospdead ~ ., data = est) %>%
  step_impute_bag(all_predictors()) 
prep.support2 = prep(recipe.support2)
baked.est <- bake(prep.support2,new_data = est)
baked.val <- bake(prep.support2,new_data = val)

#Control function for the train method using CARET. Need "classProbs = TRUE" to be able to use the logLoss metric. This tripped us up in the live session.
fitControl <- trainControl(
  method = 'none',                   # single model
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = TRUE,                  
  summaryFunction=mnLogLoss # results summary function
)

#Logistic regression as the baseline comparator.
train.logit = train(hospdead ~ .,method="glm", family=binomial(link = "logit"),metric = "logLoss",data = baked.est,trControl=fitControl)

#Here is a helper function that will generate class probabilities, print the "confusion matrix" and make a calibration plot.
#Note: the default confusion matrix sets the "cutpoint" for predicted class at Pr=0.5 - this may or may not be appropriate.
ConCalROC <- function(train.obj,dat) {
  predict.class <- predict(object = train.obj,newdata = dat,type = "raw")
  cM = confusionMatrix(data = predict.class,reference = dat$hospdead,positive = "Dead")
  predict.prob <- predict(object = train.obj,newdata = dat,type = "prob")
  calib = data.frame(obs = dat$hospdead, prob = predict.prob$Dead)
  calPlotData = calibration(obs ~ prob, data = calib,cuts = quantile(x = calib$prob,probs = seq(0,.9,.1)),class = "Dead")
  ROC = performance(prediction(predictions = calib$prob,labels = calib$obs,label.ordering = c("Alive","Dead")),"tpr","fpr")
  list(cM = cM, calPlotData = calPlotData, ROC = ROC)
}
#Calling the function we just wrote:
res.logit.est = ConCalROC(train.obj = train.logit,dat = baked.est)
res.logit.val = ConCalROC(train.obj = train.logit,dat = baked.val)

#Now, we plot and print the results
print(res.logit.est$cM)
print(res.logit.val$cM)
ggplot(res.logit.est$calPlotData)
ggplot(res.logit.val$calPlotData)
plot(res.logit.est$ROC)
plot(res.logit.val$ROC,add=TRUE,col=2)
legend(x = "bottomright",legend = c("Estimation", "Validation"),fill = 1:2)

#Setting up the training control to do 5x repeated 5-fold cross validation.
fitControl <- trainControl(
  method = 'repeatedcv',            # repeated k-fold cross validation
  number = 5,                      # number of folds
  repeats = 5,                     # number of repeats
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = TRUE,                  # should class probabilities be returned
  summaryFunction=mnLogLoss, # results summary function
  allowParallel = TRUE
) 
#Complexity parameters at which we will calculate the loss function for tuning
grid.rpart = expand.grid(cp = seq(.004,.01,.001))

#Setting up parallel processing
nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel
#Train the rpart algorithm
set.seed(314159)
train.rpart = train(recipe.support2,data = est,method = "rpart",trControl = fitControl,metric = "logLoss",tuneGrid = grid.rpart,control = rpart.control(minsplit = 30, minbucket = 4, maxdepth = 10), parms=list(
  loss=matrix(c(0,2,1,0),byrow=TRUE,nrow=2)))
#Terminate the parallel processing backend
stopCluster(cl)

#Cross-validation plot
plot(train.rpart)
#Variable importance plot
print(varImp(train.rpart))
#The final tree
rpart.plot(train.rpart$finalModel)

#ROC, confusion matrix and calibration plots
res.rpart.est = ConCalROC(train.obj = train.rpart,dat = baked.est)
res.rpart.val = ConCalROC(train.obj = train.rpart,dat = baked.val)
print(res.rpart.est$cM)
print(res.rpart.val$cM)
ggplot(res.rpart.est$calPlotData)
ggplot(res.rpart.val$calPlotData)
plot(res.rpart.est$ROC)
plot(res.rpart.val$ROC,add=TRUE,col=2)
legend(x = "bottomright",legend = c("Estimation", "Validation"),fill = 1:2)
plot(res.logit.val$ROC)
plot(res.rpart.val$ROC,add=TRUE,col=2)
legend(x = "bottomright",legend = c("Logit", "Rpart"),fill = 1:2)

#XGBoost algorithm

# Define the training control
fitControl <- trainControl(
  method = 'repeatedcv',           # repeated k-fold cross validation
  number = 5,                      # number of folds
  repeats = 5,                     # number of repeats
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = TRUE,                  # should class probabilities be returned
  summaryFunction=mnLogLoss,
  allowParallel = TRUE
)

#Now that we're tuning over more than one parameter, setting up the "grid" is
#more important. Need to worry about run-time. This one has 7x3x1x3x1x5x1 = 420 combinations.
#Each of those 420 combinations gets estimated 5x5 times for repeated CV, so need to be judicious.
#Here we are training in steps. Trying to find tree depth/complexity first.
tunegrid.xgbTree = expand.grid(
  nrounds = c(1,5,10,25,50,100,200),
  max_depth = c(1,2,3),
  eta = .3,
  gamma = c(0,.01,.1,1),
  colsample_bytree = 1,
  min_child_weight = c(1,10,50,100,200),
  subsample = 1
)

nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel
set.seed(314159)
train.xgbTree = train(hospdead ~ .,data = baked.est,method = "xgbTree",trControl = fitControl,metric="logLoss",tuneGrid = tunegrid.xgbTree,verbose = TRUE,nthread = 1)
stopCluster(cl)

plot(train.xgbTree)
train.xgbTree$bestTune

#Have established that the algorithm works best with "stumps" - trees with just one split
#So, next focus on the optimal learning rate eta. 
tunegrid.xgbTree = expand.grid(
  nrounds = c(1,50,100,200,400,600,800,1000,2000),
  max_depth = 1,
  eta = c(.0375,.075,.15,.3),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel
set.seed(314159)
train.xgbTree = train(hospdead ~ .,data = baked.est,method = "xgbTree",trControl = fitControl,metric="logLoss",tuneGrid = tunegrid.xgbTree,verbose = TRUE,nthread = 1)
stopCluster(cl)

plot(train.xgbTree)
train.xgbTree$bestTune

print(varImp(train.xgbTree))

res.xgbTree.est = ConCalROC(train.obj = train.xgbTree,dat = baked.est)
res.xgbTree.val = ConCalROC(train.obj = train.xgbTree,dat = baked.val)
print(res.xgbTree.est$cM)
print(res.xgbTree.val$cM)
ggplot(res.xgbTree.est$calPlotData)
ggplot(res.xgbTree.val$calPlotData)
plot(res.xgbTree.est$ROC)
plot(res.xgbTree.val$ROC,add=TRUE,col=2)
legend(x = "bottomright",legend = c("Estimation", "Validation"),fill = 1:2)

plot(res.logit.val$ROC)
plot(res.rpart.val$ROC,add=TRUE,col=2)
plot(res.xgbTree.val$ROC,add=TRUE,col=3)
legend(x = "bottomright",legend = c("Logit", "Rpart", "xgbTree"),fill = 1:3)

#MARS (gcvEarth) : Multivariate Adaptive Regression Splines (Note: not working well with CARET at the moment, so using earth package directly)

#First degree(no interactions), backwards selection
set.seed(314159)
train.MARS1 = earth(hospdead ~ .,data = baked.est, glm=list(family=binomial(link="logit")), degree = 1, pmethod = "backward")
#Second degree (2-way interactions), backwards selection
set.seed(314159)
train.MARS2 = earth(hospdead ~ .,data = baked.est, glm=list(family=binomial(link="logit")), degree = 2, pmethod = "backward")
#First degree(no interactions), generalized cross-validation
set.seed(314159)
train.gcvMARS1 = earth(hospdead ~ .,data = baked.est, glm=list(family=binomial(link="logit")), degree = 1, pmethod = "cv", nfold = 5, ncross = 5)
#Second degree (2-way interactions), generalized cross-validation
set.seed(314159)
train.gcvMARS2 = earth(hospdead ~ .,data = baked.est, glm=list(family=binomial(link="logit")), degree = 2, pmethod = "cv", nfold = 5, ncross = 5)

#Generate predictions on the logit scale
pred.MARS1 = predict(train.MARS1,newdata = baked.val)
pred.gcvMARS1 = predict(train.gcvMARS1,newdata = baked.val)
pred.MARS2 = predict(train.MARS2,newdata = baked.val)
pred.gcvMARS2 = predict(train.gcvMARS2,newdata = baked.val)

#Convert logit scale predictions to probability scale and plot calibration curves and ROC
invlogit = function(x){exp(x)/(1+exp(x))}

MARS1.testProbs.val = data.frame(obs=baked.val$hospdead,prob.MARS1 = as.numeric(1-invlogit(pred.MARS1)))
calPlotData.val = calibration(obs ~ prob.MARS1, data = MARS1.testProbs.val,cuts = quantile(x = MARS1.testProbs.val$prob.MARS1,probs = seq(0,.9,.1)))
ggplot(calPlotData.val)

gcvMARS1.testProbs.val = data.frame(obs=baked.val$hospdead,prob.gcvMARS1 = as.numeric(1-invlogit(pred.gcvMARS1)))
calPlotData.val = calibration(obs ~ prob.gcvMARS1, data = gcvMARS1.testProbs.val,cuts = quantile(x = gcvMARS1.testProbs.val$prob.gcvMARS1,probs = seq(0,.9,.1)))
ggplot(calPlotData.val)

MARS2.testProbs.val = data.frame(obs=baked.val$hospdead,prob.MARS2 = as.numeric(1-invlogit(pred.MARS2)))
calPlotData.val = calibration(obs ~ prob.MARS2, data = MARS2.testProbs.val,cuts = quantile(x = MARS2.testProbs.val$prob.MARS2,probs = seq(0,.9,.1)))
ggplot(calPlotData.val)

gcvMARS2.testProbs.val = data.frame(obs=baked.val$hospdead,prob.gcvMARS2 = as.numeric(1-invlogit(pred.gcvMARS2)))
calPlotData.val = calibration(obs ~ prob.gcvMARS2, data = gcvMARS2.testProbs.val,cuts = quantile(x = gcvMARS2.testProbs.val$prob.gcvMARS2,probs = seq(0,.9,.1)))
ggplot(calPlotData.val)

plot(res.logit.val$ROC)
plot(res.rpart.val$ROC,add=TRUE,col=2)
plot(res.xgbTree.val$ROC,add=TRUE,col=3)
plot(performance(prediction(predictions = MARS1.testProbs.val$prob.MARS1,labels = MARS1.testProbs.val$obs,label.ordering = c("Dead","Alive")),"tpr","fpr"),add= TRUE,col=4)
plot(performance(prediction(predictions = gcvMARS1.testProbs.val$prob.gcvMARS1,labels = gcvMARS1.testProbs.val$obs,label.ordering = c("Dead","Alive")),"tpr","fpr"),add= TRUE,col=5)
plot(performance(prediction(predictions = MARS2.testProbs.val$prob.MARS2,labels = MARS2.testProbs.val$obs,label.ordering = c("Dead","Alive")),"tpr","fpr"),add= TRUE,col=6)
plot(performance(prediction(predictions = gcvMARS2.testProbs.val$prob.gcvMARS2,labels = gcvMARS2.testProbs.val$obs,label.ordering = c("Dead","Alive")),"tpr","fpr"),add= TRUE,col=7)
legend(x = "bottomright",legend = c("Logit", "Rpart", "xgbTree","MARS1","gcvMARS1","MARS2","gcvMARS2"),fill = 1:7)

