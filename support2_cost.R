rm(list=ls())
#install.packages(c("Hmisc","tidyverse","caret","RANN","MLmetrics","ROCR","doParallel","rpart","rpart.plot","xgboost","earth","mgcv","randomForest","recipes","e1071"))
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
library(labelled)
getHdata("support2")
tbl.1 = select(as_tibble(support2),-c(sfdm2,sps,aps,surv2m,surv6m,hday,prg2m,prg6m,dnr,dnrday,death,slos,d.time,dzclass,charges,hospdead,totmcst,avtisst,adlp,adls)) %>% filter(!is.na(totcst)&!is.na(scoma)) %>% filter(totcst>0)

tbl.2 = tbl.1 %>% 
  mutate(age = as.double(age)) %>%
  mutate(num.co = as.double(num.co)) %>%
  mutate(edu = as.double(edu)) %>%
  mutate(scoma = as.double(scoma)) %>%
  mutate(totcst = as.double(totcst)) %>%
  mutate(diabetes = factor(diabetes,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(dementia = factor(dementia,levels = c("0","1"),labels = c("No","Yes"))) %>%
  mutate(meanbp = as.double(meanbp)) %>%
  mutate(wblc = as.double(wblc)) %>%
  mutate(hrt = as.double(hrt)) %>%
  mutate(resp = as.double(resp)) %>%
  mutate(temp = as.double(temp)) %>%
  mutate(pafi = as.double(pafi)) %>%
  mutate(alb = as.double(alb)) %>%
  mutate(bili = as.double(bili)) %>%
  mutate(crea = as.double(crea)) %>%
  mutate(sod = as.double(sod)) %>%
  mutate(ph = as.double(ph)) %>%
  mutate(glucose = as.double(glucose)) %>%
  mutate(bun = as.double(bun)) %>%
  mutate(urine = as.double(urine)) %>%
  mutate(adlsc = as.double(adlsc))

set.seed(314159)
i.est = createDataPartition(y = tbl.2$totcst,times = 1,p = .75,list = FALSE) #Creates a partition of the data balancing on totcst
est = tbl.2[i.est,]
val = tbl.2[-i.est,]

set.seed(314159)
recipe.support2 = recipe(totcst ~ ., data = est) %>%
  step_impute_bag(all_predictors()) 
prep.support2 = prep(recipe.support2)
baked.est <- bake(prep.support2,new_data = est)
baked.val <- bake(prep.support2,new_data = val)

fitControl <- trainControl(
  method = 'repeatedcv',                   # single model
  number = 5,
  repeats = 5,
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  summaryFunction=defaultSummary # results summary function
) 

train.lm = train(totcst ~ .,method="glm", family=gaussian(),trControl = fitControl,metric = "RMSE",data = baked.est)
train.glm = train(totcst ~ .,method="glm", family=Gamma(link="log"),trControl = fitControl,metric = "RMSE",data = baked.est, control=glm.control(epsilon = 1e-4, maxit = 200, trace = FALSE))

pred.lm = predict(train.lm,newdata = baked.val)
pred.glm = predict(train.glm,newdata = baked.val)
df.val.lm = data.frame(obs=baked.val$totcst,pred=pred.lm,model="lm",dataType="Test")
df.val.glm = data.frame(obs=baked.val$totcst,pred=pred.glm,model="glm",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm)
plotObsVsPred(df.val)

fitControl <- trainControl(
  method = 'repeatedcv',                   # repeated k-fold cross validation
  number = 5,                      # number of folds
  repeats = 5,                     # number of repeats
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  summaryFunction=defaultSummary, # results summary function
  allowParallel = TRUE
) 
grid.rpart = expand.grid(cp = seq(.004,.02,.002))

nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel
train.rpart = train(totcst ~ .,data = baked.est,method = "rpart",trControl = fitControl,metric = "RMSE",tuneGrid = grid.rpart)
stopCluster(cl)

plot(train.rpart)
print(varImp(train.rpart))
rpart.plot(train.rpart$finalModel)

pred.rpart = predict(train.rpart,newdata = baked.val)
df.val.rpart = data.frame(obs=baked.val$totcst,pred=pred.rpart,model="rpart",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.rpart)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

#XGBoost

tunegrid.xgbTree = expand.grid(
  nrounds = c(1,5,10,25,50,100,200,500),
  max_depth = c(1,2,3),
  eta = .3,
  gamma = 0,
  colsample_bytree = c(.33,.66,1),
  min_child_weight = 1,
  subsample = c(.33,.66,1)
)

nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel
train.xgbTree = train(totcst ~ .,data = baked.est,method = "xgbTree",trControl = fitControl,metric="RMSE",tuneGrid = tunegrid.xgbTree,verbose = TRUE,nthread = 1)
stopCluster(cl)

plot(train.xgbTree)
train.xgbTree$bestTune

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
train.xgbTree = train(totcst ~ .,data = baked.est,method = "xgbTree",trControl = fitControl,metric="RMSE",tuneGrid = tunegrid.xgbTree,verbose = TRUE,nthread = 1)
stopCluster(cl)

plot(train.xgbTree)
train.xgbTree$bestTune
print(varImp(train.xgbTree))

pred.xgbTree = predict(train.xgbTree,newdata = baked.val)
df.val.xgbTree = data.frame(obs=baked.val$totcst,pred=pred.xgbTree,model="xgbTree",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))


#MARS (gcvEarth)

train.MARS1 = earth(totcst ~ .,data = baked.est, degree = 1, pmethod = "backward")
train.MARS2 = earth(totcst ~ .,data = baked.est, degree = 2, pmethod = "backward")
train.gcvMARS1 = earth(totcst ~ .,data = baked.est, degree = 1, pmethod = "cv", nfold = 5, ncross = 5)
train.gcvMARS2 = earth(totcst ~ .,data = baked.est, degree = 2, pmethod = "cv", nfold = 5, ncross = 5)

pred.MARS1 = predict(train.MARS1,newdata = baked.val)
pred.gcvMARS1 = predict(train.gcvMARS1,newdata = baked.val)
pred.MARS2 = predict(train.MARS2,newdata = baked.val)
pred.gcvMARS2 = predict(train.gcvMARS2,newdata = baked.val)
df.val.MARS1 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.MARS1),model="MARS1",dataType="Test")
df.val.MARS2 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.MARS2),model="MARS2",dataType="Test")
df.val.gcvMARS1 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.gcvMARS1),model="gcvMARS1",dataType="Test")
df.val.gcvMARS2 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.gcvMARS2),model="gcvMARS2",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2)
plotObsVsPred(df.val) # do this!
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

train.GAM1 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                 diabetes + dementia + ca + s(meanbp) + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + s(alb) + 
                 s(bili) + s(crea) + s(sod) + s(ph) + s(glucose) + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)

train.GAM1 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                    diabetes + dementia + ca + s(meanbp) + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + s(alb) + 
                    s(bili) + s(crea) + s(sod) + s(ph) + s(glucose) + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "REML")

train.GAM1 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                    diabetes + dementia + ca + s(meanbp) + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + s(alb) + 
                    s(bili) + s(crea) + s(sod) + s(ph) + s(glucose) + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "REML", select = TRUE)


pred.GAM1 = predict(train.GAM1,newdata = baked.val)

df.val.GAM1 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM1),model="GAM1",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM1)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM1)

train.GAM1 <- gam(formula = totcst ~ age + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + hrt + resp + temp + pafi + alb + 
                    bili + crea + sod + ph + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM1 = predict(train.GAM1,newdata = baked.val)

df.val.GAM1 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM1),model="GAM1",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM1)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM1)

train.GAM2 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + hrt + resp + temp + pafi + alb + 
                    s(bili) + crea + sod + ph + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM2 = predict(train.GAM2,newdata = baked.val)

df.val.GAM2 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM2),model="GAM2",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM2)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM2)

train.GAM3 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + s(hrt) + resp + temp + pafi + alb + 
                    s(bili) + crea + sod + ph + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM3 = predict(train.GAM3,newdata = baked.val)

df.val.GAM3 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM3),model="GAM3",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM3)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM3)

train.GAM4 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + s(edu) + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + s(hrt) + resp + temp + pafi + alb + 
                    s(bili) + crea + sod + ph + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM4 = predict(train.GAM4,newdata = baked.val)

df.val.GAM4 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM4),model="GAM4",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM4)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM4)

train.GAM5 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + s(hrt) + resp + temp + pafi + alb + 
                    s(bili) + crea + sod + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM5 = predict(train.GAM5,newdata = baked.val)

df.val.GAM5 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM5),model="GAM5",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM5)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM5)

train.GAM6 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + wblc + s(hrt) + resp + temp + s(pafi) + alb + 
                    s(bili) + crea + sod + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM6 = predict(train.GAM6,newdata = baked.val)

df.val.GAM6 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM6),model="GAM6",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM6)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM6)

train.GAM7 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + resp + temp + s(pafi) + alb + 
                    s(bili) + crea + sod + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM7 = predict(train.GAM7,newdata = baked.val)

df.val.GAM7 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM7),model="GAM7",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM7)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM7)

train.GAM8 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + scoma + race + 
                    diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + resp + temp + s(pafi) + alb + 
                    s(bili) + crea + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM8 = predict(train.GAM8,newdata = baked.val)

df.val.GAM8 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM8),model="GAM8",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM8)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM8)


train.GAM9 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                    diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + resp + temp + s(pafi) + alb + 
                    s(bili) + crea + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM9 = predict(train.GAM9,newdata = baked.val)

df.val.GAM9 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM9),model="GAM9",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM9)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM9)


train.GAM10 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                    diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + temp + s(pafi) + alb + 
                    s(bili) + crea + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM10 = predict(train.GAM10,newdata = baked.val)

df.val.GAM10 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM10),model="GAM10",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM10)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM10)


train.GAM11 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + temp + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM11 = predict(train.GAM11,newdata = baked.val)

df.val.GAM11 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM11),model="GAM11",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM11)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM11)


train.GAM12 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + s(meanbp) + s(wblc) + s(hrt) + s(resp) + temp + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM12 = predict(train.GAM12,newdata = baked.val)

df.val.GAM12 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM12),model="GAM12",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM12)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM12)

train.GAM13 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + temp + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + s(glucose) + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM13 = predict(train.GAM13,newdata = baked.val)

df.val.GAM13 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM13),model="GAM13",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM13)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM13)


train.GAM14 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + bun + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM14 = predict(train.GAM14,newdata = baked.val)

df.val.GAM14 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM14),model="GAM14",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM14)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM14)

train.GAM15 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + s(bun) + urine + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM15 = predict(train.GAM15,newdata = baked.val)

df.val.GAM15 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM15),model="GAM15",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM15)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM15)

train.GAM16 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + s(bun) + s(urine) + adlsc, data = baked.est, method = "GCV.Cp",)
pred.GAM16 = predict(train.GAM16,newdata = baked.val)

df.val.GAM16 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM16),model="GAM16",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM16)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM16)

train.GAM17 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + s(crea) + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM17 = predict(train.GAM17,newdata = baked.val)

df.val.GAM17 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM17),model="GAM17",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM17)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM17)

train.GAM18 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM18 = predict(train.GAM18,newdata = baked.val)

df.val.GAM18 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM18),model="GAM18",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM18)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM18)

train.GAM19 <- gam(formula = totcst ~ s(age) + sex + dzgroup + num.co + edu + s(scoma,by = dzgroup) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM19 = predict(train.GAM19,newdata = baked.val)

df.val.GAM19 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM19),model="GAM19",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM19)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM19)


train.GAM20 <- gam(formula = totcst ~ s(age, by=dzgroup) + sex + dzgroup + num.co + edu + s(scoma,by = dzgroup) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM20 = predict(train.GAM20,newdata = baked.val)

df.val.GAM20 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM20),model="GAM20",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM20)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM20)



train.GAM21 <- gam(formula = totcst ~ s(age, by=dzgroup) + sex + dzgroup + num.co + edu + s(scoma,by = dzgroup) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili,by=dzgroup) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM21 = predict(train.GAM21,newdata = baked.val)

df.val.GAM21 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM21),model="GAM21",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM21)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM21)


train.GAM22 <- gam(formula = totcst ~ s(age, by=dzgroup) + sex + dzgroup + num.co + edu + s(scoma,by = dzgroup) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt,by=dzgroup) + s(resp) + s(temp) + s(pafi) + alb + 
                     s(bili,by=dzgroup) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM22 = predict(train.GAM22,newdata = baked.val)

df.val.GAM22 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM22),model="GAM22",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM22)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM22)

train.GAM23 <- gam(formula = totcst ~ s(age, by=dzgroup) + sex + dzgroup + num.co + edu + s(scoma,by = dzgroup) + race + 
                     diabetes + dementia + ca + meanbp + s(wblc) + s(hrt,by=dzgroup) + s(resp, by=dzgroup) + s(temp) + s(pafi) + alb + 
                     s(bili,by=dzgroup) + crea + s(sod) + s(ph) + glucose + s(bun) + s(urine) + s(adlsc), data = baked.est, method = "GCV.Cp",)
pred.GAM23 = predict(train.GAM23,newdata = baked.val)

df.val.GAM23 = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM23),model="GAM23",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM23)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))

summary(train.GAM23)

#22 is best so far

train.GAM = train.GAM22
pred.GAM = predict(train.GAM,newdata = baked.val, )

df.val.GAM = data.frame(obs=baked.val$totcst,pred=as.numeric(pred.GAM),model="GAM",dataType="Test")
df.val = rbind(df.val.lm,df.val.glm,df.val.xgbTree,df.val.MARS1,df.val.MARS2,df.val.gcvMARS1,df.val.gcvMARS2,df.val.GAM)
plotObsVsPred(df.val)
df.val$L2 = (df.val$obs-df.val$pred)^2
by(data = df.val[,"L2"],INDICES = df.val[,"model"],function(x) sqrt(mean(x)))


head(pred.GAM)
quantile(pred.GAM,seq(0.1,1,.1))
test1 = cut(pred.GAM,breaks = quantile(pred.GAM,seq(0,1,.1)),labels=FALSE, include.lowest = TRUE)
calib_cont = data.frame(baked.val$totcst,pred.GAM)
calib.df.GAM = data.frame(pred=as.numeric(by(data = calib_cont$pred.GAM,INDICES = test1,FUN = mean)),
se = as.numeric(by(data = calib_cont$pred.GAM,INDICES = test1,FUN = std.error)),
obs = as.numeric(by(data = calib_cont$baked.val.totcst,INDICES = test1,FUN = mean)))
plot(calib.df.GAM$pred,calib.df.GAM$obs,ylim=c(-10000,100000))
abline(a = 0,b = 1)
arrows(x0 = calib.df.GAM$pred,y0=calib.df.GAM$pred-1.96*calib.df.GAM$se,x1 = calib.df.GAM$pred,y1=calib.df.GAM$pred+1.96*calib.df.GAM$se,length = .05,angle = 90,code = 3)

test1 = cut(baked.val$totcst,breaks = quantile(baked.val$totcst,seq(0,1,.1)),labels=FALSE, include.lowest = TRUE)
calib_cont = data.frame(baked.val$totcst,pred.GAM)
calib.df.GAM = data.frame(pred=as.numeric(by(data = calib_cont$pred.GAM,INDICES = test1,FUN = mean)),
                          lb = as.numeric(by(data = calib_cont$pred.GAM,INDICES = test1,FUN = quantile, probs = 0.25)),
                          ub = as.numeric(by(data = calib_cont$pred.GAM,INDICES = test1,FUN = quantile, probs = 0.75)),
                          leftb = as.numeric(by(data = calib_cont$baked.val.totcst,INDICES = test1,FUN = quantile, probs = 0.25)),
                          rightb = as.numeric(by(data = calib_cont$baked.val.totcst,INDICES = test1,FUN = quantile, probs = 0.75)),
                          obs = as.numeric(by(data = calib_cont$baked.val.totcst,INDICES = test1,FUN = mean)))
plot(calib.df.GAM$obs,calib.df.GAM$pred,ylim=c(0,150000),xlim=c(0,150000))
abline(a = 0,b = 1)
arrows(x0 = calib.df.GAM$obs,y0=calib.df.GAM$lb,x1 = calib.df.GAM$obs,y1=calib.df.GAM$ub,length = .05,angle = 90,code = 3)
arrows(x0 = calib.df.GAM$leftb,y0=calib.df.GAM$pred,x1 = calib.df.GAM$rightb,y1=calib.df.GAM$pred,length = .05,angle = 90,code = 3)


