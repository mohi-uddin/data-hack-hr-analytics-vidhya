ls()
rm(list=ls())

#load directory

path <- "D:/mohi/analytics/av/data hack hr"

setwd(path)

#load data
train <- read.csv("Train.csv",na.strings=c(""," ","NA"))
test <- read.csv("Test.csv",na.strings=c(""," ","NA"))
View(train)
train$Poutcome=as.factor(train$Poutcome)
test$Poutcome=as.factor(test$Poutcome)


#libraray
library(mlr)
library(data.table)

#impute missing values by class
imp <- impute(train, classes = list(factor = imputeMode(), integer = imputeMedian(),numeric=imputeMedian()))
imp1 <- impute(test, classes = list(factor = imputeMode(), integer = imputeMedian(), numeric=imputeMedian()))

train <- imp$data
test <- imp1$data

table(is.na(test))
table(is.na(train))

sapply(imp_train, function(x) sum(is.numeric(x)))
z = imp_train[c("ID","Age","Pdays","Emp.Var.Rate","Cons.Price.Idx","Cons.Conf.Idx","Campaign","Previous")]

cor(z)

test$Outcome <- sample(c("no","yes"),size = 12251,replace = T)
setDT(train)
setDT(test)

#removing variables ------------

train[,Education := NULL]
test[,Education := NULL]

train[,Job := NULL]
test[,Job := NULL]

train$Pdays= NULL
test$Pdays = NULL

# Encoding initial_list_status as 0,1 -------------------------------------

train$Default[1:50]
train[,Default := as.integer(as.factor(Default))-1]
test[,Default := as.integer(as.factor(Default))-1]


train$Default[1:50]
train[,Contact := as.integer(as.factor(Contact))-1]
test[,Contact := as.integer(as.factor(Contact))-1]

# One Hot Encoding --------------------------------------------------------

train_mod <- train[,.(Poutcome,Day_Of_Week,Month,Loan,Housing,Marital)]
test_mod <- test[,.(Poutcome,Day_Of_Week,Month,Loan,Housing,Marital)]
train_mod[1:50]
# train_mod[is.na(train_mod)] <- "-1"
# test_mod[is.na(test_mod)] <- "-1"
train_ex[1:50]
train_ex <- model.matrix(~.+0, data = train_mod)
test_ex <- model.matrix(~.+0, data = test_mod)

train1 <- as.data.table(train_ex)
test1 <- as.data.table(test_ex)

new_train <- cbind(train, train1)
new_test <- cbind(test, test1)

new_train[,c("Poutcome","Day_Of_Week","Month","Loan","Housing","Marital") := NULL]
new_test[,c("Poutcome","Day_Of_Week","Month","Loan","Housing","Marital") := NULL]

train = new_train
test = new_test


for(i in colnames(train)[sapply(train, is.numeric)])
  set(x = train, j = i, value = as.factor(as.numeric(train[[i]])))


train$Emp.Var.Rate = as.numeric(train$Emp.Var.Rate)
train$Campaign = as.integer(train$Campaign)
train$Cons.Conf.Idx = as.numeric(train$Cons.Conf.Idx)
train$Cons.Price.Idx = as.numeric(train$Cons.Price.Idx)
train$Age = as.numeric(train$Age)
train$Previous = as.numeric(train$Previous)
train$ID = as.numeric(train$ID)


for(i in colnames(test)[sapply(test, is.numeric)])
  set(x = test, j = i, value = as.factor(as.numeric(test[[i]])))

str(train)
test$Emp.Var.Rate = as.numeric(test$Emp.Var.Rate)
test$Campaign = as.integer(test$Campaign)
test$Cons.Conf.Idx = as.numeric(test$Cons.Conf.Idx)
test$Cons.Price.Idx = as.numeric(test$Cons.Price.Idx)
test$Age = as.numeric(test$Age)
test$Previous = as.numeric(test$Previous)
test$ID = as.numeric(test$ID)
test$Outcome = as.factor(test$Outcome)

#create a task
#trainTask <- makeClassifTask(data = cd_train,target = "Loan_Status")
testTask <- makeClassifTask(data = test, target = "Outcome")

trainTask <- makeClassifTask(data = train,target = "Outcome", positive = "yes")
trainTask
testTask


#normalize the variables
trainTask <- normalizeFeatures(trainTask,method = "standardize")
testTask <- normalizeFeatures(testTask,method = "standardize")

trainTask <- dropFeatures(task = trainTask,features = c("ID"))

#Decision Tree
getParamSet("classif.rpart")

#make tree learner
makeatree <- makeLearner("classif.rpart", predict.type = "response")

#grid search to find hyperparameters
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)

#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#do a grid search
gscontrol <- makeTuneControlGrid()

#hypertuning #took 15 minutes
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = trainTask, par.set = gs, control = gscontrol, measures = acc)
#best parameter
stune$x #returns a list of best parameters

#cross validation result
stune$y 

#using hyperparameters for modeling
t.tree <- setHyperPars(makeatree, par.vals = stune$x)
o.tree <- setHyperPars(makeatree, par.vals = list(minsplit = 19, minbucket = 50, cp = 0.001))

#train a model
t.rpart <- train(t.tree, trainTask)
getLearnerModel(t.rpart)
m = test$ID

#predict
tpmodel <- predict(t.rpart, testTask)
head(tpmodel)
#submission file
submit <- data.frame(ID = m, Outcome = tpmodel$data$response)
write.csv(submit, "dtree_encoding.csv",row.names = F) 
