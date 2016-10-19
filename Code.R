############################## Team CTRL R _ Project 1 ##########################
getwd()

install.packages("ggplot2")
install.packages("lattice")
install.packages("caret")
install.packages("sampling") #for using function strata
install.packages("AppliedPredictiveModeling") # for visualizations in caret
install.packages("pROC")  #for variable importance 
install.packages("e1071")  # recursive function selection
install.packages("randomForest")
library(ggplot2)
library(lattice)
library(caret)
library(sampling)
library(AppliedPredictiveModeling)
library(pROC)
library(e1071)
library(randomForest)

# Importing the table
SantanderTrain <- read.csv(file.choose())
View(SantanderTrain)

nrow(SantanderTrain)
ncol(SantanderTrain)

######Checking for missing values in the data ######
colSums(is.na(SantanderTrain))

# No missing values 

# # ----- This is not working as deleting most of the columns 
# SantanderTrain_V1 <- nearZeroVar(SantanderTrain, saveMetrics = TRUE)
# colnames(SantanderTrain_V1)

# To remove 1st column as it is only ids

SantanderTrain_V1 <- SantanderTrain[,c(-1)]
View(SantanderTrain_V1)


##############################################
#### Using NearZeroVariance 
##############################################

## Removing columns with values as 0.
santander1 <- nearZeroVar(SantanderTrain_V1, saveMetrics = TRUE)
str(santander1)

## List of Zero varinace predictors
x <- santander1[santander1[,"zeroVar"] > 0,]
## shows 34 variables with zero variance
nrow(x)
##list of nearzero variance predictors
y <- santander1[santander1[,"zeroVar"] + santander1[,"nzv"] > 0, ] 
## shows 317 variables with  near zero variance
nrow(y)

########################################################
##################### Removing the zero variance columns###########
SantanderTrain_V2 <- SantanderTrain_V1[,!apply(SantanderTrain_V1, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
View(SantanderTrain_V2)


ncol(SantanderTrain_V2)
ncol(SantanderTrain_V1)
# hence out of 370 columns, 34 columns had 0 variance, final total number of columns = 336

# To understand the data type of target variable (which is 'target')

is.factor(SantanderTrain_V2$TARGET)
# result is false, hence need to convert it into factor
SantanderTrain_V2$TARGET <- as.factor(SantanderTrain_V2$TARGET)
table(SantanderTrain_V2$TARGET)

install.packages("DMwR")
library(DMwR)

##########################################
#####  Smote Approach for sampling 
##########################################
attach(SantanderTrain_V2)
samplesan<- SMOTE(TARGET~.,data = SantanderTrain_V2, perc.over = 200, k = 5, perc.under = 200,learner = NULL)
table(samplesan$TARGET)
# 0     1 
# 12032  9024 

View(samplesan)
##########################################

# Performing stratified sampling --------------- Currently we are not using this one ----------

SantanderTrain_Target= strata(SantanderTrain_V2,stratanames="TARGET",size=c(3008,3008), method = "srswor")
table(SantanderTrain_Target$TARGET)
nrow(SantanderTrain_Target)

# To create the sample data

SantanderTrain_sample <- getdata(SantanderTrain_V2, SantanderTrain_Target)
ncol(SantanderTrain_sample)
colnames(SantanderTrain_sample)
View(SantanderTrain_sample)

SantanderTrain_sample <- SantanderTrain_sample[,-c(337, 338, 339)]  # removing prob, id_unit, stratum from SantanderTrain_target Dataset
View(SantanderTrain_sample)

########################### end of stratified sampling ############################################33

# now total number of records : 6016 : 1 : 3008 and 0: 3008

# Processing on columns to remove the variance 

SantanderTrain_sample <- samplesan[,!apply(samplesan, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
ncol(SantanderTrain_sample)

# Total number of columns : 282

## creating a data.frame will all the numeric variables and creating data_frame

allnumvar <- SantanderTrain_sample[,sapply(SantanderTrain_sample, is.numeric)]
str(allnumvar)

View(SantanderTrain_sample)
View(numeric_variable)

################## To find the correlation, and removing variables which are more than 90% correlated 

correlation1 <- cor(allnumvar, use = "pairwise.complete.obs")
View(correlation1)

highcorrelation <- findCorrelation(correlation1,cutoff=0.9,verbose=T)
View(highcorrelation)
str(highcorrelation)

write.csv(highcorrelation,file = "highcorr.csv")

# to remove highly correlated variables from the sample data set 
santander2 <- SantanderTrain_sample[,-c(highcorrelation)]
ncol(santander2)
# Number of columns after removing 0 variance columns, and highly correlated variables : 141

View(santander2)

############################################
#### Visualizations 
###########################################

Input_Column_numbers <- function(n)
{
transparentTheme(trans = 0.5)
featurePlot(x = santander2[,1:n],
            y = santander2$TARGET,
            plot = "pairs"
            )
}
### Correlation between target variable and Predictors; 
### here have taken first 3 variables, 
### number of variables can be changed 

Input_Column_numbers(n = 3)

##########################################
# Important variables selection 
##########################################

Imp_variable_scoring <- filterVarImp(x = santander2[, -ncol(santander2)], y = santander2$TARGET)
top_50_imp_variables <- tail(Imp_variable_scoring,50) # selected top 50 variables based on the scoring

plot(Imp_variable_scoring,50) # plotting top 50 variables on the graph
getwd()
write.csv(Imp_variable_scoring,"top.csv") #exporting csv for the reference 

###########################################################
# Splitting data into train and validation data set : 80-20
######################  #####################################

partition <- createDataPartition(santander2$TARGET, p = 0.8,list = FALSE, times = 1)

train <- samplesan[partition,]
test <- samplesan[-partition,]
summary(train$TARGET)
nrow(train)  # Number of Records = 16846
nrow(test)   # Number of Records = 4210

###########################################################
# Forming a new dataset on train consisting only top 50 influencing variables
###########################################################
trainnew <- train[,c("var15",
                     "imp_trans_var37_ult1",
                     "num_var43_recib_ult1",
                     "var36",
                     "ind_var43_recib_ult1",
                     "num_var45_ult1",
                     "imp_aport_var13_ult1",
                     "saldo_var37",
                     "num_var37",
                     "num_var43_emit_ult1",
                     "ind_var37_cte",
                     "num_trasp_var11_ult1",
                     "imp_var43_emit_ult1",
                     "ind_var43_emit_ult1",
                     "num_op_var41_hace2",
                     "num_op_var41_comer_ult1",
                     "saldo_medio_var13_corto_ult1",
                     "imp_op_var41_comer_ult3",
                     "num_var13_corto",
                     "imp_op_var41_comer_ult1",
                     "num_meses_var13_corto_ult3",
                     "imp_op_var41_ult1",
                     "num_var45_hace2",
                     "saldo_var13",
                     "num_var22_hace2",
                     "num_var30_0",
                     "num_var8_0",
                     "num_var45_hace3",
                     "num_meses_var8_ult3",
                     "saldo_medio_var8_ult1",
                     "imp_op_var41_efect_ult3",
                     "imp_op_var41_efect_ult1",
                     "saldo_var25",
                     "num_var25",
                     "ind_var25",
                     "num_op_var41_efect_ult1",
                     "saldo_medio_var8_hace2",
                     "imp_ent_var16_ult1",
                     "num_ent_var16_ult1",
                     "num_var22_ult1",
                     "saldo_medio_var8_hace3",
                     "ind_var30_0",
                     "imp_op_var40_efect_ult3",
                     "num_op_var40_efect_ult3",
                     "num_op_var40_efect_ult1",
                     "num_var39",
                     "var21",
                     "num_var40_0",
                     "num_var37_med_ult2",
                     "imp_sal_var16_ult1",
                     "TARGET"
)]

View(trainnew)

# After considering top 50 variables, using Recursive Feature Elimination (RFE) technique to reduce the number of variables 

Subset <- c(1:8, 10, 15, 20, 25, 30)

control<-rfeControl(functions=rfFuncs,method="cv",number=10)
results<-rfe(trainnew[,1:50],trainnew[,51],sizes=Subset,rfeControl = control)
results
################# top 5 variable ###############
# The top 5 variables (out of 50):
#   var15, var36, num_var45_hace2, num_var45_hace3, num_var30_0

#The predictors function can be used to get a text string of variable names that were picked in the final model. 

predictors(results)

# The lmProfile is a list of class "rfe" that contains an object fit that is the final linear model with the remaining terms. 
# The model can be used to get predictions for future or test samples.

results$fit

# randomForest(x = x, y = y, importance = first) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 5
# 
# OOB estimate of  error rate: 14.13%
# Confusion matrix:
#   0    1 class.error
# 0 8635  991   0.1029503
# 1 1389 5831   0.1923823

head(results$resample)

# to plot the same
trellis.par.set(caretTheme())
plot(results, type = c("g", "o"))


## -----------------------------------------------------------

#############################
## Building Models 
#############################

################################### First Model: Decision Tree #######################

library(rpart)
dt <- rpart(train$TARGET ~ var15 + var36 + num_var45_hace2 + num_var45_hace3 + num_var30_0 + num_var45_ult1
            + ind_var43_recib_ult1 + ind_var30_0 + num_var22_hace2 + imp_trans_var37_ult1
            + num_var22_ult1 + imp_op_var41_comer_ult3 + ind_var37_cte 
            + imp_op_var41_efect_ult3 + num_var43_recib_ult1 
            + num_var13_corto + num_op_var41_hace2 + imp_op_var41_ult1 
            + num_trasp_var11_ult1 + imp_aport_var13_ult1 
            + num_op_var41_comer_ult1 + num_var8_0 + num_meses_var13_corto_ult3
            + imp_op_var41_comer_ult1 + num_var37
            , data = train, method = "class")
predictmodel = predict(dt,test,type = "class")
confusionMatrix1 = table(test$TARGET,predictmodel)
confusionMatrix1
Accuracy1 <- (confusionMatrix1[1,1] + confusionMatrix1[2,2])/sum(confusionMatrix1)
Accuracy1

# Accuracy: 0.8016

## Tuned DT using train function from caret package and used the complexity parameter cp to further tune the decisiontree
cvCtrl <- trainControl(method = "repeatedcv", repeats = 3)
train(TARGET ~ ., data = trainnew, method = "rpart", tuneLength = 30, trControl = cvCtrl)

##cp = 0.0004847645

tuneddt <- rpart(train$TARGET ~ var15 + var36 + num_var45_hace2 + num_var45_hace3 + num_var30_0 + num_var45_ult1
                 + ind_var43_recib_ult1 + ind_var30_0 + num_var22_hace2 + imp_trans_var37_ult1
                 + num_var22_ult1 + imp_op_var41_comer_ult3 + ind_var37_cte 
                 + imp_op_var41_efect_ult3 + num_var43_recib_ult1 
                 + num_var13_corto + num_op_var41_hace2 + imp_op_var41_ult1 
                 + num_trasp_var11_ult1 + imp_aport_var13_ult1 
                 + num_op_var41_comer_ult1 + num_var8_0 + num_meses_var13_corto_ult3
                 + imp_op_var41_comer_ult1 + num_var37, data = train, method = "class", cp = 0.0004847645)
tunedpredictmodel = predict(tuneddt,test,type = "class")
confusionMatrix2 = table(test$TARGET,tunedpredictmodel)
confusionMatrix2
Accuracy2 <- (confusionMatrix2[1,1] + confusionMatrix2[2,2])/sum(confusionMatrix2)
Accuracy2

## Updated Accuracy with CP factor:  0.8467933

######################################################
# Implementing Random Forest using 25 most important variables
#####################################################

library(randomForest)
rf = randomForest(train$TARGET~var15 + var36 + num_var45_hace2 + num_var45_hace3 + num_var30_0 + num_var45_ult1
                  + ind_var43_recib_ult1 + ind_var30_0 + num_var22_hace2 + imp_trans_var37_ult1
                  + num_var22_ult1 + imp_op_var41_comer_ult3 + ind_var37_cte 
                  + imp_op_var41_efect_ult3 + num_var43_recib_ult1 
                  + num_var13_corto + num_op_var41_hace2 + imp_op_var41_ult1 
                  + num_trasp_var11_ult1 + imp_aport_var13_ult1 
                  + num_op_var41_comer_ult1 + num_var8_0 + num_meses_var13_corto_ult3
                  + imp_op_var41_comer_ult1 + num_var37, data=train, ntree=100, nodesize=25, importance=TRUE)
predictmodel = predict(rf,test,type = "class")
confusionMatrix3 = table(test$TARGET,predictmodel)
confusionMatrix3
Accuracy3 <- (confusionMatrix3[1,1] + confusionMatrix3[2,2])/sum(confusionMatrix3)
Accuracy3 
#Accuracy = 0.8503563

################################
##using train function to tune further the randomforest mtry=26
#########################################

rf_model<-train(TARGET~.,data=trainnew,method="rf",
                trControl=trainControl(method="repeatedcv",number=3))
rf_model
##using mtry=26 in the randomforest model
tunedrf = randomForest(train$TARGET~var15 + var36 + num_var45_hace2 + num_var45_hace3 + num_var30_0 + num_var45_ult1
                       + ind_var43_recib_ult1 + ind_var30_0 + num_var22_hace2 + imp_trans_var37_ult1
                       + num_var22_ult1 + imp_op_var41_comer_ult3 + ind_var37_cte 
                       + imp_op_var41_efect_ult3 + num_var43_recib_ult1 
                       + num_var13_corto + num_op_var41_hace2 + imp_op_var41_ult1 
                       + num_trasp_var11_ult1 + imp_aport_var13_ult1 
                       + num_op_var41_comer_ult1 + num_var8_0 + num_meses_var13_corto_ult3
                       + imp_op_var41_comer_ult1 + num_var37, data=train, ntree=100, nodesize=25, importance=TRUE, mytry=26)
predictmodel = predict(rf,test,type = "class")
confusionMatrix4 = table(test$TARGET,predictmodel)
confusionMatrix4
Accuracy4 <- (confusionMatrix4[1,1] + confusionMatrix4[2,2])/sum(confusionMatrix4)
Accuracy4

# Accuracy after tuning the model: 0.8448
