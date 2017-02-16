library(data.table)
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(gbm)

# Read the data
train=fread("train.csv")
train$index="train"
test=fread("test.csv")
test$index="test"

#Create a target variable in test set
test$Purchase=NA

# Combine training and test data for pre-processing
dfCombined=rbind(train, test)

#Feature engineering
dfCombined$Stay_In_Current_City_Years=as.character(dfCombined$Stay_In_Current_City_Years)
dfCombined$Stay_In_Current_City_Years[dfCombined$Stay_In_Current_City_Years=="4+"]="4"
dfCombined$Stay_In_Current_City_Years=factor(dfCombined$Stay_In_Current_City_Years)
dfCombined$Product_Category_1=factor(dfCombined$Product_Category_1)
dfCombined$Product_Category_2=factor(dfCombined$Product_Category_2)
dfCombined$Product_Category_3=factor(dfCombined$Product_Category_3)
dfCombined$Occupation=as.factor(dfCombined$Occupation)

#Visualize trends in data
ggplot(dfCombined, aes(x=Age, fill=Gender)) + geom_bar(position = "dodge")
ggplot(dfCombined, aes(x=Product_Category_1, fill=Gender)) + geom_bar(position = "dodge")
ggplot(dfCombined, aes(x=Product_Category_2, fill=Gender)) + geom_bar(position = "dodge")
ggplot(dfCombined, aes(x=Product_Category_3, fill=Gender)) + geom_bar(position = "dodge")


#***************************************************
#*************Feature Creation**********************
#***************************************************

#Check NA
sum(is.na(dfCombined[,1:8])) ##no NA in first 8 columns

##Can high spenders be identified based on occupation?
ggplot(dfCombined, aes(x=Occupation)) + geom_bar()
tapply(dfCombined$Purchase, dfCombined$Occupation, function(x) median(x, na.rm=T))
dfCombined$HighSpenders="N"
dfCombined$HighSpenders[dfCombined$Occupation %in% c("8", "12", "15", "17")]="Y"
ggplot(dfCombined, aes(HighSpenders)) + geom_bar()
tapply(dfCombined$Purchase, dfCombined$HighSpenders, function(x) median(x, na.rm=T))

###########Age vs Purchase

tapply(dfCombined$Purchase[dfCombined$index=="train"], dfCombined$Age[dfCombined$index=="train"], mean)

############city Category vs Purchase
tapply(dfCombined$Purchase[dfCombined$index=="train"], dfCombined$City_Category[dfCombined$index=="train"], median) ##city C spends higher than A and B
ggplot(dfCombined, aes(x=City_Category, fill=HighSpenders)) + geom_bar(position = "dodge")

####stay period in current city vs Purchase
tapply(dfCombined$Purchase[dfCombined$index=="train"], dfCombined$Stay_In_Current_City_Years[dfCombined$index=="train"], median)
ggplot(dfCombined, aes(x=City_Category, fill=Stay_In_Current_City_Years)) + geom_bar(position = "dodge")

###Marital Status vs Purchase
tapply(dfCombined$Purchase[dfCombined$index=="train"], dfCombined$Marital_Status[dfCombined$index=="train"], mean)

###check product Categories vs purchase
plot(table(dfCombined$Product_Category_1))
round(prop.table(table(dfCombined$Product_Category_1))*100)
barplot(tapply(dfCombined$Purchase[dfCombined$index=="train"], dfCombined$Product_Category_1[dfCombined$index=="train"], median))

####Clean product categories
dfCombined$Product_Category_2=as.character(dfCombined$Product_Category_2)
dfCombined$Product_Category_2[is.na(dfCombined$Product_Category_2)]="0"
dfCombined$Product_Category_2=factor(dfCombined$Product_Category_2)

dfCombined$Product_Category_3=as.character(dfCombined$Product_Category_3)
dfCombined$Product_Category_3[is.na(dfCombined$Product_Category_3)]="0"
dfCombined$Product_Category_3=factor(dfCombined$Product_Category_3)

## Create dummy variables
dvProductCat=dummyVars(~Age+Marital_Status+ HighSpenders+ Gender+Stay_In_Current_City_Years+ Product_Category_1+Product_Category_2+Product_Category_3, data = dfCombined)
dvProductCatData=predict(dvProductCat, newdata = dfCombined)

##Check principal componenets
pcaComp=prcomp(x = dvProductCatData)
pctVar=pcaComp$sdev^2/sum(pcaComp$sdev^2)*100
cumsum(pctVar)
pcaData= pcaComp$x[,1:34]
pcaData %>% colnames()

#### save files
write.csv(pcaData, "pcaData.csv", row.names = F)
write.csv(dfCombined, "dfCombined.csv", row.names = F)


####prepare data for modeling
modelData=NULL
modelData$User_ID=dfCombined$User_ID
modelData$Product_ID=dfCombined$Product_ID
modelData$Purchase=dfCombined$Purchase
modelData$index=dfCombined$index

modelData=as.data.frame(modelData)
modelData=cbind(modelData, pcaData) #combine with PCA data

modelDataTrain=NULL
modelDataTest=NULL

# subset tarining and testing data
modelDataTrain=modelData[modelData$index=="train", ]
modelDataTest=modelData[modelData$index=="test", ]

modelDataTrain=as.data.frame(modelDataTrain)
modelDataTest=as.data.frame(modelDataTest)

##save files
write.csv(modelData, "modelData.csv", row.names = F)
write.csv(modelDataTrain, "modelDataTrain.csv", row.names = F)
write.csv(modelDataTest, "modelDataTest.csv", row.names = F)

################## apply modelling####################
######################################################
####################################################

####model 1 - normal tree ---

sum(is.na(modelDataTrain))

modelDataTrainRpart=modelDataTrain
modelDataTrainRpart$User_ID=NULL
modelDataTrainRpart$Product_ID=NULL
modelDataTrainRpart$index=NULL

modelDataTestRpart=modelDataTest
modelDataTestRpart$User_ID=NULL
modelDataTestRpart$Product_ID=NULL
modelDataTestRpart$Purchase=NULL
modelDataTestRpart$index=NULL

rp1=rpart(Purchase~ . , data = modelDataTrainRpart)
#make predictions
predictRp1=predict(rp1, newdata=modelDataTestRpart)
modelDataTestRpart$Purchase = predictRp1
# create submission file
submission1=data.frame(modelDataTest$User_ID, modelDataTest$Product_ID, modelDataTestRpart$Purchase)
submission1 %>% head() %>% View()
names(x = submission1) = c("User_ID", "Product_ID", "Purchase")
write.csv(submission1 , "submission1.csv", row.names = F)

####random forest

dfTrain$Gender=factor(dfTrain$Gender)
dfTrain$Age=factor(dfTrain$Age)
dfTrain$City_Category=factor(dfTrain$City_Category)
dfTrain$Stay_In_Current_City_Years=as.integer(dfTrain$Stay_In_Current_City_Years)
dfTrain$Product_Category_1=as.integer(dfTrain$Product_Category_1)
dfTrain$Product_Category_2=as.integer(dfTrain$Product_Category_2)
dfTrain$Product_Category_3=as.integer(dfTrain$Product_Category_3)

dfTest$Gender=factor(dfTest$Gender)
dfTest$Age=factor(dfTest$Age)
dfTest$City_Category=factor(dfTest$City_Category)
dfTest$Stay_In_Current_City_Years=as.integer(dfTest$Stay_In_Current_City_Years)
dfTest$Product_Category_1=as.integer(dfTest$Product_Category_1)
dfTest$Product_Category_2=as.integer(dfTest$Product_Category_2)
dfTest$Product_Category_3=as.integer(dfTest$Product_Category_3)

dfTrain1=dfTrain[1:200000,]
dfTrain2=dfTrain[100000:200000,]

library(randomForest)

rf1=randomForest(Purchase ~ Gender + Age + Occupation + City_Category + Stay_In_Current_City_Years + Marital_Status + Product_Category_1 + Product_Category_2 + Product_Category_3, data = dfTrain1)

dfTest$PredictRf1 = predict(rf1, newdata = dfTest)

#### gbm
library(gbm)

#set cross validation parameters
trainControl=trainControl(method = "CV", number = 5)
# fit model
gbm1=gbm(Purchase ~ Expensive+Gender + Age + Occupation + City_Category + Stay_In_Current_City_Years + Marital_Status + Product_Category_1 + Product_Category_2 + Product_Category_3, data = dfTrain1, distribution = "gaussian")

# fit model using caret
gbmfit1=train(Purchase ~ Expensive+ Gender + Age + Occupation + City_Category + Stay_In_Current_City_Years + Marital_Status + Product_Category_1 + Product_Category_2 + Product_Category_3, data=dfTrain, method="gbm", 
                  trControl = trainControl)



# Make predictions
dfTest$PredictGBM1= predict(gbmfit1, newdata = dfTest )
sub5=data.frame(User_ID=dfTest$User_ID, Product_ID=dfTest$Product_ID, Purchase=dfTest$PredictGBM1)

#create submission file
write.csv(sub5, "sub5.csv", row.names = F)


######### Got poor results
######### Apply more feature engineering for gbm model
#########

#Categories

library(stringr)
dfCombined %>% head() %>% View()

dfCombined$Expensive=NA
dfCombined$Len=str_sub(dfCombined$Product_ID,-2)
dfCombined$Expensive[dfCombined$Len %in% c("53", "93")] = "Low"
dfCombined$Expensive[dfCombined$Len %in% c("36", "44", "45")] = "Med"
dfCombined$Expensive[dfCombined$Len %in% c("42")] = "High"
dfCombined$Freq=NA

Freq=dfCombined %>% group_by(User_ID) %>% summarise(n())
Spend = dfCombined %>% group_by(User_ID) %>% mutate(amt= mean(Purchase, na.rm=T))
Spend = data.frame(Spend$User_ID, Spend$amt)

summary(Spend$Spend.amt)
ggplot(Spend, aes(x=Spend.amt)) + geom_histogram(bins = 100)
 
Spend$Spend_Cat="Low"
Spend$Spend_Cat[Spend$Spend.amt > 8142 & Spend$Spend.amt < 10181] = "Med"
Spend$Spend_Cat[Spend$Spend.amt >= 10181] = "High"
Spend=unique(Spend)
names(Spend) = c("User_ID", "amt", "Spend_Cat")

spendFreq=merge(Spend, Freq, by= "User_ID")
spendFreq %>% head() %>% View
names(spendFreq) = c("User_ID", "amt", "Spend_Cat", "Freq")

dfCombined2=merge(dfCombined, spendFreq, by="User_ID", all.x = T)
dfCombined2 %>% head() %>% View
dfCombined2$Freq.x=NULL

dfCombined2$Gender=factor(dfCombined2$Gender)
dfCombined2$Age=factor(dfCombined2$Age)
dfCombined2$City_Category=factor(dfCombined2$City_Category)
dfCombined2$Stay_In_Current_City_Years=as.integer(dfCombined2$Stay_In_Current_City_Years)
dfCombined2$Product_Category_1=as.integer(dfCombined2$Product_Category_1)
dfCombined2$Product_Category_2=as.integer(dfCombined2$Product_Category_2)
dfCombined2$Product_Category_3=as.integer(dfCombined2$Product_Category_3)
dfCombined2$Spend_Cat=factor(dfCombined2$Spend_Cat)
dfCombined2$HighSpenders=factor(dfCombined2$HighSpenders)
dfCombined2$Expensive=factor(dfCombined2$Expensive)

dfTest2=dfCombined2[dfCombined2$index=="test",]
dfTrain2=dfCombined2[dfCombined2$index=="train",]

dfTest3=dfCombined3[dfCombined3$index=="test",]
dfTrain3=dfCombined3[dfCombined3$index=="train",]

###refit gbm

trainControl=trainControl(method = "CV")

gbmfit3=train(Purchase ~ prod_mean+amt+Freq.y+Spend_Cat+HighSpenders+Expensive+ Gender + Age + Occupation + City_Category + Stay_In_Current_City_Years + Marital_Status + Product_Category_1 + Product_Category_2 + Product_Category_3, data=dfTrain3, method="gbm", trControl = trainControl)

#make predictions
predictdata= predict(gbmfit3, newdata = dfTest3)
dfTest3$predict=predictdata

#write file for submission
sub7=data.frame(User_ID=dfTest3$User_ID, Product_ID=dfTest3$Product_ID, Purchase=dfTest3$PredictGBM3)
write.csv(sub7, "sub7.csv", row.names = F)