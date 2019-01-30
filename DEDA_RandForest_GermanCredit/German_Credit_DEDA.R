install.packages("pacman")  #Only if you don't already have pacman
library(pacman)
pacman::p_load(caret,rpart,tabplot,rpart.plot,ROCR,randomForest,ggplot2)





url="http://freakonometrics.free.fr/german_credit.csv"
credit=read.csv(url, header = TRUE, sep = ",")
str(credit)
F=c(1,2,4,5,7,8,9,10,11,12,13,15,16,17,18,19,20) # Variables that are factors 
for(i in F) credit[,i]=as.factor(credit[,i])

tableplot(credit, sortCol = Creditability) # Nice plot of the variables
trainIndex <- createDataPartition(credit$Creditability, p = .7, list = FALSE)  # Split data into train and test and ensure that we keep the distribution of the target Variable. i.e. "Creditability"
df_train <- credit[trainIndex,] 
df_test  <- credit[-trainIndex,]

# First Model: TREE 
tree_model <- rpart(Creditability ~ ., data = df_train)



prp(tree_model,type=2,extra=1) # Plots the tree

fit_tree <- predict(tree_model,newdata=df_test,type="prob")[,2]

pred = prediction( fit_tree, df_test$Creditability)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCArbre=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCArbre,"\n")


# Seconde Model: Random Forest
RF <- randomForest(Creditability ~ .,data = df_train)
fit_RF <- predict(RF,newdata=df_test,type="prob")[,2]
pred = prediction( fit_RF, df_test$Creditability)
perf <- performance(pred, "tpr", "fpr")

plot(perf)






AUC = function(i) {
  set.seed(i)
  trainIndex <- createDataPartition(credit$Creditability, p = .7, list = FALSE) 
  df_train <- credit[trainIndex,] 
  df_test  <- credit[-trainIndex,]
  rpart <-rpart(Creditability ~ ., data = df_train)
  summary(rpart)
  fitrpart <- predict(rpart,newdata=df_test,type="prob")[,2]
  pred = prediction(fitrpart, df_test$Creditability)
  AUC_tree = performance(pred, measure = "auc")@y.values[[1]]
  RF <- randomForest(Creditability ~ .,
                     data = df_train)
  fitForet <-predict(RF, newdata = df_test, type = "prob")[, 2]
  pred = prediction(fitForet, df_test$Creditability)
  AUC_RF = performance(pred, measure = "auc")@y.values[[1]]
  return(c(AUC_tree, AUC_RF))
}
A = Vectorize(AUC)(1:200)

A2 <- as.data.frame(t(A))


ggplot(A2,aes(x = V1, y = V2)) +
  geom_point() +
  ylab("Random Forest (AUC)") +
  xlab("CART (AUC)") +
  xlim(0.6,1) +
  ylim(0.6,1) +
 geom_abline(mapping= aes(intercept=0.0,slope = 1.0, color="45 Degree line")) +
scale_colour_manual(values="red") +
  labs(colour="")

