install.packages("pacman")  #Only if you don't already have pacman
library(pacman)
pacman::p_load(caret,rpart,tabplot,rpart.plot,ROCR,randomForest,ggplot2)

library(devtools)
install_github("mtennekes/tabplot")
library(tabplot)


file="data_sample.csv"
varnames = c("Cash", "Inventories", "Current Assets", "Tangible Assets", "Intangible Assets", "Total Assets",
             "Accounts Receivable", "Lands and Buildings", "Equity", "Accrual for Pension Liabilities",
             "Total Current Liabilities", "Total Longterm Liabilities", "Bank Debt", "Accounts Payable",
             "Sales", "Amortization Depreciation", "Interest Expenses", "EBIT", "Operating Income",
             "Net Income", "Increase Inventories", "Increase Liabilities", "Increase Cash", "Number Employees",
             "Solvency")

credit=read.csv(file, header = TRUE, sep = ",", col.names = varnames)
str(credit)
credit$Solvency = as.factor(credit$Solvency)
tableplot(credit, sortCol = 1) # Nice plot of the variables
trainIndex <- createDataPartition(credit$Solvency, p = .7, list = FALSE)  # Split data into train and test and ensure that we keep the distribution of the target Variable. i.e. "Creditability"
df_train <- credit[trainIndex,] 
df_test  <- credit[-trainIndex,]

# First Model: TREE 
tree_model <- rpart(Solvency ~ ., data = df_train, control = ("maxdepth = 100"))



prp(tree_model,type=2,extra=1) # Plots the tree

fit_tree <- predict(tree_model, newdata = df_test,type="prob")[,2]

pred = prediction( fit_tree, df_test$Solvency)
perf <- performance(pred, "tpr", "fpr")

perf_cart = data.frame(data= c(perf@x.values, perf@y.values))
colnames(perf_cart) = c("FPR", "TPR")
p1 = ggplot(perf_cart, aes(x = FPR, y = TPR)) +
  geom_line() +
  ylab("True positive rate" ) +
  xlab("False positive rate") +
  theme(
    panel.background = element_rect(fill = "transparent"), # bg of the panel
    plot.background = element_rect(fill = "transparent", color = NA), # bg of the plot
    panel.grid.major = element_blank(), # get rid of major grid
    panel.grid.minor = element_blank(), # get rid of minor grid
    legend.background = element_rect(fill = "transparent"), # get rid of legend bg
    legend.box.background = element_rect(fill = "transparent") # get rid of legend panel bg
  )
p1
ggsave(p1, filename = "ROC CART.png",  bg = "transparent")




AUCArbre=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCArbre,"\n")


# Second Model: Random Forest
RF <- randomForest(Solvency ~ .,data = df_train)
fit_RF <- predict(RF,newdata=df_test,type="prob")[,2]
pred = prediction( fit_RF, df_test$Solvency)
perf <- performance(pred, "tpr", "fpr")
perf_rf = data.frame(data= c(perf@x.values, perf@y.values))
colnames(perf_rf) = c("FPR", "TPR")
p2 = ggplot(perf_rf, aes(x = FPR, y = TPR)) +
  geom_line() +
  ylab("True positive rate" ) +
  xlab("False positive rate") +
  theme(
    panel.background = element_rect(fill = "transparent"), # bg of the panel
    plot.background = element_rect(fill = "transparent", color = NA), # bg of the plot
    panel.grid.major = element_blank(), # get rid of major grid
    panel.grid.minor = element_blank(), # get rid of minor grid
    legend.background = element_rect(fill = "transparent"), # get rid of legend bg
    legend.box.background = element_rect(fill = "transparent") # get rid of legend panel bg
  )
ggsave(p2, filename = "ROC RF.png",  bg = "transparent")

AUCArf=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCArf,"\n")




AUC = function(i) {
  set.seed(i)
  trainIndex <- createDataPartition(credit$Solvency, p = .7, list = FALSE) 
  df_train <- credit[trainIndex,] 
  df_test  <- credit[-trainIndex,]
  rpart <-rpart(Solvency ~ ., data = df_train)
  summary(rpart)
  fitrpart <- predict(rpart,newdata=df_test,type="prob")[,2]
  pred = prediction(fitrpart, df_test$Solvency)
  AUC_tree = performance(pred, measure = "auc")@y.values[[1]]
  RF <- randomForest(Solvency ~ .,
                     data = df_train)
  fitForet <-predict(RF, newdata = df_test, type = "prob")[, 2]
  pred = prediction(fitForet, df_test$Solvency)
  AUC_RF = performance(pred, measure = "auc")@y.values[[1]]
  return(c(AUC_tree, AUC_RF))
}
A = Vectorize(AUC)(1:200)

A2 <- as.data.frame(t(A))


p3 = ggplot(A2,aes(x = V1, y = V2)) +
  geom_point() +
  ylab("Random Forest (AUC)" ) +
  xlab("CART (AUC)") +
  xlim(0.7,1) +
  ylim(0.7,1) +
  geom_abline(mapping= aes(intercept=0.0,slope = 1.0), color = "red") +
  scale_colour_manual(values="red") +
  labs(colour="") +
  theme(
    panel.background = element_rect(fill = "transparent"), # bg of the panel
    plot.background = element_rect(fill = "transparent", color = NA), # bg of the plot
    panel.grid.major = element_blank(), # get rid of major grid
    panel.grid.minor = element_blank(), # get rid of minor grid
    legend.background = element_rect(fill = "transparent"), # get rid of legend bg
    legend.box.background = element_rect(fill = "transparent") # get rid of legend panel bg
  )

ggsave(p3, filename = "Resampling RF vs. CART.png",  bg = "transparent", width = 5, height = 5)









