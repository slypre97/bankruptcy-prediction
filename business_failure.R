#This is the prediction of business failure
##done by me you know who
##skriiii-paaa lets do it

#step 1 data preprocessing 
#dataset is of this way
#--------------------------------------------------------------------------------------------------------------------------------------
#- Size
#    - Sales
#- Profit
#    - ROCE: profit before tax=capital employed (%)
#    - FFTL: funds flow (earnings before interest, tax & depreciation)=total liabilities
#- Gearing
#    - GEAR: (current liabilities + long-term debt)=total assets
#    - CLTA: current liabilities=total assets
#- Liquidity
#    - CACL: current assets=current liabilities
#    - QACL: (current assets â€“ stock)=current liabilities
#    - WCTA: (current assets â€“ current liabilities)=total assets
#- LAG: number of days between account year end and the date the annual report and accounts were filed at company registry.
#- AGE: number of years the company has been operating since incorporation date.
#- CHAUD: coded 1 if changed auditor in previous three years, 0 otherwise
#- BIG6: coded 1 if company auditor is a Big6 auditor, 0 otherwise

#-------------------------------------------------------------------------------------------------------------------------------------------

####kkkkkkkk in this  dataset  you will see my magic

#first we start by loading dataset
business_failure <- read.csv("bankruptcy.csv", header = TRUE)
View(business_failure)

#summary of data
NROW(business_failure)

#lets take a look at the bankruptcy rate
table(business_failure$FAIL)

#yes we have 60 companies with only 30 having failed

#Structure of our data
str(business_failure)

#devide our data into 2 train and test
set.seed(3)
id <- sample(2,nrow(business_failure),prob = c(0.6,0.4), replace = TRUE)
train <- business_failure[id==1,]
test <- business_failure[id==2,]
View(train)
View(test)

#change variable fail into a factor
business_failure$FAIL <- as.factor(business_failure$FAIL)

#lets visualize our data
attach(business_failure)
library(ggplot2)

summary(SALES)
boxplot(SALES)


#Exploratory modelling1
# lets start with SVM
#we believe that sales is our hypothesis
library(caret)

trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
trctrl
svm_Lineaar <- train(FAIL~., data = train[c("FAIL", "SALES")], method = "svmLinear", trControl=trctrl,
                      #preProcess=c("range","scale"),
                      tuneLength=10)
svm_Lineaar

trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
svm_Lineaar2 <- train(FAIL~., data = train[c("FAIL", "SALES","AGE")], method = "svmLinear", trControl=trctrl,
                     #preProcess=c("range","scale"),
                     tuneLength=10)
svm_Lineaar2

trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
svm_Lineaar3 <- train(FAIL~., data = train[c("FAIL", "SALES","AGE","ROCE","FFTL")], method = "svmLinear", trControl=trctrl,
                      #preProcess=c("range","scale"),
                      tuneLength=10)
svm_Lineaar3

trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
svm_Lineaar4 <- train(FAIL~., data = train[c("FAIL", "SALES","AGE","ROCE","FFTL","CLTA","GEAR")], method = "svmLinear", trControl=trctrl,
                      #preProcess=c("range","scale"),
                      tuneLength=10)
svm_Lineaar4

#i believe that we have already done our filtration feature selection already so we should predict with all the data
train$FAIL <- as.factor(train$FAIL)
str(train$FAIL)
trctrl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
svm_Lineaar5 <- train(FAIL~., data = train[c("FAIL", "SALES","AGE","ROCE","FFTL","CLTA","GEAR","CHAUD","CO","CR","FF","CACL","QACL","WCTA","LAG","BIG6")], method = "svmLinear", trControl=trctrl,
                           #preProcess=c("range","scale"),
                           #tuneGrid = grid,
                           tuneLength=10)
svm_Lineaar5
#plot(test_pred)

test_pred <- predict(svm_Lineaar5, newdata = test)
test_pred

test$FAIL <- as.factor(test$FAIL)
str(test$FAIL)

confusionMatrix(test_pred, test$FAIL )

grid <- expand.grid(C = c(0,0.01,1.05,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5))
svm_Lineaar5.grid <- train(FAIL~., data = train[c("FAIL", "SALES","AGE","ROCE","FFTL","CLTA","GEAR","CHAUD","CO","CR","FF","CACL","QACL","WCTA","LAG","BIG6")], method = "svmLinear", trControl=trctrl,
                      #preProcess=c("range","scale"),
                      tuneGrid = grid,
                      tuneLength=10)
svm_Lineaar5.grid
plot(svm_Lineaar5.grid)

test_predresult <- as.data.frame(test_pred)
colnames(test_predresult) <- "SVM_failresult"
View(test_predresult)

#exploratory modelling2
#lets go on to the ann
library(neural)
library(neuralnet)
library(nnet)

bf_n <- neuralnet(FAIL~ SALES + AGE+ROCE+FFTL+CLTA+GEAR+CHAUD+CO+CR+FF+CACL+QACL+WCTA+LAG+BIG6,
                  data = train,
                  hidden = 3,
                  threshold = 0.01,
                  stepmax = 2000,
                  linear.output = FALSE,
                  lifesign = "full",
                  lifesign.step = 10,
                  err.fct = "sse",
                  rep = 1
)
plot(bf_n)

oldweights <- bf_n$weights

bf_n2 <- neuralnet(FAIL~ SALES + AGE+ROCE+FFTL+CLTA+GEAR+CHAUD+CO+CR+FF+CACL+QACL+WCTA+LAG+BIG6,
                  data = train,
                  hidden = 3,
                  threshold = 0.01,
                  stepmax = 2000,
                  linear.output = FALSE,
                  lifesign = "full",
                  lifesign.step = 10,
                  err.fct = "sse",
                  startweights = oldweights
)
plot(bf_n2)

oldweights1 <- bf_n2$weights

bf_n3 <- neuralnet(FAIL~ SALES + AGE+ROCE+FFTL+CLTA+GEAR+CHAUD+CO+CR+FF+CACL+QACL+WCTA+LAG+BIG6,
                   data = train,
                   hidden = 3,
                   threshold = 0.01,
                   stepmax = 2000,
                   linear.output = FALSE,
                   lifesign = "full",
                   lifesign.step = 10,
                   err.fct = "sse",
                   startweights = oldweights1
)
plot(bf_n3)

oldweights2 <- bf_n3$weights

bf_n4 <- neuralnet(FAIL~ SALES + AGE+ROCE+FFTL+CLTA+GEAR+CHAUD+CO+CR+FF+CACL+QACL+WCTA+LAG+BIG6,
                   data = train,
                   hidden = 3,
                   threshold = 0.01,
                   stepmax = 2000,
                   linear.output = FALSE,
                   lifesign = "full",
                   lifesign.step = 10,
                   err.fct = "sse",
                   startweights = oldweights2
)
plot(bf_n4)

oldweights3 <- bf_n4$weights

bf_n5 <- neuralnet(FAIL~ SALES + AGE+ROCE+FFTL+CLTA+GEAR+CHAUD+CO+CR+FF+CACL+QACL+WCTA+LAG+BIG6,
                   data = train,
                   hidden = 3,
                   threshold = 0.01,
                   stepmax = 2000,
                   linear.output = FALSE,
                   lifesign = "full",
                   lifesign.step = 10,
                   err.fct = "sse",
                   startweights = oldweights3
)
plot(bf_n5)


#str(test$FAIL)

bf_pred <- compute(bf_n4, scale(test[,2:16]))

test_predresult2 <- as.data.frame(bf_pred$net.result)
#colnames(test_predresult2) <- C("fail2")
View(test_predresult2)

#result_bf <- as.data.frame(test_predresult2[apply(test_predresult2, 1, which.is.max)])
#colnames(result_bf) <- c("result_bf")
#View(result_bf)

results <- data.frame(actual = test$FAIL, prediction = test_predresult2)
colnames(test_predresult2) <- c("prediction")
roundedresults1 <- sapply(results,round,digits =0)
roundedresults1 <- data.frame(roundedresults1)
attach(roundedresults1)
table(test$FAIL,prediction)

test_predresult3 <- sapply(test_predresult2,round,digits =0)
colnames(test_predresult3) <- c("ANN_failresult")
View(test_predresult3)

#confusionMatrix(result_bf, test$FAIL )
#okay now we have our results we go on to the ensemble method
#first we need to calculate mean of fail1 and fail2
library(dplyr)

#we have to create a table with 3 columns fail1 and fail3 and average
dt <- data.frame(test_predresult,test_predresult3)
View(dt)
fail <- write.csv(dt, file = "fail", row.names = TRUE)

#test_predresult <- as.numeric_version(test_predresult)
##dt <- as.numeric_version(dt)
#test_predresult <- !is.factor(test_predresult)

enm <- dt %>% mutate(Avg=rowMeans(cbind(dt), na.rm=T))
View(enm)

enm <- dt %>% rowwise() %>% mutate(Avg=mean(c(dt), na.rm=T)) 


table(test$FAIL,enm$Avg)

