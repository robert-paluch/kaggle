#################################
#                               #
#         FUNCTIONS             #
#                               #    
#################################

transform_age_class <- function(titanic.df, threshold){
  titanic.df$Age <- ifelse(titanic.df$Age>threshold, "Adult", "Child")
  titanic.df$Age[which(is.na(titanic.df$Age))] <- "Adult"
  titanic.df$Pclass <- as.factor(titanic.df$Pclass)
  return(titanic.df)
}

transform_to_factors <- function(titanic.df) {
  titanic.df$Pclass <- as.factor(titanic.df$Pclass)
  titanic.df$SibSp <- as.factor(titanic.df$SibSp)
  titanic.df$Parch <- as.factor(titanic.df$Parch)
  return(titanic.df)
}

partitionData <- function(data) {
  require(caret)
  samples <- createDataPartition(data$Survived, p = 0.8, list = FALSE)
  return(samples)
}

find_cutoff <- function(score, class) {
  require(ROCit)
  rocit.obj <- rocit(score,class)
  plot(rocit.obj)
  best.yi.index <- which.max(rocit.obj$TPR-rocit.obj$FPR)
  best.cutoff <- rocit.obj$Cutoff[best.yi.index]
  return(best.cutoff)
}

err.rate <- function(org.class, pred.class) {
  
  CM <- table(org.class, pred.class)
  
  return(1 - sum(diag(CM)) / sum(CM))
}

do.rf <- function(data, newdata, n) {
  require(randomForest)
  rf <- randomForest(Survived ~ Pclass+Sex+SibSp+Parch, data = data, ntree = n)
  rf.pred <- predict(rf, newdata = newdata)
  
  return(err.rate(newdata$Survived, rf.pred))
}

#################################
#                               #
#         READ DATA             #
#                               #    
#################################

titanic.train <- read.csv("../data/titanic/train.csv")
titanic.train$Survived <- as.factor(titanic.train$Survived)
titanic.train <- transform_to_factors(titanic.train)

titanic.test <- read.csv("../data/titanic/test.csv")
titanic.test <- transform_to_factors(titanic.test)


#################################
#                               #
#       COMPUTATIONS            #
#                               #    
#################################

samples <- partitionData(titanic.train)
titanic.train.train <- titanic.train[samples,]
titanic.train.test <- titanic.train[-samples,]

rf <- randomForest(Survived ~ Pclass+Sex+Age, data = titanic.train.train, ntree = 500)
rf.pred <- predict(rf, newdata = titanic.train.test)
rf.err <- err.rate(titanic.train.test$Survived, rf.pred)

titanic.lr <- glm(Survived~Pclass+Sex+Age, data=titanic.train, family=binomial(link = "logit"))

#################################
#                               #
#       PREDICTIONS             #
#                               #    
#################################


titanic.rf.pred <- predict(titanic.rf, newdata = titanic.train.test)

p <- predict(titanic.lr, newdata = titanic.train, type = "response")
cutoff <- find_cutoff(p, titanic.train$Survived)
p <- predict(titanic.lr, newdata = titanic.test, type = "response")

kaggle <- data.frame(PassengerId = titanic.test$PassengerId,
                     Survived = ifelse(p>cutoff, 1, 0))
write.csv(kaggle, file = "../outputs/titanic_submission_logit3.csv", row.names = FALSE)