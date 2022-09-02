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

#################################
#                               #
#         READ DATA             #
#                               #    
#################################

titanic.train <- read.csv("../data/titanic/train.csv")
titanic.train <- transform_age_class(titanic.train, 16)
titanic.test <- read.csv("../data/titanic/test.csv")
titanic.test <- transform_age_class(titanic.test, 16)

#################################
#                               #
#       COMPUTATIONS            #
#                               #    
#################################

samples <- partitionData(titanic.train)
titanic.train.train <- titanic.train[samples,]
titanic.train.test <- titanic.train[-samples,]

titanic.lr <- glm(Survived~Pclass+Sex+Age, data=titanic.train, family=binomial(link = "logit"))

#################################
#                               #
#       PREDICTIONS             #
#                               #    
#################################

p <- predict(titanic.lr, newdata = titanic.train, type = "response")
cutoff <- find_cutoff(p, titanic.train$Survived)
p <- predict(titanic.lr, newdata = titanic.test, type = "response")

kaggle <- data.frame(PassengerId = titanic.test$PassengerId,
                     Survived = ifelse(p>cutoff, 1, 0))
write.csv(kaggle, file = "../outputs/titanic_submission_logit3.csv", row.names = FALSE)