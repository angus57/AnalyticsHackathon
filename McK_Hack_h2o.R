library(h2o)

h2o.init()

setwd("C:/Users/along/Documents/AJ Local/McKinsey")

# Import data for h20
trainPath <- "train_wo_impute.csv"
train <- h2o.uploadFile(path = trainPath, destination_frame = "train")

# Update variables to factors
train[,17] <- as.factor(train[,17])
train[,18] <- as.factor(train[,18])
train[,19] <- as.factor(train[,19])

h2o.getTypes(train)


testPath <- "Test Dataset.h2o.csv"
test <- h2o.uploadFile(path = testPath, destination_frame = "test")

# Update variables to factors
test[,2] <- as.factor(test[,2])
test[,9] <- as.factor(test[,9])
test[,11] <- as.factor(test[,11])

# Identify predictors and response

y <- c("renewal","id")
x <- setdiff(names(train), y)
y <- "renewal"

# For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
#test[,y] <- as.factor(test[,y])

aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_runtime_secs = 600,
                  seed=123,
                  stopping_rounds = 5,
                  balance_classes = TRUE,
                  max_after_balance_size = 3
                  
                )

# View the AutoML Leaderboard
lb <- aml@leaderboard
lb


#Tuning the GBM Model
h2o.table(train[,y])
rate_per_class_list <- c(.1, 1)

McK_gbm <- h2o.gbm(x = x,
        y = y, 
        training_frame = train,
        ntrees = 5000,
        nfolds = 5,
        #max_depth = 15,
        stopping_rounds = 5,
        stopping_tolerance = 1e-3, 
        stopping_metric = "AUC", 
        #score_tree_interval = 10,
        ## smaller learning rate is better
        ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
        learn_rate = 0.01,                                                         
        
        ## learning rate annealing: learning_rate shrinks by 1% after every tree 
        ## (use 1.00 to disable, but then lower the learning_rate)
        #learn_rate_annealing = 0.99,
        sample_rate_per_class = rate_per_class_list,
        seed = 123)


print(h2o.auc(McK_gbm, xval = TRUE))
McK_gbm

# The leader model is stored here
aml@leader

h2o.gainsLift(aml@leader, xval = TRUE)

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

pred <- h2o.predict(aml, test)  # predict(aml, test) also works
predtrain <- h2o.predict(aml, train)

train_predictions <- h2o.cbind(train, predtrain)
write.csv(as.data.frame(train_predictions), file = 'h2o.trainpredictions.csv')

test_predictions <- h2o.cbind(test, pred)
write.csv(as.data.frame(test_predictions), file = 'h2o.testpredictions.csv')

# or:
pred <- h2o.predict(aml@leader, test)