###############################################
#                                             #
#         Data Mining Assignment              #
#         Student Name : Aausuman Deep        #
#         Student Number : 119220605          #
#                                             #
###############################################


# Clearing the environment
rm(list = ls())

# Clearing the console
cat("\014")

# Libraries needed for this assignment
library(caret)
library(e1071)
library(randomForest)
library(mlr)

# Location variable for data
local_path = "/Users/aausuman/Documents/College Stuff/Trimester 2/CS6405/Project/Newsgroups/"

# Creating a list of files
list_of_files <- list.files(path = local_path, recursive = TRUE)

###################################################################
#                                                                 #
#  Classification functions that will be used in our assignment   #
#                                                                 #
###################################################################

# Naive Bayes || Returns predictions performed by Naive Bayes Classifier
NaiveBayes = function(full_data, train_indices, train_data, test_data){
  model = naiveBayes(as.factor(Class_of_file)~., data = train_data)
  model_pred = predict(model, newdata = test_data)
  return(model_pred)
}

# KNN || Returns predictions performed by KNN Classifier
KNN = function(full_data, train_data, train_indices, test_data){
  task = makeClassifTask(data = full_data, target = "Class_of_file")
  learner = makeLearner("classif.knn", k = round(sqrt(nrow(train_data))))
  model = mlr::train(learner, task, subset = train_indices)
  model_pred = predict(model, newdata = test_data)
  return(model_pred)
}

# Random Forest || Returns predictions performed by Random Forest Classifier
RandomForest = function(train_data, test_data){
  model = randomForest(as.factor(Class_of_file)~., data = train_data)
  model_pred = predict(model, newdata = test_data, type = "class")
  return(model_pred)
}

# Decision Tree || Returns predictions performed by Decision Tree Classifier
DecisionTree = function(full_data, train_indices, test_data){
  task = makeClassifTask(data = full_data, target = "Class_of_file")
  learner = makeLearner("classif.rpart")
  model = mlr::train(learner, task, subset = train_indices)
  model_pred = predict(model, newdata = test_data)
  return(model_pred)
}

# Support Vector Machines || Returns predictions performed by Support Vector Machines Classifier
SupportVectorMachine = function(train_data, test_data, kernel_type){
  model = svm(as.factor(Class_of_file)~., data = train_data, kernel = kernel_type)
  model_pred = predict(model, newdata = test_data)
  return(model_pred)
}


############################################### 
#                                             #  
#     1. Without preprocessing the data       #
#                                             #
###############################################

# One single vector for storing all the unprocessed unique words in all files
repository = c()

# One single vector for storing the class of each file we go through
classes = c()

# Storing values in the two vectors created above
for(i in list_of_files){
  current_file_path = paste(local_path, i, sep="")
  current_file = file(current_file_path, open = 'r')
  current_file_class = stringi::stri_list2matrix(strsplit(i, "/"))[1]
  classes = append(classes, current_file_class)
  lines = readLines(current_file)
  for(j in lines){
    if(j != ""){
      words = stringi::stri_list2matrix(strsplit(j, " +"))
      # Due to system memory boundations, 
      # I am only reading the first two words of each line, rather than all the words of each line
      for(k in words[1:2]){
        if(!k %in% repository){
          repository = append(repository, k)
        }
      }
    }
  }
}

# Creating a Bag of words data frame and giving it column names using the repository just created
bag_of_words = data.frame()
repository = na.omit(repository)
for (k in repository){
  bag_of_words[[k]] <- as.numeric()
}

# Checking file by file for frequency counts of all our unique words
# And storing those frequencies in the bag of words data frame, row by row for each file
for(i in 1:length(list_of_files)){
  current_file_path = paste(local_path, list_of_files[i], sep="")
  current_file = file(current_file_path, open = 'r')
  lines = readLines(current_file)
  words_current_file = c()
  for(k in lines){
    if(k != ""){
      words = stringi::stri_list2matrix(strsplit(k, " +"))
      for(l in words){
        words_current_file = append(words_current_file, l)
      }
    }
  }
  observation = rep(0, each=length(repository))
  words_freq = table(words_current_file)
  words_freq = cbind.data.frame(names(words_freq),as.numeric(words_freq))
  for(m in 1:length(words_freq[,1])){
    index = which(words_freq[m,1] == repository)
    observation[index] = words_freq[m,2]
  }
  observation = as.data.frame(t(observation))
  names(observation) = repository
  bag_of_words = rbind(bag_of_words, observation)
}

# Adding the final column of classes for each of our 400 observations
# which represents each file, to our bag of words
bag_of_words[["Class_of_file"]] <- as.vector(classes)

#########################
#                       #
#   Exploration Task    #
#                       #
#########################

# Finding our top 200 most occuring words (features)

ncols = as.numeric(ncol(bag_of_words))
sum_of_occurences = rep(0, each=ncols-1)
for(i in 1:(ncols-1)){
  # Adding all frequencies per variable in the 400 observations
  sum_of_occurences[i] = sum(as.vector(bag_of_words[i]))   
}
# Representing those summed up frequencies as a data frame next to each respective word
words_with_occurences = cbind.data.frame(repository, as.numeric(sum_of_occurences))
sorted_words_with_occurences = words_with_occurences[order(words_with_occurences[,2], decreasing = TRUE),]
names(sorted_words_with_occurences) = c("Words","Total Occurences")

for(i in 1:200){
  print(sorted_words_with_occurences[i,])
}

# Finding our top 200 most occuring words (features) of length more than 4 and less than 20

sorted_words_with_occurences$Words_Length = nchar(as.character(sorted_words_with_occurences[,1]))
count = 1
for(i in 1:length(sorted_words_with_occurences[,1])){
  if(sorted_words_with_occurences[i,3] >= 4 && sorted_words_with_occurences[i,3] <= 20){
    print(sorted_words_with_occurences[i,])
    count = count + 1
    if(count == 201){ 
      break
    }
  }
}

# Because the words extracted by us uptil now have not been preprocessed
# Some of them will cause errors ahead as they do not follow R's naming conventions
# So we need to explicitly make them valid (by naming them as a default list of V's)
for(i in 1:length(repository)){
  names(bag_of_words)[i] = paste("V", i, sep = "")
}

#########################
#                       #
#   Basic Evaluation    #
#                       #
#########################

# Splitting our bag of words into training and testing set
n = nrow(bag_of_words)
set.seed(1)
i.train = sample(1:n, size = round(0.7*n), replace = FALSE)
bag_of_words_train = bag_of_words[i.train,]
bag_of_words_test = bag_of_words[-i.train,]

# Naive Bayes classification algorithm

predictions = NaiveBayes(bag_of_words, i.train, bag_of_words_train, bag_of_words_test)
confusion_matrix_nb = table(predictions, bag_of_words_test$Class_of_file)
recall_class1 = confusion_matrix_nb[1,1]/sum(confusion_matrix_nb[1,])
recall_class2 = confusion_matrix_nb[2,2]/sum(confusion_matrix_nb[2,])
recall_class3 = confusion_matrix_nb[3,3]/sum(confusion_matrix_nb[3,])
recall_class4 = confusion_matrix_nb[4,4]/sum(confusion_matrix_nb[4,])
accuracy_nb_1 = sum(diag(confusion_matrix_nb))/sum(confusion_matrix_nb)

# KNN classification algorithm

predictions = KNN(bag_of_words, bag_of_words_train, i.train, bag_of_words_test)
accuracy_knn_1 = performance(predictions, measures = acc)

# Random Forest classification algorithm

predictions = RandomForest(bag_of_words_train, bag_of_words_test)
confusion_matrix_rf = table(predictions, bag_of_words_test$Class_of_file)
recall_class1 = confusion_matrix_rf[1,1]/sum(confusion_matrix_rf[1,])
recall_class2 = confusion_matrix_rf[2,2]/sum(confusion_matrix_rf[2,])
recall_class3 = confusion_matrix_rf[3,3]/sum(confusion_matrix_rf[3,])
recall_class4 = confusion_matrix_rf[4,4]/sum(confusion_matrix_rf[4,])
accuracy_rf_1 = sum(diag(confusion_matrix_rf))/sum(confusion_matrix_rf)

# Accuracy of these three classifiers on our unprocessed data
cbind(accuracy_nb_1, accuracy_knn_1, accuracy_rf_1)


############################################### 
#                                             #  
#     2. With preprocessing the data          #
#                                             #
###############################################

# One single vector for storing all the processed unique words in all files
repository = c()

# One single vector for storing the class of each file we go through
classes = c()

# Storing values in the vector created above
for(i in list_of_files){
  current_file_path = paste(local_path, i, sep="")
  current_file = file(current_file_path, open = 'r')
  current_file_class = stringi::stri_list2matrix(strsplit(i, "/"))[1]
  classes = append(classes, current_file_class)
  lines = readLines(current_file)
  for(j in lines){
    if(j != ""){
      # not considering the punctuations and numbers as features for text processing
      j = gsub("[^[:alpha:] ]", " ", j)
      words = stringi::stri_list2matrix(strsplit(j, " +"))
      # Due to system memory boundations, 
      # I am only reading the first two words of each line, rather than all the words of each line
      for(k in words[1:2]){
        # converting all words to lower case
        k = as.character(lapply(k, tolower))
        if(!k %in% repository){
          repository = append(repository, k)
        }
      }
    }
  }
}

# Removing the most common stop words from the repository
stop_words = c("a", "and", "the", "to", "in", "that", "so")
repository = repository[!repository %in% stop_words]

# Removing all words with length less than 3
repository = na.omit(repository)
too_short = c()
for(i in repository){
  if(nchar(i) < 3){
    too_short = append(too_short, i)
  }
}
repository = repository[!repository %in% too_short]

# Creating a Bag of words data frame and giving it column names using the repository just created
bag_of_words = data.frame()
for (k in repository){
  bag_of_words[[k]] <- as.numeric()
}

# Checking file by file for frequency counts of all our unique words
# And storing those frequencies in the bag of words data frame, row by row for each file
for(i in 1:length(list_of_files)){
  current_file_path = paste(local_path, list_of_files[i], sep="")
  current_file = file(current_file_path, open = 'r')
  lines = readLines(current_file)
  words_current_file = c()
  for(k in lines){
    if(k != ""){
      k = gsub("[^[:alpha:] ]", " ", k)
      words = stringi::stri_list2matrix(strsplit(k, " +"))
      for(l in words){
        l = as.character(lapply(l, tolower))
        words_current_file = append(words_current_file, l)
      }
    }
  }
  observation = rep(0, each=length(repository))
  words_freq = table(words_current_file)
  words_freq = cbind.data.frame(names(words_freq),as.numeric(words_freq))
  for(m in 1:length(words_freq[,1])){
    index = which(words_freq[m,1] == repository)
    observation[index] = words_freq[m,2]
  }
  observation = as.data.frame(t(observation))
  names(observation) = repository
  bag_of_words = rbind(bag_of_words, observation)
}

# Adding the final column of classes for each of our 400 observations
# which represents each file, to our bag of words
bag_of_words[["Class_of_file"]] <- as.vector(classes)

#########################
#                       #
#   Exploration Task    #
#                       #
#########################

# Finding our top 200 most occuring words (features)

ncols = as.numeric(ncol(bag_of_words))
sum_of_occurences = rep(0, each=ncols-1)
for(i in 1:(ncols-1)){
  sum_of_occurences[i] = sum(as.vector(bag_of_words[i]))   
}
words_with_occurences = cbind.data.frame(repository, as.numeric(sum_of_occurences))
sorted_words_with_occurences = words_with_occurences[order(words_with_occurences[,2], decreasing = TRUE),]
names(sorted_words_with_occurences) = c("Words","Total Occurences")

for(i in 1:200){
  print(sorted_words_with_occurences[i,])
}

# Finding our top 200 most occuring words (features) of length more than 4 and less than 20

sorted_words_with_occurences$Words_Length = nchar(as.character(sorted_words_with_occurences[,1]))
count = 1
for(i in 1:length(sorted_words_with_occurences[,1])){
  if(sorted_words_with_occurences[i,3] >= 4 && sorted_words_with_occurences[i,3] <= 20){
    print(sorted_words_with_occurences[i,])
    count = count + 1
    if(count == 201){ 
      break
    }
  }
}

# Now, so as to have better and meaningful classification models in the following steps
# I am dropping all the variables (features) from our bag of words which
# have total occurence less than 10 across all 400 files, as they won't be that impactful
# for our classification.
irrelevant_features = c()
reverse_sorted_words_with_occurences = sorted_words_with_occurences[order(sorted_words_with_occurences[,2], decreasing = FALSE),]
for(i in 1:length(reverse_sorted_words_with_occurences[,1])){
  if(as.numeric(reverse_sorted_words_with_occurences[i,2]) > 9) break
  irrelevant_features[i] = as.character(reverse_sorted_words_with_occurences[i,1])
}

bag_of_words = bag_of_words[,!names(bag_of_words) %in% irrelevant_features]

# Making all variable names of this optimal bag of words valid
names(bag_of_words) = make.names(names(bag_of_words))

#######################
#                     #
#  Basic Evaluation   #
#                     #
#######################

# Splitting our bag of words into training and testing set
n = nrow(bag_of_words)
set.seed(1)
i.train = sample(1:n, size = round(0.7*n), replace = FALSE)
bag_of_words_train = bag_of_words[i.train,]
bag_of_words_test = bag_of_words[-i.train,]

# Naive Bayes classification algorithm

predictions = NaiveBayes(bag_of_words, i.train, bag_of_words_train, bag_of_words_test)
confusion_matrix_nb = table(predictions, bag_of_words_test$Class_of_file)
recall_class1 = confusion_matrix_nb[1,1]/sum(confusion_matrix_nb[1,])
recall_class2 = confusion_matrix_nb[2,2]/sum(confusion_matrix_nb[2,])
recall_class3 = confusion_matrix_nb[3,3]/sum(confusion_matrix_nb[3,])
recall_class4 = confusion_matrix_nb[4,4]/sum(confusion_matrix_nb[4,])
accuracy_nb_2 = sum(diag(confusion_matrix_nb))/sum(confusion_matrix_nb)

# KNN classification algorithm

predictions = KNN(bag_of_words, bag_of_words_train, i.train, bag_of_words_test)
accuracy_knn_2 = performance(predictions, measures = acc)

# Random Forest classification algorithm

predictions = RandomForest(bag_of_words_train, bag_of_words_test)
confusion_matrix_rf = table(predictions, bag_of_words_test$Class_of_file)
recall_class1 = confusion_matrix_rf[1,1]/sum(confusion_matrix_rf[1,])
recall_class2 = confusion_matrix_rf[2,2]/sum(confusion_matrix_rf[2,])
recall_class3 = confusion_matrix_rf[3,3]/sum(confusion_matrix_rf[3,])
recall_class4 = confusion_matrix_rf[4,4]/sum(confusion_matrix_rf[4,])
accuracy_rf_2 = sum(diag(confusion_matrix_rf))/sum(confusion_matrix_rf)

# Accuracy of these three classifiers on our preprocessed data
cbind(accuracy_nb_2, accuracy_knn_2, accuracy_rf_2)

# Comparing these three classifier accuracies on unprocessed vs preprocessed data
cbind(accuracy_nb_1, accuracy_nb_2)
cbind(accuracy_knn_1, accuracy_knn_2)
cbind(accuracy_rf_1, accuracy_rf_2)

##########################
#                        #
#    Robust Evaluation   #
#                        #
##########################

# Hold out and Cross Validation

# We have already "held-out" 30% of our data as testing set, and 70% data as training set
# We will perform Cross validation on the 70% training set and then validate that on the test set
# We will perform Cross validation on the Random Forest method

nr = nrow(bag_of_words_train)
nc = ncol(bag_of_words_train)

x = bag_of_words_train[,1:nc-1]
y = bag_of_words_train[,nc]

K = 20
folds = cut(1:nr, K, labels=FALSE)
err.kfold = numeric(K)
for(k in 1:K){
  # training sample
  i = which(folds==k)
  x.train = x[-i,]
  y.train = y[-i]
  # test sample
  x.test = x[i,]
  y.test = y[i]
  # train model on training sample:
  rf_model_cv = randomForest(as.factor(y.train)~., data = x.train)
  # generate prediction on testing sample:
  rf_model_cv_pred = predict(rf_model_cv, newdata = x.test)
  # compute prediction error:
  confusion_matrix_rf_cv = table(rf_model_cv_pred, y.test)
  accuracy_rf_cv = sum(diag(confusion_matrix_rf_cv))/sum(confusion_matrix_rf_cv)
  err.kfold[k] = mean( (1-accuracy_rf_cv)^2 )
}
rmse_train_set = median(sqrt(err.kfold))
# Now we fit the model on whole dataset and evaluate performance on held out testing set
rf_model_cv_final = randomForest(as.factor(y)~., data = x)
rf_model_cv_final_pred = predict(rf_model_cv_final, newdata=bag_of_words_test[,1:nc-1])
confusion_matrix_rf_cv_final = table(rf_model_cv_final_pred, bag_of_words_test[,nc])
accuracy_rf_cv_final = sum(diag(confusion_matrix_rf_cv_final))/sum(confusion_matrix_rf_cv_final)
final.mse = mean( (1-accuracy_rf_cv_final)^2 )
rmse_test_set = sqrt(final.mse)
cbind(rmse_train_set, rmse_test_set)
# The closer these two error rates are to each other, the better 

# Decision tree

predictions = DecisionTree(bag_of_words, i.train, bag_of_words_test)
accuracy_dctree = performance(predictions, measures = acc)

# Support Vector Machine

predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "linear")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmlinear = conf_matrix$overall[1]
conf_matrix$table
recall_class1 = conf_matrix$table[1,1]/sum(conf_matrix$table[1,])
recall_class2 = conf_matrix$table[2,2]/sum(conf_matrix$table[2,])
recall_class3 = conf_matrix$table[3,3]/sum(conf_matrix$table[3,])
recall_class4 = conf_matrix$table[4,4]/sum(conf_matrix$table[4,])

predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "polynomial")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmpolynomial = conf_matrix$overall[1]
conf_matrix$table
recall_class1 = conf_matrix$table[1,1]/sum(conf_matrix$table[1,])
recall_class2 = conf_matrix$table[2,2]/sum(conf_matrix$table[2,])
recall_class3 = conf_matrix$table[3,3]/sum(conf_matrix$table[3,])
recall_class4 = conf_matrix$table[4,4]/sum(conf_matrix$table[4,])

predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "radial")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmradial = conf_matrix$overall[1]
conf_matrix$table
recall_class1 = conf_matrix$table[1,1]/sum(conf_matrix$table[1,])
recall_class2 = conf_matrix$table[2,2]/sum(conf_matrix$table[2,])
recall_class3 = conf_matrix$table[3,3]/sum(conf_matrix$table[3,])
recall_class4 = conf_matrix$table[4,4]/sum(conf_matrix$table[4,])

# Feature selection using Recursive feature Elimination

set.seed(1)
subsets<- c(400,900, ncol(bag_of_words))
bag_of_words_train = na.omit(bag_of_words_train)
ctrl <- rfeControl(functions = rfFuncs, method = "cv",number = 10,verbose = FALSE)
rf.rfe <- rfe(bag_of_words_train[,1:nc-1], as.factor(bag_of_words_train$Class_of_file), sizes = subsets, rfeControl = ctrl)
imp_var=rf.rfe$optVariables

# Dropping all other variables from the bag of words except the ones considered important 
# by the above performed Recursive Feature Elimination, and the class variable
bag_of_words = bag_of_words[,names(bag_of_words) %in% c(imp_var, "Class_of_file")]

# Splitting our further reduced bag of words into training and testing set
n = nrow(bag_of_words)
set.seed(1)
i.train = sample(1:n, size = round(0.7*n), replace = FALSE)
bag_of_words_train = bag_of_words[i.train,]
bag_of_words_test = bag_of_words[-i.train,]

# Naive Bayes classification algorithm

predictions = NaiveBayes(bag_of_words, i.train, bag_of_words_train, bag_of_words_test)
statistics = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_nb_3 = statistics$overall[1]

# KNN classification algorithm

predictions = KNN(bag_of_words, bag_of_words_train, i.train, bag_of_words_test)
accuracy_knn_3 = performance(predictions, measures = acc)

# Random Forest classification algorithm

predictions = RandomForest(bag_of_words_train, bag_of_words_test)
confusion_matrix_rf = table(predictions, bag_of_words_test$Class_of_file)
accuracy_rf_3 = sum(diag(confusion_matrix_rf))/sum(confusion_matrix_rf)

# Decision tree

predictions = DecisionTree(bag_of_words, i.train, bag_of_words_test)
accuracy_dctree_2 = performance(predictions, measures = acc)

# Support Vector Machine

predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "linear")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmlinear_2 = conf_matrix$overall[1]
conf_matrix$table

predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "polynomial")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmpolynomial_2 = conf_matrix$overall[1]
conf_matrix$table
predictions = SupportVectorMachine(bag_of_words_train, bag_of_words_test, "radial")
conf_matrix = confusionMatrix(as.factor(bag_of_words_test$Class_of_file), predictions)
accuracy_svmradial_2 = conf_matrix$overall[1]
conf_matrix$table

# Comparing results with accuracies of preprocessed data before feature selection
cbind(accuracy_nb_2, accuracy_nb_3) 
cbind(accuracy_knn_2, accuracy_knn_3)
cbind(accuracy_rf_2, accuracy_rf_2)
cbind(accuracy_dctree, accuracy_dctree_2)
cbind(accuracy_svmlinear, accuracy_svmlinear_2)
cbind(accuracy_svmpolynomial, accuracy_svmpolynomial_2)
cbind(accuracy_svmradial, accuracy_svmradial_2)

# Hyper parameter tuning

# Create model with default paramters
control = trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(2)
metric = "Accuracy"
mtry = sqrt(nc-1)
tunegrid = expand.grid(.mtry=mtry)
rf_default = caret::train(Class_of_file~., data=bag_of_words, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)
# Tuning using caret by doing a Grid Search
control = trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(2)
tunegrid = expand.grid(.mtry=c(1:15))
rf_gridsearch = caret::train(Class_of_file~., data=bag_of_words, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

#################           Assignment Ends Here        ###########################










