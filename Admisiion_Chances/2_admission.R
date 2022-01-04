# An education department in the US needs to analyze the factors that influence 
# the admission of a student into a college.
# Analyze the historical data and determine the key drivers.
#-----------------------------------------------------------------------
# Variables in the Dataset:
# • GRE (Graduate Record Exam scores)
# • GPA (grade point average)
# • Rank refers to prestige of the undergraduate institution. The variable rank
# takes on the values 1 through 4. Institutions with a rank of 1 have the highest
# prestige, while those with a rank of 4 have the lowest.
# • Admit is a response variable; admit/don’t admit is a binary variable where 1 
# indicates that student is admitted and 0 indicates that student is not admitted.
# • SES refers to socioeconomic status: 1 - low, 2 - medium, 3 - high.
# • Gender_male (0, 1) = 0 -> Female, 1 -> Male
# • Race – 1, 2, and 3 represent Hispanic, Asian, and African-American
#--------------------------------------------------------------------------
# Analysis information:
# Predictive
# • Run logistic model to determine the factors that influence the admission 
# process of a student (Drop insignificant variables)
# • Transform variables to factors wherever required
# • Calculate accuracy of the model
# • Try other modeling techniques like decision tree and SVM and select a 
# champion model
# • Determine the accuracy rates for each model
# • Select the most accurate model
# • Identify other Machine learning or statistical techniques that can be used
#--------------------------------------------------------------------------
# Descriptive
# • Categorize the grade point average into High, Medium, and Low (with admission
# probability percentages) and plot it on a point chart.
# • Cross grid for admission variables with GRE Categorization is shown below:
# GRE        Categorized
# 0-440         Low
# 440-580      Medium
# 580 +         High
#----------------------------------------------------------------------------
#Clear Everything
rm(list = ls())

#Install Necessary packages and load library
install.packages("plyr", "caret", "e1071")
library(plyr)
library(caret)
library(e1071)

#Load the data
admission<-read.csv("Project 1_Admission.csv")
View(admission)
summary(admission)
# Analysis:
# There are less admission since mean is less than half
# Median gre score is 580 and median gpa is 3.395
# There are many applicants from low and medium socioeconomic status than high socioeconomic status since the mean 
# is 1.992<2
# There are more female applicants since mean is less han half and median is 0
# There are many applicants from Hispanic and Asian Race than African American since the mean is 1.962<2
# Half applicants are from high rank institution (1,2) and other half applicants are from lower rank institution 
#(3,4)

#Data Structure
str(admission)
# Analysis:
# There are 400 obs with 7 variables
# admit, ses, Gender_Male, Race, rank  should be categorical instead of discrete
# gre is discrete variable 
# gpa is continuous variable

# Data Cleaning
#Change necessary datatype
admission$admit<-sapply(admission$admit, factor)
admission$ses<-sapply(admission$ses, factor)
admission$Gender_Male<-sapply(admission$Gender_Male, factor)
admission$Race<-sapply(admission$Race, factor)
admission$rank<-sapply(admission$rank, factor)
str(admission)

#See the summary of the data
summary(admission)
# Analysis:
# There are fewer admission (127 out of 400) as analyzed before

# Build the model "logistic regression" on all data
log_reg_model <- glm(admit ~ gre + gpa + rank + Race + ses + Gender_Male, data = admission, 
                     family = "binomial")
summary(log_reg_model)

# Analysis:
# Each one-unit change in gre will increase the log odds of getting admit by 0.002, and its p-value indicates that
# it is somewhat significant in determining the admit.
# 
# Each unit increase in GPA increases the log odds of getting admit by 0.81 and p-value indicates that it is 
# somewhat significant in determining the admit.
# 
# Applying from rank 1 institution will increase the log odds of getting admit by 1.361 against applying from 
# rank 2 institution wehereas applying from rank 4 institution decrease the log odds of admission by -0.22
# 
# The difference between Null deviance and Residual deviance tells us that the model is a good fit. Greater the 
# difference better the model. Null deviance is the value when you only have intercept in your equation with no 
# variables and Residual deviance is the value when you are taking all the variables into account. It makes sense 
# to consider the model good if that difference is big enough.

#----------------------------------------------------------------------------------------------------------
# Not required for this project
# Predict the chances of admission for a student with following profile:
# gre: 750, gpa: 3.90, ses: 2, Race: 2, Gender_Male: 0, rank: 1

#x <- data.frame(gre=790,gpa=3.8,ses=as.factor(2), Gender_Male=as.factor(0), Race=as.factor(2), rank=as.factor(1))
#p<- predict(log_reg_model,x)
#p
# 73% chance of getting admission
#-----------------------------------------------------------------------------------------------------------
#predict
predict <- predict(log_reg_model, type = 'response')
#confusionMatrix
table(admission$admit, predict > 0.5)

# The above model is based on all data. Lets do it by creating train and test data
# Create Training Data
input_ones <- admission[which(admission$admit == 1), ]  # all 1's
input_zeros <- admission[which(admission$admit == 0), ]  # all 0's
set.seed(100)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))  # 1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))  # 0's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)  # row bind the 1's and 0's 

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)  # row bind the 1's and 0's 

#Build the model
logistic_regression<-glm(admit ~ gre + gpa + rank + Race + ses + Gender_Male, data = trainingData,family="binomial")
summary(logistic_regression)
summary(log_reg_model)
#There are lot of differences between log_reg_model and Logistic_regression model

predicted <- predict(logistic_regression, testData, type="response")
predicted

#End of Logistic Regression Model
#--------------------------------------------------------------------------------------------------------------
# SVM Method
# We will use two metods to split data. First data will be splitted into 70-30 and training 
# data will have equal number of admits and non-admits so that there is no class bias. Second, training data 
# will contain 70% data without checking bias in admit
svm_admit<-svm(admit ~ gre + gpa + rank + Race + ses + Gender_Male, trainingData)

confusionMatrix(trainingData$admit, predict(svm_admit), positive='1')

#test data
Prediction<-predict(svm_admit,testData[-1])
Prediction_results<-table(pred=Prediction,true=testData[,1])
print(Prediction_results)

#Second way to split data withhout considering class bias
#Splitting the data
sample_split<-floor(0.7*nrow(admission))
set.seed(1)
training<-sample(seq_len(nrow(admission)), size=sample_split)
#training data
admit_train<-admission[training,]
admit_test<-admission[-training,]
table(admit_train$admit)

svm_admit<-svm(admit ~ gre + gpa + rank + Race + ses + Gender_Male, admit_train)

confusionMatrix(admit_train$admit, predict(svm_admit), positive='1')

#test data
Prediction<-predict(svm_admit,admit_test[-1])
Prediction_results<-table(pred=Prediction,true=admit_test[,1])
print(Prediction_results)

# End of SVM Model
#----------------------------------------------------------------------------------------------------------
# Decision Tree Model
tree_model<-rpart(admit~.,data=admission, method="class")
tree_model

#predict
tree_predict<-predict(tree_model, admission)
tree_predict
table(tree_predict,admission$admit)
#this is not working because tree_predict is not predicting as 0s and 1s

folded_up<-createFolds(admission, k=10, list=TRUE,returnTrain=FALSE)
train_set<-names(folded_up[1])
admission[folded_up$train_set,]
#this is not working either

#Few additional things to do
#analyze results
printcp(tree_model)
plotcp(tree_model)
summary(tree_model)
plot(tree_model)

#End of Tree model
#---------------------------------------------------------------------------------------------------------
#Naive Bayes Classifier
naive_model<-naiveBayes(admit~.,data=admission)
naive_model

#predict
naive_predict<-predict(naive_model, admission)
naive_predict
table(naive_predict,admission$admit)

#End of Naive Bayes Classifier
#---------------------------------------------------------------------------------------------------------
#Descriptive Statistics
#Categorize gre scores
admission_gre_categorized<-transform(admission, gre_category = ifelse(gre<440, "low",ifelse(gre<580, "medium", "High")))
admission_gre_gpa_categorized<-transform(admission_gre_categorized, gpa_category = ifelse(gpa<3.0, "low",ifelse(gpa<3.5, "medium", "High")))
admission_gre_gpa<- admission_gre_gpa_categorized[,-2:-3]
View(admission_gre_gpa)
str(admission_gre_gpa)
summary(admission_gre_gpa)