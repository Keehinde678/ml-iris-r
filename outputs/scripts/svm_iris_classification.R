#!/usr/bin/env Rscript
# Title: SVM Classification on Iris Dataset
# Description: This script loads the Iris dataset, trains an SVM model,
# evaluates performance using confusion matrix and accuracy, and saves output images.

# Load required libraries
if (!require("e1071")) install.packages("e1071", dependencies=TRUE)
if (!require("caret")) install.packages("caret", dependencies=TRUE)
if (!require("ggplot2")) install.packages("ggplot2", dependencies=TRUE)

library(e1071)
library(caret)
library(ggplot2)

# Load Iris dataset
data(iris)

# Set seed for reproducibility
set.seed(123)

# Split data into training and test sets (80/20)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train SVM model
svm_model <- svm(Species ~ ., data = trainData, kernel = "linear")

# Predict on test data
predictions <- predict(svm_model, testData)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, testData$Species)
print(conf_matrix)

# Save confusion matrix plot
png("svm_confusion_matrix.png")
plot(conf_matrix$table, main="SVM Confusion Matrix", col = c("lightblue", "lightgreen", "lightpink"))
dev.off()

# Save accuracy as text
accuracy <- conf_matrix$overall["Accuracy"]
write(paste("Accuracy:", round(accuracy, 4)), file = "svm_accuracy.txt")

# Save a sample plot (e.g., Sepal Length vs Width colored by species)
plot_path <- "svm_feature_plot.png"
png(plot_path)
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point(size = 3) +
  ggtitle("Sepal Length vs Sepal Width") +
  theme_minimal()
dev.off()

cat("Script completed. Accuracy, confusion matrix, and plots saved.\n")
