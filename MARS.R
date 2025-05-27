if (!require(earth)) install.packages("earth")
library(earth)

X <- as.matrix(read.table("x_train.txt", sep = " "))
y <- as.vector(read.table("y_train.txt")[, 1])
df <- data.frame(X)
colnames(df) <- paste0("X", 1:ncol(X)) 
df$target <- y

# cols_to_remove <- c()
# cols_to_remove <- c(13, 178, 194, 298, 305, 117, 228, 462, 414, 425, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)+1 # VIF-removed variables
df <- df[, c(3)]
str(df)

mars_model <- earth(target ~ ., data = df, degree = 1) # degree = 2
ev <- evimp(mars_model)
print(ev)
