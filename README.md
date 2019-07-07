[![Progress](https://img.shields.io/badge/Progress-15%25-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Incomplete-orange.svg)]()

# Artificial Intelligence vs Machine Learning vs Deep Learning

- Artificial Intelligence(AI) is any technique which enables machines/computers to mimic human behaviour
- Machine Learning(ML) is subset of AI that use Statistical methods for machines/computers to learn without being explicitly programmed and improve with experience
- Deep Learning is subset of ML that use Multi-Layer Neural Network computation more feasible

# Knowledge needed to master in this area
## Mathematics (http://www.deeplearningbook.org/)
- [X] Linear Algebra
- [X] Probability Theory & Statistics
- [X] Multivariate Calculus
- [X] Algorithm & Complexity

# Type of Machine Learning Algorithms
- Supervised Learning (From a target (dependent variable) to be predicted from given set of predictors (independent variables)
- Unsupervised Learning (No target (dependent variable) usually used for clustering)
- Reinforcement Learning (Machine is trained to make decisions based on algorithm to learn from previous experience)

![alt text](https://s-media-cache-ak0.pinimg.com/736x/8b/23/3e/8b233e2d7f26b00d0c594894917a127b--supervised-learning-variables.jpg "Logo Title Text 1")

# TERMINOLOGY
- Supervised Machine Learning- Combining input from learning to produce useful predictions on unseen data
- Example = Regression problem (predict the value of house- continuous value) and Classification problem (determine which is more likely   to happen (exp- Cancer)
![](src/supervised.PNG)
- Unsupervised Machine Learning- Making sense of pattern or insight where we don't know in advance
- Example = Social network analysis
- Label- For example in spam filtering labels could be spam or not spam (target that we want to predict)
- Features- Way we present the data
- Label example- Has {features, label}:(x,y) used in training
- Unlabeled example- has {features, no label}:(x,y) used in testing
- Model- learned by mapping example to predicted label

> Arthur Samuel: Field of study that gives computers the ability to learn without being explicitly programmed.
> Tom Mitchell: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

# List of Machine Learning Algorithms (Examples)
## Linear Regression
Finding straight line or hyperplane that best fits to set of points

Multi Feature Variables in Linear Regression
![](src/Multi_feature.PNG)

Scaling and Mean Normalization
![](src/Mean_normalization.PNG)

![Alt Text](https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif)

Comparison Gradient Descent and Normal Equation
![](src/Compare_gradient.PNG)

## Reducing loss
As we train a model reducing loss will give best trained model. Stochastic Gradient Descent involves one example at a time while for Mini-Batch Gradient Descent batches 10-1000

## Type of Training models
- Static model- traine offline and exactly once
- Dynamic model- trained online and through continuous updates

## What is Regression Analysis?
Regression Analysis investigates the relationship between predictor (independent variable) and target (dependent variable) where it is predictive modelling technique. Examples including time series, forecasting as well as finding casual effects relationship between variables.

## Linear Regression
To estimate real values based on continous variables (using regression line represents by linear equation y=ax+b; where y=dependent variable, a=slope, x=independent variable, b=intercept)

## Type of Linear Regression
- Simple Linear Regression (one independent variable)
- Multiple Linear Regression (more than one independent variables)

## Python code for Linear Regression
```bash
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
```
## R code for Linear Regression

```bash
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train <- input_variables_values_training_datasets
y_train <- target_variables_values_training_datasets
x_test <- input_variables_values_test_datasets
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
linear <- lm(y_train ~ ., data = x)
summary(linear)
#Predict Output
predicted= predict(linear,x_test) 
```
## Other type of Regressions
- Logistic Regression
- Polynomial Regression
- Stepwise Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

## Logistic Regression
## Decision Tree
## SVM
## Naive Bayes
## kNN
## K-Means
## Random Forest
## Dimension Reduction Algorithm
## Gradient Boosting Algorithms:
- GBM
- XGBoost
- LightGBM
- CatBoost

## K-Means clustering
K-Mean clustering use a method of vector quantization for (cluster analysis). Cluster analysis is where grouping task is among similarity to each other for same group. It is an unsupervised learning.
![alt text](https://i.stack.imgur.com/cIDB3.png "Logo Title Text 1")

![](src/kmeans-steps.png)

## Steps in K-Means
- Choose K centroids (K rows randomly select)
- Assigns every data point to its closest centroid
- Recalculate the average/mean of all data points in a cluster of centroid
- Assign data points to theirs closest centroids

![](https://www.r-statistics.com/wp-content/uploads/2012/01/equation_1_RinAction2CH16.png)

## Determine the best value for K
Using Elbow method, whithin a range of values of K, example (1 to 10) then calculate the sum of squared errors.
![alt text](https://qph.ec.quoracdn.net/main-qimg-678795190794dd4c071366c06bf32115-c "Logo Title Text 1")
Calculation for sum of square errors
![alt text](http://i.imgur.com/aLKDhbF.png "Logo Title Text 1")

# CREDIT/REFERENCES
## Blog
[r-bloggers-K-Means](https://www.r-bloggers.com/k-means-clustering-from-r-in-action/)
## Online Course
[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
## Website
[Common-machine-learning-algorithms](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
## Github
[Siraj Raval](https://github.com/llSourcell)

## Big Data
Three major sources of big data
- People (social media)
- Organizations (transaction info)
- Machines (sensor)

## Characteristics of Big Data
- Volume (size)
- Variety (complexity)
- Velocity (speed)
- Valence (connectedness)
- Veracity (quality)
- Value (integrating different types of data sources)

## Big Data Engineering
- [X] Step 1- Acquire data (identify data sets, retrive data, query data)
- [X] Step 2- Prepare data sets:
- Explore (undertsand data,preliminary analysis)
- Pre-process (clean, integrate, package)
- [X] Step 3- Analysis data (using analytic techniques, build models)
- [X] Step 4- Report (interpret, summarize, visualize, post-process)
- [X] Step 5- Act (apply results)

## Hadoop Ecosystem
- Enable scalibility
- Handle fault tolerence
- Optimized for a variety of data types
- Facilited shared environment
- Community supported

## Big Data Ecosystem
![](src/hadoop.PNG)
- [X] HDFS -distributed file system (scalable storage, fault tolerence)
- [X] Hadoop YARN -flexible scheduling, resource management
- [X] Hadoop MapReduce -simplify programming model
- Map -> apply()
- Reduce -> summarize()
- [X] Hive & Pig -High-level programming models
- Hive -SQL-like query
- Pig -dataflow scripting
- [X] Giraph -specilized model for graph processing
- [X] Storm, Spark & Flink -real-time and in-memeory processing
- [X] Hbase, Cassandra & MongoDB -NoSQl for non-files
- [X] Zookeeper -schronization, high-availability, configuration)

## Hadoop Command Lines
- **$ hadoop fs -ls /** Show all directories
- **$ hadoop fs -du test/mytext.txt** Show size of the file
- **$ cat > mytext.txt** Create text file
- **$ hadoop fs -mkdir test** Make directory called test
- **$hadoop fs -cat test/mytext.txt | grep my** Search keyword #my in text file and display it
- **$hadoop fs -du s test** Show size of the folder

## Hadoop Components
- MapReduce - Framework for processing huge datasets of distributed problems using large number of nodes
- Flume - A service for moving large amounts of data around a cluster soon after the data is produced
## HDFS is designed for:
 Large files, streaming data access, and commodity hardware
## What is the purpose of the shuffle operation in Hadoop MapReduce?
 To transfer each mapper's output to the appropriate reducer node based on a partitioning function
## Which of the following is a duty of the DataNodes in HDFS?
 Store and retrieve blocks when told to by clients or the NameNode
## Which of the following is a duty of the NameNode in HDFS?
 Maintain the file system tree and metadata for all files and directories
## Which component determines the specific nodes that a MapReduce task will run on?

## Data Lakes
- Big data storage for current and future analysis
## lambda architecture
- A method to process streaming data by utilizing batch processing and real time processing
## Difference between data lakes and data warehouses
- Data lakes house raw data while data warehouses contain pre-formatted data
## Schema-on-read
- Data is stored as raw data until it is read by an application where the application assigns structure
## Schema-on-write
- The process where data is pre-formatted prior to being read but the schema is loaded on read
## streaming
- Utilizing real time data to compute and change the state of an application continuously
## streaming data
- Data is unbounded in size but requires only finite time and space to process it
- Data manipulation is near real time
- Does not ping the source interactively for a response upon receiving the data
- independent computations that do not rely on previous or future data
- Small time windows for working with data

## Data models
- Conceptual Data Model
- Logicsk Data Model
- Physicak Data Model

----------------------------------------------------------------------------------------------------------------------------------------

## Information Security
[![Progress](https://img.shields.io/badge/Progress-5%25-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Incomplete-orange.svg)]()

- [X] Penetration testing
- [X] Vulnerable testing
- [X] Firewall/Antivirus
- [X] Commands in Kali Linux

## Android App Security

Android provides a sandboxed app execute env. A customized embedded Linux system interacts
with the phone hardware and an off-processor cellular radio. 

Java compiler creates JVM bytecode, the Dalvik dx compiler consumes the .class files, recompiles them to
Dalvik bytecode and writes the app into single.dex file.

The process consists of translation, reconstruction and interpretation of 3 basic elements of the app
: the constant pools, class definitions and data segment.

Constant pool: constant (references to other classes,method names, numerical constant)
Class definitions: basic infos(access flags, class names)
Data segment: method code executed by target VM number of DVM registers used, local variable table,
and operand stack sizes), class and instance variable

## Risks
- Rapidly developed and deployed applications
- Coarse permission systems
- Privacy invading behaviors
- malware
- limited security models

## Aims
- Design Dalvik decompiler ded
- Analyze 21 million LOC

## Hypothesis
- Misuse of privacy sensitive infos (phone identifiers (IMEI,IMSI,ICC-ID), geo location


Giveaway of the day
- -esque (in the style of) ex: Phone identiﬁers, e.g., IMEI, IMSI, and ICC-ID, were used for everything from “cookie-esque” tracking to accountsnumbers.
- renaissance
- posit (put in position) - Wherepossible, we identify root causes and posit the severity ofdiscoveredvulnerabilities. 
- breadth (the distance between two) - we consider a breadth of concerns including both dangerous functionality and vulnerabilities.
- substantially -significant extent

## Computer science terms
- IPC (Interprocess Comm)
- Intent (

## Differences of JVM and DVM
- Application structure
	-Java apps more than one .class files
	-Dalvik apps single .dex contain all classes
- Register architecture
	-JVM (stack-based)
	-DVM (register-based)
- Instruction set
	-Java has 200 opcodes
	-Dalvik has 218 opcodes
- Constant pool structure
	-Java app replicete elements within .class files (referer and referent method names)
	-dx compiler eliminates the replication
- Control flow structure (loops, switch statements, exception handlers)	
	-Java bytecode structure loosely mirrowa the source code
	-Dalvik bytecode vice versa
- Ambiguous primitive types (int, float, long, dounle)
	-Java bytecode variable assignments distinguish
	-Dalvik use same opcodes (untyped)
- Null references
	-Dalvik not specify instead use zero value constant
- Comparison of object references
	-Java bytecode uses typed opcodes (ifnull and ifnotnull)
	-Dalvik bytecode use more simplistic
- Storage of primitive types in arrays
	-Java bytecode is unambiguous
	-Dalvik opcode uses ambiguous opcodes (aget foe int/float)
	
## Ded decompiler
- Application retargetting (recovering typing info,translating constant pool and retargeting the bytecode)
	-Type inference
		- identify class and method constants and variables
		- infers register types by observing how they are used in subsequent operation with known type operands
	-Constant pool conversion
		-Java maintains for each class, Java bytecode uses constant pool for most references
		-Dalvik uses single constant pool, Dalvik bytecode places primitive type constant in bytecode
	-Method retargeting
		- First process the bytecode to reorganize structures that cannot be directly retargeted
		- Linearly traverse the DVM bytecode and translate to JVM			
- Optimization and Decompilation

## Development of Android Apk

## Prerequisites
- [X] Java
- [X] Java Development Kit
- [X] Android Studio

# Wireless Commands
- [X] iwconfig
- [X] iwlist - iwlist wlan0 scanning
- [X] ifconfig wlan0 up
- [X] iw wlan0 scan
- [X] airmon-ng start wlan0 11
- [X] aireplay-ng --test wlan0mon
- [X] iw dev wlan0 set type monitor
- [X] aireplay-ng --test -i wlan0 wlan1
- [X] airserv-ng -d wlan0
- [X] airdump-ng --bssid --channel wlan
- [X] aircrack-ng 
- [X]wlan.fctype==2
