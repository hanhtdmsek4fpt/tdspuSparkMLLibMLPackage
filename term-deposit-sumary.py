# Step 1:
# Disable warnings, set Matplotlib inline plotting and load Pandas package
import warnings
warnings.filterwarnings('ignore') 
# Import pandas and matplotlib
import pandas as pd
#import matplitlib pylot
import matplotlib.pyplot as plt
#import SparkSession
from pyspark.sql import SparkSession

# Step 2:
# Get sparkSession in pyspark console
spark = SparkSession.builder\
    .master("local")\
    .appName("Bank Marketing Prediction")\
    .getOrCreate()
# Load data from csv file
dataset_all = spark.read.load("./data/ bank.csv",
                  format="com.databricks.spark.csv",
                  header="true", inferSchema="true")
# Check schema
dataset_all.printSchema()
pd.DataFrame(dataset_all.take(5), columns=dataset_all.columns).transpose()
# Step 3:
#Normalized data start
binary_yes_no_map = {'yes':1.0, 'no':0.0}
toNum = UserDefinedFunction(lambda k: binary_yes_no_map[k], DoubleType())
job_map = {'admin.':0.0, 'blue-collar':1.0, 'entrepreneur':2.0, 'housemaid':3.0, \
	   'management':4.0, 'retired':5.0, 'self-employed':6.0, 'services':7.0, \
	   'student':8.0, 'technician':9.0, 'unemployed':10.0, 'unknown':11.0}
jobToNum = UserDefinedFunction(lambda k: job_map[k], DoubleType()) 
marital_map = {'divorced':0.0, 'married':1.0, 'single':2.0}
maritalToNum = UserDefinedFunction(lambda k: marital_map[k], DoubleType())
education_map = {'primary':0.0, 'secondary':1.0, 'tertiary':2.0, 'unknown':3.0}
educationToNum = UserDefinedFunction(lambda k: education_map[k], DoubleType())
contact_map = {'cellular':0.0, 'telephone':1.0, 'unknown':2.0} 
contactToNum = UserDefinedFunction(lambda k: contact_map[k], DoubleType())
month_map = {'jan':0.0, 'feb':1.0, 'mar':2.0, 'apr':3.0, 'may':4.0, 'jun':5.0, \
		'jul':6.0, 'aug':7.0, 'sep':8.0, 'oct':9.0, 'nov':10.0, 'dec':11.0}
monthToNum = UserDefinedFunction(lambda k: month_map[k], DoubleType())
poutcome_map = {'failure':0.0, 'success':1.0, 'other':2.0, 'unknown':3.0}
poutcomeToNum = UserDefinedFunction(lambda k: poutcome_map[k], DoubleType())
dataset_all_modified = dataset_all.drop('default')\
		.withColumn('housing', toNum(dataset_all['housing']))\
		.withColumn('loan', toNum(dataset_all['loan']))\
		.withColumn('job', jobToNum(dataset_all['job']))\
		.withColumn('marital', maritalToNum(dataset_all['marital']))\
		.withColumn('education', educationToNum(dataset_all['education']))\
		.withColumn('contact', contactToNum(dataset_all['contact']))\
		.withColumn('month', monthToNum(dataset_all['month']))\
		.withColumn('poutcome', poutcomeToNum(dataset_all['poutcome']))\
		.withColumn('deposit', toNum(dataset_all['deposit'])).cache()
pd.DataFrame(dataset_all_modified.take(5), columns=dataset_all_modified.columns).transpose()
dataset_all_modified.describe().toPandas().transpose()
#Normalized data end

# Step 4:
# Randomly split dataset into 2 sub-datatsets with
# 80% for training and 20% for testing
train_data, test_data = dataset_all_modified.randomSplit([0.8, 0.2])

# Step 5:
# Correlation analysis
numeric_features = [t[0] for t in
                    train_data.dtypes if t[1] == "int" or t[1] == "double"]
sampled_data = train_data.select(numeric_features) \
    .sample(False, 0.02).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(12, 12))
# Rotate axis labels and remove axis ticks
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
#Show the data plotting
plt.show()


#Step 6:
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
def labelData(data):
    # label: row[end], features: row[0:end-1]
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))
training_data, testing_data = labelData(dataset_all_modified).randomSplit([0.8, 0.2])
model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,categoricalFeaturesInfo={5:2, 6:2},impurity='gini', maxBins=32)
print model.toDebugString()

#Step 7
from pyspark.mllib.evaluation import MulticlassMetrics
def getPredictionsLabels(model, test_data):
    predictions=model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))
def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print 'Precision of True ', metrics.precision(1)
    print 'Precision of False', metrics.precision(0)
    print 'Recall of True    ', metrics.recall(1)
    print 'Recall of False   ', metrics.recall(0)
    print 'F-1 Score         ', metrics.fMeasure()
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()
predictions_and_labels=getPredictionsLabels(model, testing_data)
printMetrics(predictions_and_labels)

#Step 8
training_data, testing_data = labelData(satisfied_dataset_all).randomSplit([0.8, 0.2])
model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,categoricalFeaturesInfo={5:2,6:2},impurity='gini',  maxBins=32)
predictions_and_labels=getPredictionsLabels(model, testing_data)
printMetrics(predictions_and_labels)

# Step 9
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# Vectorize
vecAssembler = VectorAssembler(inputCols=['age','job','marital',’education','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'],outputCol="features")
# Transform data
df_train = vecAssembler.transform(train_data)
pd.DataFrame(df_train.take(5), columns=df_train.columns).transpose()
dt = DecisionTreeClassifier(labelCol="deposit", featuresCol="features")
pipeline = Pipeline(stages=[vecAssembler, dt])
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
#Select prediction information
predictions.select("prediction", "Classification", "features").toPandas().head(10)
evaluator = BinaryClassificationEvaluator(labelCol="deposit", rawPredictionCol="prediction")
evaluator.evaluate(predictions)
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth,[2,3,4,5,6,7,8,9,10,11,12]).build()
# Set up 3-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator, 	
                          numFolds=3)
CV_model = crossval.fit(train_data)
tree_model = CV_model.bestModel.stages[1]
print(tree_model)
predictions_improved = CV_model.bestModel.transform(test_data)
predictions_improved.select("prediction", "deposit", "features").toPandas().head(10)
evaluator.evaluate(predictions_improved)