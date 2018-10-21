#pyspark --packages com.databricks:spark-csv_2.10:1.3.0 --master local[*]

import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder\
    .master("local")\
    .appName("Bank Marketing Prediction")\
    .getOrCreate()

dataset_all = spark.read.load("paper2/data/bank.csv",
                             format="com.databricks.spark.csv",
                             header="true",
                             inferSchema="true")
dataset_all.printSchema()
pd.DataFrame(dataset_all.take(5), columns=dataset_all.columns).transpose()
dataset_all.describe().toPandas().transpose()

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
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest

def labelData(data):
    # label: row[end], features: row[0:end-1]
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

training_data, testing_data = labelData(dataset_all_modified).randomSplit([0.8, 0.2])

model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=2, maxBins=32)

print model.toDebugString()

from pyspark.mllib.evaluation import MulticlassMetrics
def getPredictionsLabels(model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))

def printMetrics(predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print 'Precision of True ', metrics.precision(1)
    print 'Precision of False', metrics.precision(0)
    print 'Recall of True    ', metrics.recall(1)
    print 'Recall of False   ', metrics.recall(0)
    print 'F-1 Score         ', metrics.fMeasure()
    print 'Confusion Matrix\n', metrics.confusionMatrix().toArray()


predictions_and_labels = getPredictionsLabels(model, testing_data)

printMetrics(predictions_and_labels)

dataset_all.groupby('deposit').count().toPandas()

satisfied_dataset_all = dataset_all_modified.sampleBy('deposit', fractions={1: 1.0, 0: 5289./5873}).cache()
satisfied_dataset_all.groupby('deposit').count().toPandas()


training_data, testing_data = labelData(satisfied_dataset_all).randomSplit([0.8, 0.2])

model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=2, maxBins=32)
predictions_and_labels = getPredictionsLabels(model, testing_data)

printMetrics(predictions_and_labels)


