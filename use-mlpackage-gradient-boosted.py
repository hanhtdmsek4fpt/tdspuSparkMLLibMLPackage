#pyspark --packages com.databricks:spark-csv_2.10:1.3.0 --master local[*]

import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier

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

satisfied_dataset_all = dataset_all_modified.sampleBy('deposit', fractions={1: 1.0, 0: 5289./5873}).cache()
satisfied_dataset_all.groupby('deposit').count().toPandas()

train_data, test_data = satisfied_dataset_all.randomSplit([0.8, 0.2])

vecAssembler = VectorAssembler(inputCols=['age',
					  'job',
					  'marital',
					  'education',
					  'balance',
					  'housing',
					  'loan',
					  'contact',
					  'day',
					  'month',
					  'duration',
					  'campaign',
					  'pdays',
					  'previous',
					  'poutcome'],
                               outputCol="features")

df_train = vecAssembler.transform(train_data)
pd.DataFrame(df_train.take(5), columns=df_train.columns).transpose()

gbt = GBTClassifier(labelCol="deposit", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=[vecAssembler, gbt])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)
predictions.select("prediction", "deposit", "features").toPandas().head(25)

evaluator = BinaryClassificationEvaluator(
    labelCol="deposit", rawPredictionCol="prediction")

evaluator.evaluate(predictions)

paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [2,3,4,5,6,7,8,9,10,11,12]).build()

# Set up 3-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator, 
                          numFolds=3)

CV_model = crossval.fit(train_data)

tree_model = CV_model.bestModel.stages[1]
print(tree_model)

predictions_improved = CV_model.bestModel.transform(test_data)

predictions_improved.select("prediction", "deposit", "features").toPandas().head(25)

evaluator.evaluate(predictions_improved)
