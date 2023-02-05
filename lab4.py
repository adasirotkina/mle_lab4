import cx_Oracle as orcCon
import os
import numpy as np
import pandas as pd
from pyspark.context import SparkContext, SparkConf
from pyspark import SparkContext
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql import SQLContext, SparkSession

user = os.environ.get("PYTHON_USER", "system")

dsn = os.environ.get("PYTHON_CONNECT_STRING", "localhost/XE")

pw = os.environ.get("PYTHON_PASSWORD")
if pw is None:
    pw = 'ada8970884'

conn = orcCon.connect(user, pw, dsn)
cursor = conn.cursor()
# Execute query
sql = "SELECT * FROM result"
cursor.execute(sql)

result = pd.DataFrame(cursor.fetchall())
result.columns = ['code', 'predict']

sql = "SELECT * FROM product"
cursor.execute(sql)

FEATURES_COL = ['energy_kcal_100g',
                'energy_100g',
                'fat_100g',
                'saturated_fat_100g',
                'carbohydrates_100g',
                'sugars_100g',
                'proteins_100g',
                'salt_100g',
                'sodium_100g']

product = pd.DataFrame(cursor.fetchall())
product.columns = ['code'] + FEATURES_COL

product = product.merge(result, how='left', on='code')

conf = SparkConf().set("spark.cores.max", "16") \
    .set("spark.driver.memory", "16g") \
    .set("spark.executor.memory", "16g") \
    .set("spark.executor.memory_overhead", "16g") \
    .set("spark.driver.maxResultsSize", "0")

sc = SparkContext('local')

spark = SparkSession(sc).builder.master("local[10]").config("spark.driver.memory", "10g").getOrCreate()

sqlContext = SQLContext(sc)
product.to_csv('input.csv', index=False)
path = 'input.csv'

# df = sqlContext.read.csv(path, header=True) # requires spark 2.0
# data = pd.read_csv(path).set_index('code')

df = spark.read.csv(path, header = True, inferSchema = True)

assembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df = assembler.transform(df)

label_stringIdx = StringIndexer(inputCol='predict', outputCol='labelIndex')
df = label_stringIdx.fit(df).transform(df)

train, test = df.randomSplit([0.7, 0.3], seed=2018)

rf = RandomForestClassifier(featuresCol='features', labelCol='labelIndex')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)

accuracy = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="precisionByLabel").evaluate(predictions)
recall = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="recallByLabel").evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="f1").evaluate(predictions)

print("Metrics for Random Forest")
print("Accuracy = %s" % (accuracy))
print("Precision = %s" % (precision))
print("Recall = %s" % (recall))
print("F1 = %s" % (f1))

lr = LogisticRegression(featuresCol='features', labelCol='labelIndex')
lrModel = lr.fit(train)
predictions = lrModel.transform(test)

accuracy = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="precisionByLabel").evaluate(predictions)
recall = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="recallByLabel").evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="f1").evaluate(predictions)

print("Metrics for Logistic Regression")
print("Accuracy = %s" % (accuracy))
print("Precision = %s" % (precision))
print("Recall = %s" % (recall))
print("F1 = %s" % (f1))