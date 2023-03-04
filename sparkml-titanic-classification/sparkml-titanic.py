# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Research/References:
# MAGIC -  https://learn.microsoft.com/en-us/azure/databricks/getting-started/dataframes-python
# MAGIC -  https://www.databricks.com/notebooks/gallery/GettingStartedWithSparkMLlib.html

# COMMAND ----------

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler

import mlflow
import numpy as np
import pandas as pd

# COMMAND ----------

# Enable MLflow autologging for this notebook
mlflow.autolog()

# COMMAND ----------

df = spark.read.table("hive_metastore.default.titanic")


# COMMAND ----------

df.printSchema()

# COMMAND ----------

df = df.na.drop()

# COMMAND ----------

display(df)

# COMMAND ----------

trainDF, testDF = df.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

# COMMAND ----------

categoricalCols = ["Sex"]

# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]) 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 


# Convert it to a numeric value using StringIndexer.
labelToIndex = StringIndexer(inputCol="Survived", outputCol="label")


# COMMAND ----------

# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["Age", "Fare"]
#assemblerInputs = numericCols + ['SexOHE']https://adb-4454080788316148.8.azuredatabricks.net/?o=4454080788316148#
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Pipeline one step at time

# COMMAND ----------

oddone = stringIndexer.fit(df).transform(trainDF)
display(oddone)

# COMMAND ----------

oddone = encoder.fit(oddone).transform(oddone)
display(oddone)

# COMMAND ----------

oddone = labelToIndex.fit(oddone).transform(oddone)
display(oddone)

# COMMAND ----------

dyrus = vecAssembler.transform(oddone)
display(dyrus)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
 
lr = LogisticRegression(featuresCol="features", labelCol="Survived", regParam=1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC Run Pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
 
# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)
 
# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)

# COMMAND ----------

display(predDF)

# COMMAND ----------

display(predDF.select("features", "label", "prediction", "probability"))

