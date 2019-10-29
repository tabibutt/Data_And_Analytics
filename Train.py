This Notebook uses ALS to train a Model for Generating Recommendations using Implicit Data (Confidence Metric)
The Type of Interactions considered are Add to Favorites, Add to Basket and Purchases.
The Number of Interactions are summed to an Integer to calculate Confidence of an Item for Each User
The Dataset used is from August 2018 - August 2019
The list of Hyperparameters used are the following:
(i) regparam = 0.01
(ii) alpha = 10
(iii) rank=35
(iv) maxIter=15

#Reading the Dataset. The Dataset is n*n Martics for itema and users. Sample file is avaiable in the repository

ratings=spark.sql('select * from Implicit_Ratings_Using_Confidence_limited_Interactions_ECOM_Version_V1')

#apply post filtering for the Black and sensitive products
#Reading the Black Listed Products as a table stored

black_listed_products=spark.sql("select * from Black_listed_Products_V1")
black_listed_products=black_listed_products.withColumnRenamed('Product_ID', 'Varenr')
black_listed_products=black_listed_products.select('Varenr')
ratings=ratings.join(black_listed_products, 'Varenr', how='left_anti')

#Reading the JSON Feed Online from the DataFeedWatch Link 

import json,urllib.request
import requests
data = requests.get("https://feeds.datafeedwatch.com/27521/ed7116a2eb05b8eec69972bcf3c9313e9058a4d3.json")
import json 
# convert into JSON:
y = json.dumps(data.json())
#converting the JSON into some specific structure to fit into Spark Dataframe
jsonStrings=[y]
jsonRDD = sc.parallelize(jsonStrings)
final_df = spark.read.options(inferSchema = 'true', multiline = "true").json(jsonRDD)
#Since we need the products data in the Feed Only
from pyspark.sql.functions import explode
product = final_df.select(explode(final_df.products).alias("Items"))
#Extracting the relevant product Ids and their product categories
product=product.select(product.Items.product.id.alias("Varenr"),product.Items.flags.isWebshopActive.alias("isWebshopActive"), product.Items.product.fullProductName.alias("Product_Full_Name"), product.Items.product.displayPrice.alias("displayPrice"), product.Items.product.brandName.alias("brandName"), product.Items.flags.isInStock.alias("isInStock"), product.Items.stock.onlineStockCount.alias("onlineStockCount"))
#Lets convert the to the Datatypes needed
from pyspark.sql.types import *
product=product.withColumn("displayPrice", product["displayPrice"].cast(DoubleType()))
product=product.withColumn("Varenr", product["Varenr"].cast(LongType()))
product=product.withColumn("isWebshopActive", product["isWebshopActive"].cast(BooleanType()))
product=product.withColumn("onlineStockCount", product["onlineStockCount"].cast(LongType()))
product=product.withColumn("isInStock", product["isInStock"].cast(BooleanType()))
#We are only using the Active products in ECOM
product_list_removal=product.filter((product['onlineStockCount']==0) | (product['brandName']=='CHANEL') | (product['displayPrice']<30.0))
product_list_removal=product_list_removal.select('Varenr')
#lets apply the list to our User Predictions 
ratings=ratings.join(product_list_removal, 'Varenr', how='left_anti')

#We need to convert the Varenr to LongType, compulsively 

from pyspark.sql.types import LongType
ratings=ratings.withColumn("Varenr", ratings["Varenr"].cast(LongType()))

#Keeping the Users with Atleast 15 Ratings 

users_list=ratings.groupBy('Medlemsnr').count()
users_list=users_list.filter(users_list['count']>=15).select('Medlemsnr')
ratings=ratings.join(users_list, 'Medlemsnr', 'inner')


#lets create an index column for the Medlemsnr
#from pyspark.sql.functions import row_number

from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window
Index_Users=ratings.select('Medlemsnr').distinct()
Index_Users = Index_Users.withColumn(
    "Medlemsnr_index",
    row_number().over(Window.orderBy(monotonically_increasing_id()))-1
)
ratings=ratings.join(Index_Users, 'Medlemsnr', 'inner')


#select the required columns 

ratings = ratings.select('Medlemsnr_index', 'Varenr', 'Confidence_Rating')

from reco_utils.dataset.spark_splitters import (
    spark_random_split, 
    spark_chrono_split, 
    spark_stratified_split,
    spark_timestamp_split
)
import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator

training, test = spark_stratified_split(
    ratings, ratio=0.65, filter_by="user",
    col_user='Medlemsnr_index', col_item='Varenr', seed=42
)

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(alpha=10, rank=35, maxIter=15, regParam=0.01, 
          userCol="Medlemsnr_index", itemCol="Varenr", ratingCol="Rating",
          coldStartStrategy="drop",
          implicitPrefs=True, seed=42)
model = als.fit(training)

#started logging the Model
with mlflow.start_run():
    mlflow.spark.log_model(model, "MyALSModel")
    modelpath = "/dbfs/ml/SparkModel/"
    mlflow.spark.save_model(model, modelpath)
    
new_model=mlflow.pyfunc.load_model('/dbfs/ml/SparkModel/', suppress_warnings=True)

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
test1=test.select('Medlemsnr_index', 'Varenr')
test1=test1.head(10)
result_pdf = test1.select("*").toPandas()

result_pdf = test.select("*").toPandas()

sample=result_pdf.head()

result_pdf=result_pdf[['Medlemsnr_index', 'Varenr']]
  
%%time
new_model.predict(sample)

    

