# **🚀 Retail Sales Analysis Using PySpark in Databricks Community Edition**

## **📌 Project Overview**

This project analyzes retail sales data using PySpark in Databricks Community Edition. The goal is to clean, transform, and extract insights from the data while storing it efficiently and applying a simple machine learning model.

---

## **🛠 1. Setup Databricks Community Edition**

1. Sign up/Login to [Databricks Community Edition](https://community.cloud.databricks.com/).
2. Create a new cluster:
   - Go to **Clusters** → **Create Cluster**.
   - Choose **Single Node** mode.
   - Select **Runtime: Apache Spark (latest version)**.
   - Click **Create Cluster**.

---

## **📂 2. Upload Dataset**

- 📥 Download a retail dataset from Kaggle (e.g., Online Retail Dataset) or create synthetic data.
- 📤 Upload the dataset to **DBFS (Databricks File System)**:
  - Go to **Data** → **Create Table** → Upload CSV file.
  - Use `/FileStore/tables/` as the file path.

---

## **📊 3. Load Data Using PySpark**

```python
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("Retail Sales Analysis").getOrCreate()

# Load Data
df = spark.read.csv("/FileStore/tables/online_retail.csv", header=True, inferSchema=True)

# Display Data
df.show(5)
```

---

## **🧹 4. Data Cleaning & Transformation**

### 4.1 🔍 Check for Missing Values

```python
from pyspark.sql.functions import col

df.select([col(c).isNull().sum().alias(c) for c in df.columns]).show()
```

**🛠 Fix:** Drop rows with missing values.

```python
df = df.na.drop()
```

### 4.2 🔄 Convert Data Types

```python
df = df.withColumn("InvoiceDate", df["InvoiceDate"].cast("timestamp"))
df = df.withColumn("Quantity", df["Quantity"].cast("integer"))
df = df.withColumn("UnitPrice", df["UnitPrice"].cast("double"))
```

### 4.3 ➕ Create New Columns

Add a **Total Amount** column:

```python
from pyspark.sql.functions import col

df = df.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
```

---

## **📈 5. Exploratory Data Analysis (EDA)**

### 5.1 📊 Basic Statistics

```python
df.describe().show()
```

### 5.2 🔝 Top Selling Products

```python
from pyspark.sql.functions import sum

df.groupBy("StockCode").agg(sum("Quantity").alias("TotalQuantity")) \
  .orderBy(col("TotalQuantity").desc()).show(10)
```

### 5.3 🌍 Total Sales Per Country

```python
df.groupBy("Country").agg(sum("TotalAmount").alias("TotalSales")) \
  .orderBy(col("TotalSales").desc()).show(10)
```

---

## **💾 6. Data Storage (Delta Lake)**

Convert the cleaned data to **Delta Format** for faster querying.

```python
df.write.format("delta").mode("overwrite").save("/FileStore/delta/retail_sales")
```

Read from Delta Lake:

```python
df_delta = spark.read.format("delta").load("/FileStore/delta/retail_sales")
df_delta.show(5)
```

---

## **🤖 7. Build a Simple Machine Learning Model**

### **7.1 📌 Prepare Data for Modeling**

Predict whether an invoice will have high sales (above average).

```python
from pyspark.sql.functions import avg, when

# Find average sales
avg_sales = df.select(avg("TotalAmount")).collect()[0][0]

# Create a label column (1 if above avg, else 0)
df = df.withColumn("HighSales", when(col("TotalAmount") > avg_sales, 1).otherwise(0))
```

### **7.2 🔀 Train-Test Split**

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Feature Engineering
feature_cols = ["Quantity", "UnitPrice"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = vector_assembler.transform(df).select("features", "HighSales")

# Train-Test Split
train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)
```

### **7.3 🤖 Train Logistic Regression Model**

```python
lr = LogisticRegression(labelCol="HighSales", featuresCol="features")
model = lr.fit(train_data)

# Predictions
predictions = model.transform(test_data)
predictions.select("features", "HighSales", "prediction").show(10)
```

### **7.4 📏 Evaluate Model**

```python
evaluator = BinaryClassificationEvaluator(labelCol="HighSales")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy}")
```

---

## **🔁 8. Automate with Databricks Jobs**

- Convert Notebook into a Job for automatic execution.
- Schedule Jobs using the Databricks Scheduler.

---

## **📊 9. Visualize Data in Databricks**

Use **Databricks SQL** or **Power BI** to create dashboards.

---

# **🎯 Final Output**

✅ Retail Sales Data Cleansed & Stored in Delta Lake  
✅ Top-selling Products & Sales Insights Extracted  
✅ Simple ML Model Predicting High-Sales Transactions  
✅ Automated Pipeline with Databricks Jobs  

---

## **🚀 Next Steps**

🔹 Deploy ML Model Using Databricks MLFlow  
🔹 Real-time Data Ingestion Using AutoLoader  

---

This project provides a full **end-to-end PySpark pipeline** in Databricks. Happy coding! 🎉

