from pyspark.sql import SparkSession
import requests

# Define the user query
user_query = "Calculate hourly total revenue for March 5, 2016. After that, calculate for March 6, 2016. Then show the revenue difference for every hour of the two days."

#user_query = "Find the average fare amount for trips with and without tolls. Return the output as a single dataframe"
#user_query = "Get the total fare amount for each vendor."
#user_query = "What was the count of payment types for Vendor ID 1 on 1 March 2016 between 1 am and 2 am?"

# Create Spark Session
spark = SparkSession.builder.appName('nlp-query-engine').getOrCreate()

# Load data
df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/yellow_tripdata_2016_03.csv")

df.show(5)

# Define your API key from Groq
API_KEY = "KEY-HERE"

# Define the schema for Spark SQL
schema_definition = """
Table: trips
Columns:
- VendorID (LONG): Unique identifier for the taxi vendor.
- tpep_pickup_datetime (STRING): Date and time when the trip started.
- tpep_dropoff_datetime (STRING): Date and time when the trip ended.
- passenger_count (LONG): Number of passengers in the trip.
- trip_distance (DOUBLE): Distance of the trip in miles.
- pickup_longitude (DOUBLE): Longitude coordinate of the pickup location.
- pickup_latitude (DOUBLE): Latitude coordinate of the pickup location.
- RatecodeID (LONG): Rate code assigned to the trip.
- store_and_fwd_flag (STRING): Indicates whether the trip record was stored in the vehicle's memory before sending to the vendor.
- dropoff_longitude (STRING): Longitude coordinate of the drop-off location.
- dropoff_latitude (DOUBLE): Latitude coordinate of the drop-off location.
- payment_type (LONG): Payment method used for the trip.
- fare_amount (DOUBLE): Base fare amount charged for the trip.
- extra (DOUBLE): Additional charges incurred during the trip.
- mta_tax (DOUBLE): Tax imposed by the Metropolitan Transportation Authority.
- tolls_amount (DOUBLE): Total tolls paid during the trip.
- improvement_surcharge (DOUBLE): Surcharge for improvements to the taxi service.
- total_amount (DOUBLE): Total amount charged to the passenger, excluding tips.
"""

# Define the prompt

prompt = f"""
You are an expert in PySpark. Your task is to convert a user’s natural language question into a valid PySpark query.
The dataset is stored in a PySpark DataFrame called `df`, and it follows this schema:

{schema_definition}
The generated PySpark code should use only this schema and assume that data is stored in a DataFrame named 'df'.

### **Rules for Code Generation:**
1. The PySpark code must operate **only on the given schema** and **assume the data is stored in `df`**.
2. The response must be **valid executable PySpark code**, following correct syntax.
3. Do **not** add any explanations, comments, or print statements—**return only the raw PySpark code**.
4. Ensure that the final result is stored in a new DataFrame called `resultDf`.
5. Always use PySpark DataFrame transformations.
6. All necessary PySpark functions MUST be imported at the beginning:
     from pyspark.sql.functions import col, count, sum, avg, when, lit, date_format, hour, year
7. **Use PySpark SQL functions** (from pyspark.sql.functions), such as `col()`, `sum()`, `avg()`, `when()`, `count()`, etc., as needed.
8. If the query requires **filtering**, use `.filter()`.
9. If filtering by datetime, use string comparisons like:
df.filter((col("timestamp") >= "YYYY-MM-DD HH:MM:SS") & (col("timestamp") <= "YYYY-MM-DD HH:MM:SS")) .

Example PySpark code:
df.filter(date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd") == "2016-03-01")

10. If the query requires **aggregation**, use `.groupBy().agg()`.
11. If the query involves **sorting**, use `.orderBy()`.
12. If the query requires **window functions**, use `Window` from `pyspark.sql.window`.
13. If an unknown column is mentioned in the query, return an **error message** in Python format (e.g., `"Column XYZ does not exist in the dataset."`).
14. If the user query contains multiple steps, generate separate intermediate DataFrames for each step.  
15. Store results from each step in a separate DataFrame (e.g., `resultDf1`, `resultDf2`, etc.).
16. If the query involves comparing results, use `.join()` or appropriate comparison logic.
17. Ensure final output is stored in `resultDf`, combining results as required.

Example query: Filter data for tip amount more than 0. Then find total tips per vendor
Example expected PySpark Code:
filteredDf = df.select("VendorID","tip_amount").filter(col("tip_amount") > 0)
resultDf =   filteredDf.groupBy("VendorID").agg(sum("tip_amount").alias("total_tip"))

18. If the query requires comparing different subsets of data (e.g., different days), store them separately and then merge them correctly.

19. If calculating differences between two datasets (e.g., revenue per hour for two days), use join.
20. Ensure date-based filtering uses proper comparisons (e.g., `col("date") >= "YYYY-MM-DD"`), NOT string operations like `.startswith()`.
21. Group only by columns explicitly required by the query:
If the query asks for a total over time, group by hour(), day(), etc.
Do not include unnecessary columns (e.g., "VendorID") unless explicitly stated.

22. Every column used in .groupBy() must have an explicit alias.
Expected PySpark code:
df.groupBy(year(col("timestamp")).alias("year"))

23. When performing joins, always use aliased DataFrames to prevent column name conflicts:

 df1 = df1.alias("df1")
 df2 = df2.alias("df2")
 resultDf = df1.join(df2, df1.year == df2.year, how="inner")

24. When selecting columns after a join, always reference them with their alias:

 resultDf = resultDf.select(
    df1.year, 
    df1.total_tips.alias("total_tips_5"),
    df2.total_tips.alias("total_tips_6"),
    (df1.total_tips_5 - df2.total_tips_6).alias("tip_difference"))


25. If the same DataFrame is used multiple times (e.g., filtering different subsets), apply caching:

Example Query and Expected PySpark Code:
### Query: Find the total hourly tip amount for March 1 2016 and March 2 2016. 
### Guideline for code generation:
Start by caching DataFrame 'df' first. Then apply count(). Filter for date 1. Filter for date 2.
### Generated code:
df.cache().count()
df1 = df.filter(date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd") == "2016-03-01")
df2 = df.filter(date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd") == "2016-03-02")



### **Example Queries & Expected PySpark Code:**
#### Query 1: "Get the total fare amount for each vendor."
Generated Code:

from pyspark.sql.functions import col, sum

resultDf = df.groupBy("VendorID").agg(sum("fare_amount").alias("total_fare"))

#### Query 2: "Show all trips where passenger count is more than 3."
Generated Code:

resultDf = df.filter(col("passenger_count") > 3)

Query 3: "Find the average trip distance for each payment type."
Generated Code:

from pyspark.sql.functions import avg

resultDf = df.groupBy("payment_type").agg(avg("trip_distance").alias("avg_distance"))

Apply the following Spark optimization best practices in the PySpark code:
- Use column pruning (only select necessary columns).
- Apply predicate pushdown (filter before selecting columns).
- Reduce shuffling and partitioning overhead.

Now, generate the PySpark code for this user query:
User Query: {user_query}

Return only the PySpark code without any explanation. Do not add any extra text."""

# Define the request payload
payload = {
    "model": "llama3-8b-8192",  # Use an available model like Llama3-8B
    "messages": [
        {"role": "system", "content": "You are an expert in PySpark."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.0
}

# Send the request to Groq API
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

# Extract and execute the generated PySpark code
response_json = response.json()
pyspark_code = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

print("Generated PySpark Code:")
print(pyspark_code)

# Execute the generated PySpark code dynamically
exec(pyspark_code, globals())

# Assuming the generated code creates an output DataFrame named 'resultDf'
print("Execution Completed. Displaying Result:")
resultDf.show()
