from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re
import datetime

# Initialize Spark Session
spark = SparkSession.builder.appName("NameClassification").getOrCreate()

# Path to your CSV file
csv_file_path = "/content/name_matching.csv"

# Read CSV into DataFrame
df = spark.read.option("header", True).csv(csv_file_path)

# Select the 'name' and 'name_variant' columns
df = df.select("name", "name_variant")

# Preprocessing Function
def preprocess_input(input_value):
    if input_value is None:
        return ""
    # Convert input to string
    input_str = str(input_value)
    # Remove leading and trailing spaces
    input_str = input_str.strip()
    # Remove extra spaces within the string
    input_str = re.sub(r'\s+', '', input_str)
    return input_str

# Classification Function
def classify_input(input_str):
    if input_str is None or input_str == "":
        return "Unknown"
    
    # Check for Numeric
    if re.fullmatch(r'\d+', input_str):
        return "Numeric"
    
    # Check for Date format
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for date_format in date_formats:
        try:
            datetime.datetime.strptime(input_str, date_format)
            return "Date"
        except ValueError:
            continue
    
    # Check for Alphanumeric
    if re.fullmatch(r'[A-Za-z0-9]+', input_str):
        return "Alphanumeric"
    
    # # Check for Character Only (including special characters)
    # if re.fullmatch(r'[^\w\s]+', input_str):
    #     return "Character Only"
    if all(c.isalpha() or not c.isalnum() for c in input_str):
        return "Character Only"
    
    return "Unknown"

# Register UDFs
preprocess_udf = udf(preprocess_input, StringType())
classify_udf = udf(classify_input, StringType())

# Apply UDFs to 'name' and 'name_variant' columns
df = df.withColumn("cleaned_name", preprocess_udf(col("name")))
df = df.withColumn("name_classification", classify_udf(col("cleaned_name")))
df = df.withColumn("cleaned_name_variant", preprocess_udf(col("name_variant")))
df = df.withColumn("variant_classification", classify_udf(col("cleaned_name_variant")))

# Show the resulting DataFrame
df.select("name", "cleaned_name", "name_classification", "name_variant", "cleaned_name_variant", "variant_classification").show(truncate=False)
