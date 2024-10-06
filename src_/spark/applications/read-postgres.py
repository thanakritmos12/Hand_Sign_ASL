import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create spark session
spark = (SparkSession
         .builder
         .getOrCreate()
         )


postgres_db = sys.argv[1]
postgres_user = sys.argv[2]
postgres_pwd = sys.argv[3]

df_remote_work = (
    spark.read
    .format("jdbc")
    .option("url", postgres_db)
    .option("dbtable", "public.remote_work_mental_health")
    .option("user", postgres_user)
    .option("password", postgres_pwd)
    .load()
)

# Filter rows where Industry is 'IT' and calculate average hours worked per week
df_IT_hours = (
    df_remote_work
    .filter(df_remote_work.Industry == "IT")
    .groupBy("Industry")
    .agg(F.mean("Hours_Worked_Per_Week").alias("avg_hours_worked_per_week"))
)

output_path = "/usr/local/spark/assets/data/output_postgres/Industry_IT_Hours_Worked_Per_Week.csv"

# Save result to CSV
df_IT_hours.coalesce(1).write.format("csv").mode("overwrite").save(output_path, header=True)
