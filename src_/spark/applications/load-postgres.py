import sys
from pyspark.sql import SparkSession

# Create spark session
spark = (SparkSession
         .builder
         .getOrCreate()
         )

# รับพาธของไฟล์ Impact_of_Remote_Work_on_Mental_Health.csv และข้อมูลเชื่อมต่อ PostgreSQL จาก arguments
remote_work_file = sys.argv[1]
postgres_db = sys.argv[2]
postgres_user = sys.argv[3]
postgres_pwd = sys.argv[4]

# อ่านไฟล์ CSV Impact_of_Remote_Work_on_Mental_Health.csv
df_remote_work_csv = (
    spark.read
    .format("csv")
    .option("header", True)
    .load(remote_work_file)
)

# เขียนข้อมูลลงในตาราง PostgreSQL ที่ชื่อว่า public.user_scores
(
    df_remote_work_csv.write
    .format("jdbc")
    .option("url", postgres_db)
    .option("dbtable", "public.user_scores")
    .option("user", postgres_user)
    .option("password", postgres_pwd)
    .mode("overwrite")
    .save()
)