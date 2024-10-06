import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import requests

# Create spark session
spark = (SparkSession
         .builder
         .getOrCreate()
         )

# Accept database credentials from command-line arguments
postgres_db = sys.argv[1]
postgres_user = sys.argv[2]
postgres_pwd = sys.argv[3]

def send_line_notify(message):
    line_notify_token = 'U7BbRCeJyAtIFI5zmdsGGBuYNwzTI3nunsFkKSempiL'  # เปลี่ยนเป็น Token ของคุณ
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    payload = {'message': message}
    
    response = requests.post(line_notify_api, headers=headers, data=payload)

    if response.status_code == 200:
        print("ส่งข้อความ LINE สำเร็จ")
    else:
        print(f"ส่งข้อความ LINE ล้มเหลว: {response.status_code}, {response.text}")

# Load data from public.user_score table
df_user_score = (
    spark.read
    .format("jdbc")
    .option("url", postgres_db)
    .option("dbtable", "public.user_scores")
    .option("user", postgres_user)
    .option("password", postgres_pwd)
    .load()
)

# Sort by Score descending and total_time ascending
# df_sorted_scores = (
#     df_user_score
#     .orderBy(F.desc("Score"), F.asc("total_time"))  # Sort by Score and total_time
#     .limit(5)
# )


# คำนวณ Top 5 คะแนนสูงสุด
df_top_scores = (
    df_user_score
    .groupBy("Username")
    .agg(
        F.max("Score").alias("max_score"),
        F.min("total_time").alias("min_time")
    )
    .orderBy(F.desc("max_score"), F.asc("min_time"))
    .limit(5)
)
# จัดรูปแบบข้อความ
top_scores_list = df_top_scores.collect()
message = "🏆 Top 5 User Scores of the Week 🏆\n"
for i, row in enumerate(top_scores_list, 1):
    message += f"{i}. Username: {row['Username']}\nScore: {row['max_score']}\nTime: {row['min_time']}\n"

# ส่งข้อความไปยัง LINE
send_line_notify(message)

# Define output path for the CSV file
output_path = "/usr/local/spark/assets/data/output_postgres/Sorted_User_Scores.csv"

# Save the sorted result to CSV
df_top_scores.coalesce(1).write.format("csv").mode("overwrite").save(output_path, header=True)
