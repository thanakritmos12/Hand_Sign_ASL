from tabulate import tabulate

# สร้างข้อมูลตาราง
data = [
    ["Random Forest", "✓", "✓", "", "", "", "✓"],
    ["Decision Trees", "", "", "✓", "✓", "✓", ""],
    ["Support Vector Machines", "", "", "", "", "", ""],
    ["Neural Networks", "", "", "", "", "", ""],
    ["K-Nearest Neighbors", "", "✓", "✓", "✓", "✓", ""]
]

# สร้างชื่อหัวตาราง
headers = ["โมเดล", "ลด overfitting", "จัดการข้อมูลที่มีมิติสูง", 
           "ตีความง่าย", "ความเร็วในการฝึกอบรม", "ใช้หน่วยความจำต่ำ", 
           "เหมาะสำหรับข้อมูลไม่สมดุล"]

# แสดงตารางในรูปแบบที่สวยงาม
print(tabulate(data, headers=headers, tablefmt="grid", stralign="center"))
