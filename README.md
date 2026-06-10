# 🤟 Hand Sign ASL — American Sign Language Recognition System

ระบบ Real-time Hand Sign Recognition สำหรับภาษามือ ASL (American Sign Language) โดยใช้ Computer Vision และ Machine Learning พร้อม Web Application ที่รองรับการใช้งานจริง

---

## 🎯 Overview

โปรเจคนี้พัฒนาระบบที่สามารถ

- ตรวจจับและแปลภาษามือ ASL แบบ real-time ผ่านกล้อง
- รองรับการจำแนก hand sign หลายตัวอักษร
- มี Web Application สำหรับใช้งานจริง
- บันทึก user log และประวัติการใช้งาน

---

## 🗂️ Project Structure

```
Hand_Sign_ASL/
├── src/                    # Source code หลัก
├── images/                 # Dataset รูปภาพ hand signs
├── templates/              # HTML templates สำหรับ Web App
├── collect_imgs.py         # เก็บ dataset จากกล้อง
├── create_dataset.py       # สร้าง dataset สำหรับ training
├── train_classifier.py     # Training ML model
├── app.py                  # Flask Web Application
├── app_postgres.py         # App เชื่อมต่อ PostgreSQL
├── main_app.py             # Main application
├── practice.py             # โหมดฝึกซ้อม
├── model.p / model1-3.p    # Trained models
├── data.pickle             # Processed dataset
├── docker-compose.yml      # Docker configuration
└── user_log.csv            # บันทึกการใช้งาน
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Hand Detection | MediaPipe |
| Classification | Random Forest / ML Classifier |
| Web Framework | Flask |
| Database | PostgreSQL |
| Deployment | Docker |
| Language | Python, HTML |

---

## 🚀 Installation

**วิธีที่ 1 — รันด้วย Docker (แนะนำ)**
```bash
git clone https://github.com/thanakritmos12/Hand_Sign_ASL.git
cd Hand_Sign_ASL
docker-compose up
```

**วิธีที่ 2 — รันโดยตรง**
```bash
pip install -r requirements.txt
python app.py
```

---

## 📋 How It Works

```
กล้อง → MediaPipe (Hand Landmark Detection)
      → Feature Extraction (21 key points × 2 axes)
      → ML Classifier (Random Forest)
      → Prediction → Web App Display
```

1. **Collect** — ถ่ายภาพ hand sign แต่ละตัวอักษรด้วย `collect_imgs.py`
2. **Create Dataset** — แปลงภาพเป็น hand landmarks ด้วย `create_dataset.py`
3. **Train** — เทรน classifier ด้วย `train_classifier.py`
4. **Deploy** — รัน web app ด้วย `app.py`

---

## 👥 Team Members

| Name | Student ID |
|---|---|
| Kulwadee Suttajit | 6410110039 |
| Thanakrit Chimplipak | 6410110194 |
| Arifin Madstoon | 6410110748 |

---

## 🎓 Course

Computer Vision Project — PSU
