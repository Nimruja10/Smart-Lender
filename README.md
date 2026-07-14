
# Smart Lender 

## Project Overview

Smart Lender is a Machine Learning-based web application that predicts whether a loan application will be **Approved** or **Rejected** based on the applicant's personal and financial details. The project uses the **XGBoost** algorithm to provide accurate loan eligibility predictions through a simple Flask web application.

---

## Features

- Loan approval prediction using Machine Learning.
- User-friendly Flask web application.
- Displays Approved or Rejected loan status.
- Shows applicant summary after prediction.
- Displays key decision factors affecting loan approval.
- Supports multiple loan eligibility checks.

---

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Locust (Performance Testing)

---

## Prerequisites

Before running the project, install the following software:

- Python 3.12 or later
- Visual Studio Code (Recommended)
- Git (Optional)

---

## Installation

### Step 1: Clone or Download the Repository

Download or clone this repository to your local machine.

### Step 2: Open the Project

Open the project folder in **Visual Studio Code**.

### Step 3: Install Required Packages

Open the terminal and run:

```bash
pip install -r requirements.txt
```

---

## Train the Machine Learning Model

Run the following command:

```bash
python train_model.py
```

This will:

- Load the dataset
- Preprocess the data
- Train the machine learning model
- Evaluate model performance
- Save the trained model as **model.pkl**

---

## Run the Flask Application

Execute:

```bash
python app.py
```

The Flask server will start successfully.

---

## Open the Application

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## How to Use

1. Open the Smart Lender home page.
2. Enter all required applicant details.
3. Click **Check Loan Eligibility**.
4. View the prediction result.
5. Review the applicant summary.
6. Check the key decision factors.
7. Click **Apply Again** or **Review and Resubmit** for another prediction.

---

# Repository Structure

```text
Smart-Lender/
│
├── 1. Brainstorming & Ideation/
├── 2. Requirement Analysis/
├── 3. Project Design Phase/
├── 4. Project Planning Phase/
├── 5. Project Development Phase/
├── 6. Project Testing/
├── 7. Project Documentation/
├── 8. Project Demonstration/
│
├── Project Code/
│
├── Demo_Video_Link.md
│
└── README.md
```

---

# Project Code Structure

```text
Project Code/
│
├── app.py                     # Flask web application
├── train_model.py             # Model training script
├── model.pkl                  # Trained machine learning model
├── requirements.txt           # Required Python packages
├── locustfile.py              # Performance testing script
│
├── dataset/
│   └── loan.csv               # Loan dataset
│
├── notebook/
│   ├── model.py               # Model-related Python script
│   └── Smart_Lender.ipynb     # Jupyter Notebook
│
├── static/
│   └── style.css              # CSS file
│
└── templates/
    ├── index.html             # Home page
    └── result.html            # Prediction result page
```

---

## Performance Testing

The project includes a **Locust** script (`locustfile.py`) for performance testing.

To run Locust:

```bash
locust -f locustfile.py
```

Then open:

```
http://localhost:8089
```

Start the load test by specifying the number of users and spawn rate.

---

## Test Results

- Model trained successfully.
- Flask application executed successfully.
- Loan prediction works correctly.
- Approved case tested successfully.
- Rejected case tested successfully.
- Applicant summary displayed correctly.
- Key decision factors displayed successfully.
- Performance testing completed using Locust.

---

## Future Enhancements

- Online deployment on cloud platform.
- User authentication.
- Email notification for prediction results.
- Loan EMI calculator.
- PDF report download.
- Multi-language support.

---

## Author

**Gollapalli Nimruja Lakshmi Siva Ganga**

**Project:** Smart Lender 

---
