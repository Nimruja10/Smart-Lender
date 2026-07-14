"""
Smart Lender – Locust Performance Test File
============================================
Run with:
    locust -f locustfile.py --host=http://localhost:5000

Then open browser at: http://localhost:8089
"""

from locust import HttpUser, task, between


class SmartLenderUser(HttpUser):
    # Each simulated user waits 1–3 seconds between tasks
    wait_time = between(1, 3)

    @task(1)
    def load_home_page(self):
        """Test GET / – Home page with loan form"""
        self.client.get("/")

    @task(3)
    def submit_prediction_approved(self):
        """Test POST /predict – Likely approved application"""
        self.client.post("/predict", data={
            "gender": "Male",
            "married": "Yes",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": "6000",
            "coapplicant_income": "2000",
            "loan_amount": "150",
            "loan_amount_term": "360",
            "credit_history": "1",
            "property_area": "Semiurban"
        })

    @task(2)
    def submit_prediction_rejected(self):
        """Test POST /predict – Likely rejected application"""
        self.client.post("/predict", data={
            "gender": "Female",
            "married": "No",
            "dependents": "3+",
            "education": "Not Graduate",
            "self_employed": "Yes",
            "applicant_income": "2000",
            "coapplicant_income": "0",
            "loan_amount": "300",
            "loan_amount_term": "360",
            "credit_history": "0",
            "property_area": "Rural"
        })
