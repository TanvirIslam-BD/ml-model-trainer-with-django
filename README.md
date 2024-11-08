# ml-model-trainer-with-django
Machine Learning Model Trainer With Python Django 

Rest API with JWT Authentication



````
curl -X POST http://127.0.0.1:8000/api/token/ \
-H "Content-Type: application/json" \
-d '{
    "username": "EventBookingAPI",
    "password": "123456789"
}'
````


Response:
````
 {
    "refresh": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTczMTE3MDkwNywiaWF0IjoxNzMxMDg0NTA3LCJqdGkiOiJiYTJmNzVhOTEwNzg0YTk3ODFkZTIzZWFlOTZhMGVjMiIsInVzZXJfaWQiOjN9.szcDg8h_wjN-Idx_GcI4_SOimio_4h6sWtPYfA8dabA",
  
    "access": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzMxMDg0ODA3LCJpYXQiOjE3MzEwODQ1MDcsImp0aSI6IjJlMGUyYjU4NGE2MjQ5MTNiYTk4MTA3M2MxZDJkOTE3IiwidXNlcl9pZCI6M30.mQ-xyMxVCgIzdRQWqad_PWgAJWxHJ0IWBNd1-auuR4M"
 }
````





Use rest API with your_jwt_access_token
````
curl -X POST http://localhost:8000/api/predict/ \
-H "Authorization: Bearer your_jwt_access_token" \
-H "Content-Type: application/json" \
-d '{
    "Customer": "Chinedu  J. Orjiudeh",
    "Organisation": "OCTA HQ -lifeoftwinklee",
    "Event": "LUNGU RAVE",
    "Processor": "Cash",
    "Booking type": "Cash",
    "Refund status": "No Refund",
    "Currency": "AUD",
    "Status": "Active",
    "Date": "11/3/2024 12:08:00 AM",
    "Tickets": 1,
    "Amount": 0,
    "Service Charge": 0,
    "Coupon amount": 0
}'


Response:
{
    "prediction": [
        1
    ]
}