# This code not optimal it for tesing use only
# The output might not relatable of the real world because it only base on the dataset used
# The dataset might have errors, unrealistic values, or incorrect data entries.


import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

print('Hello, Welcome to car price predictor.')
time.sleep(3)

Brand = input("Enter Brand: ")
Brand = Brand.capitalize()
Model = input("Enter Model: ")
Model = Model.capitalize()
Year = int(input("Enter What Year: "))
Engine_Size = float(input("Enter Engine Size: "))
Fuel_Type = input("Enter Fuel Type: ")
Fuel_Type = Fuel_Type.capitalize()
Transmission = input("Enter Transmission: ")
Transmission = Transmission.capitalize()
Mileage = int(input("Enter Mileage: "))
Doors = int(input("Enter number of Doors: "))
Owner_Count = int(input("Enter Owner_Count: "))

df = pd.DataFrame([[Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count]], columns=["Brand", "Model", "Year", "Engine_Size", "Fuel_Type", "Transmission", "Mileage", "Doors", "Owner_Count"])

categorical_cols = df.select_dtypes(include=['object']).columns

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

loaded_model = joblib.load("xgboost_car_price_model.pkl")
pred = loaded_model.predict(df)
pred = loaded_model.predict(df)
pred = float(pred[0])

print(f"The Car Price Predicted around: {pred}")