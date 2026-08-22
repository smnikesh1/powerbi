import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("train.csv")

# 2. Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# 3. Create monthly sales
monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Sales"].sum().reset_index()

# Convert month to date
monthly_sales["Order Date"] = monthly_sales["Order Date"].dt.to_timestamp()

# 4. Create month number
monthly_sales["Month_Number"] = range(len(monthly_sales))

# 5. Split data into training and testing
split = int(len(monthly_sales) * 0.8)

train = monthly_sales.iloc[:split]
test = monthly_sales.iloc[split:]

# 6. Prepare training data
X_train = train[["Month_Number"]]
y_train = train["Sales"]

X_test = test[["Month_Number"]]
y_test = test["Sales"]

# 7. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict test data
y_pred = model.predict(X_test)

# 9. Check model accuracy
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== MODEL PERFORMANCE =====")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 10. Predict next 6 months
last_month_number = monthly_sales["Month_Number"].max()

future_month_numbers = range(
    last_month_number + 1,
    last_month_number + 7
)

future_predictions = model.predict(
    pd.DataFrame({"Month_Number": future_month_numbers})
)

# 11. Create future dates
last_date = monthly_sales["Order Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

# 12. Display future predictions
future_data = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

print("\n===== FUTURE SALES PREDICTION =====")
print(future_data)

# 13. Plot historical + future prediction
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    marker="o",
    label="Future Predicted Sales"
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Future Sales Forecast")
plt.legend()
plt.grid(True)

plt.show()