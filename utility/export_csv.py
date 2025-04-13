from sqlalchemy import create_engine
import pandas as pd

# Connect to local MySQL
engine = create_engine("mysql+pymysql://root:0000@127.0.0.1:3307/aise3010finalproject-db")

# Export tables to CSV
try:
    train_data = pd.read_sql("SELECT * FROM train_data", engine)
    test_data = pd.read_sql("SELECT * FROM test_data", engine)
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    print("Exported train_data.csv and test_data.csv")
except Exception as e:
    print(f"Export error: {e}")
finally:
    engine.dispose()