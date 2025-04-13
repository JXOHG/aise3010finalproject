from sqlalchemy import create_engine

try:
    engine = create_engine("mysql+pymysql://root:0000@34.124.127.131:3306/aise3010finalproject-db")
    with engine.connect() as conn:
        print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    engine.dispose()