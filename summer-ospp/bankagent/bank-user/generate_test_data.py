# generate_test_data.py
import json
import random
import hashlib
from datetime import datetime, timedelta

def generate_accounts(num=10):
    accounts = []
    for i in range(1, num+1):
        user_id = f"100{i:02d}"
        accounts.append({
            "user_id": user_id,
            "username": f"customer{i}",
            "password": hashlib.sha256(f"Password123!{i}".encode()).hexdigest(),
            "phone": f"13800138{random.randint(100,999)}",
            "account_type": random.choice(["personal", "business"]),
            "account_balance": round(random.uniform(1000, 1000000), 2),
            "last_login": (datetime.utcnow() - timedelta(days=random.randint(0,30))).isoformat() + 'Z',
            "transactions": generate_transactions()
        })
    return accounts

def generate_transactions(max_txns=5):
    txns = []
    for i in range(random.randint(1, max_txns)):
        txns.append({
            "id": f"txn{random.randint(1000,9999)}",
            "date": (datetime.utcnow() - timedelta(days=random.randint(1,90))).strftime("%Y-%m-%d"),
            "amount": round(random.uniform(10, 10000), 2),
            "type": random.choice(["deposit", "withdrawal", "transfer"])
        })
    return txns

if __name__ == "__main__":
    accounts = generate_accounts()
    with open('bank_users.json', 'w') as f:
        json.dump(accounts, f, indent=2)