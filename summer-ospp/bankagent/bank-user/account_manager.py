# account_manager.py
import json
import hashlib
from datetime import datetime

class AccountManager:
    def __init__(self, json_file='accounts.json'):
        self.json_file = json_file
        self.accounts = self._load_accounts()
    
    def _load_accounts(self):
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save_accounts(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.accounts, f, indent=2)
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username, password):
        hashed_pw = self.hash_password(password)
        for account in self.accounts:
            if account['username'] == username and account['password'] == hashed_pw:
                account['last_login'] = datetime.utcnow().isoformat() + 'Z'
                self._save_accounts()
                return account
        return None
    
    def get_account(self, user_id):
        for account in self.accounts:
            if account['user_id'] == user_id:
                return account
        return None
    
    def update_account(self, user_id, updates):
        for i, account in enumerate(self.accounts):
            if account['user_id'] == user_id:
                self.accounts[i].update(updates)
                self._save_accounts()
                return True
        return False