# dify_integration.py
from account_manager import AccountManager

account_manager = AccountManager()

def authenticate_user(username, password):
    """用于Dify认证的API端点"""
    account = account_manager.authenticate(username, password)
    if account:
        return {
            "success": True,
            "user_id": account['user_id'],
            "account_type": account['account_type']
        }
    return {"success": False, "message": "Invalid credentials"}

def get_account_info(user_id):
    """获取账户信息供智能体使用"""
    account = account_manager.get_account(user_id)
    if account:
        return {
            "balance": account['account_balance'],
            "phone": account['phone'],
            "transactions": account['transactions']
        }
    return None