from flask import Flask, request, jsonify
import json
import hashlib
import os, time, threading

app = Flask(__name__)

# ======= 数据加载 =======
with open('bank_users.json', 'r', encoding='utf-8') as f:
    users = json.load(f)["userData"]

# 可选：画像文件，用于投资推荐（若没有该文件，请先生成或按需调整）
try:
    with open('user_profiles.json', 'r', encoding='utf-8') as f:
        user_profiles = {u["userId"]: u for u in json.load(f)["users"]}
except Exception:
    user_profiles = {}  # 没有画像文件时置空，也能跑，只是推荐逻辑将退化为保守方案

# ======= 投诉数据文件（最小持久化） =======
TICKETS_FILE = 'tickets.json'
_tickets_lock = threading.Lock()

def _load_tickets():
    if not os.path.exists(TICKETS_FILE):
        return {"tickets": []}
    with open(TICKETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_tickets(data):
    with _tickets_lock:
        with open(TICKETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def _gen_ticket_id():
    # TKT-YYYYMMDD-abcdef（毫秒简化）
    return f"TKT-{time.strftime('%Y%m%d')}-{int(time.time()*1000)%1000000:06d}"

def _now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ======= 工具函数 =======
def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def find_user_by_phone_password(phone: str, password_plain: str):
    """根据手机号+明文密码返回用户对象，不存在返回 None"""
    if not phone or not password_plain:
        return None
    hashed = md5_hash(password_plain)
    for u in users:
        if u.get('phoneNumber') == phone and u.get('password') == hashed:
            return u
    return None

def find_user_by_id(user_id: str):
    for u in users:
        if u.get('userId') == user_id:
            return u
    return None

def recommend_by_profile(profile: dict):
    """
    基于画像给出示例推荐清单。
    你可按业务需求替换成更精细的规则或模型打分。
    """
    risk = (profile.get("riskLevel") or "low").lower()
    exp = (profile.get("investmentExperience") or "none").lower()
    income = (profile.get("incomeLevel") or "low").lower()

    # 简单规则示例
    if risk == "high":
        products = [
            "Tech Growth Fund",
            "Crypto Index ETF",
            "AI Startups Portfolio"
        ]
    elif risk == "medium":
        if exp in ["advanced", "moderate"]:
            products = [
                "Global Equity Fund",
                "Balanced Income Fund",
                "Emerging Market ETF"
            ]
        else:
            products = [
                "Balanced Income Fund",
                "Index Bond Fund"
            ]
    else:  # low
        products = [
            "Government Bond Fund",
            "Fixed Income ETF",
            "Capital Protection Plan"
        ]

    # 可按收入微调（演示）
    if income == "high" and "Capital Protection Plan" in products:
        products.remove("Capital Protection Plan")
        products.append("Investment-Grade Corporate Bond Fund")

    return products


# ======= 接口实现 =======

# 🔐 用户认证
@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    data = request.get_json(silent=True) or {}
    phone_number = data.get("phoneNumber")
    password = data.get("password")
    if not phone_number or not password:
        return jsonify({"status": "fail", "message": "Missing phoneNumber or password"}), 400

    user = find_user_by_phone_password(phone_number, password)
    if user:
        summary_text = (
            f"{user['username']}, you have successfully logged in. "
            f"Your account type is {user['accountType']}, and your level is {user['accountLevel']}."
        )
        return jsonify({
            "userId": user['userId'],
            "username": user['username'],
            "accountLevel": user['accountLevel'],
            "accountType": user['accountType'],
            "status": "active",
            "text": summary_text
        })
    return jsonify({"status": "fail", "message": "Authentication failed"}), 401


# 💰 余额查询
@app.route('/api/balance', methods=['POST'])
def balance_query():
    data = request.get_json(silent=True) or {}
    phone_number = data.get("phoneNumber")
    password = data.get("password")
    if not phone_number or not password:
        return jsonify({"error": "Missing phoneNumber or password"}), 400

    user = find_user_by_phone_password(phone_number, password)
    if user:
        return jsonify({
            "username": user['username'],
            "accountBalance": user['accountBalance'],
            "accountLevel": user['accountLevel'],
            "text": f"{user['username']}, your current account balance is {user['accountBalance']:,.2f} RMB."
        })
    return jsonify({"error": "Unauthorized or user not found"}), 401


# 📄 交易记录查询
@app.route('/api/transactions', methods=['POST'])
def transaction_query():
    data = request.get_json(silent=True) or {}
    phone_number = data.get("phoneNumber")
    password = data.get("password")
    if not phone_number or not password:
        return jsonify({"error": "Missing phoneNumber or password"}), 400

    user = find_user_by_phone_password(phone_number, password)
    if user:
        transactions = user.get('transactions', [])
        summary = []
        for tx in transactions[:3]:  # 仅摘要前三条
            ts = tx.get('timestamp', 'N/A')
            ttype = tx.get('type', 'N/A')
            amt = tx.get('amount', 0.0)
            summary.append(f"{ts}: {ttype} of {amt:,.2f} RMB")
        summary_text = (
            f"{user['username']}, here are your recent transactions: " + "; ".join(summary) + "."
            if summary else f"{user['username']}, no recent transactions were found."
        )
        return jsonify({
            "username": user['username'],
            "transactions": transactions,
            "text": summary_text
        })
    return jsonify({"error": "Unauthorized"}), 401


# 🧠 投资产品推荐
# 入参优先级：userId > (phoneNumber + password)
# 返回字段：username, recommendedProducts[], summary（并镜像到 text，便于 Dify 映射为主输出）
@app.route('/api/recommend', methods=['POST'])
def recommend_products():
    data = request.get_json(silent=True) or {}
    print("[/api/recommend] body:", data)

    # 1) 优先使用 userId
    user_id = data.get("userId")
    user = None
    if user_id:
        user = find_user_by_id(user_id)
        print(f"  lookup by userId={user_id} -> {bool(user)}")

    # 2) 若无 userId，则允许用 phoneNumber + password 先认证
    if not user and ("phoneNumber" in data and "password" in data):
        phone = data.get("phoneNumber")
        password = data.get("password")
        user = find_user_by_phone_password(phone, password)
        print(f"  lookup by phone={phone} -> {bool(user)}")

    if not user:
        return jsonify({"error": "User not found or unauthorized. Please provide valid userId or phoneNumber/password."}), 401

    # 3) 画像获取
    profile = user_profiles.get(user["userId"], {
        "userId": user["userId"],
        "username": user["username"],
        "riskLevel": "low",
        "incomeLevel": "low",
        "investmentExperience": "none"
    })
    products = recommend_by_profile(profile)

    summary = (
        f"Hello {user['username']}, based on your profile (Risk: {profile.get('riskLevel', 'low')}, "
        f"Income: {profile.get('incomeLevel', 'low')}), we recommend: {', '.join(products)}."
    )

    return jsonify({
        "username": user["username"],
        "recommendedProducts": products,
        "summary": summary,
        "text": summary
    })


# ===================== 投诉接口 =====================

def _ensure_user_by_payload(data):
    """
    支持两种方式关联用户：
    1) 直接传 userId
    2) 传 phoneNumber + password（明文），后端校验
    （如你已实现 Bearer token，可在此处优先校验 Authorization 头）
    """
    # 1) userId
    user_id = data.get("userId")
    if user_id:
        u = find_user_by_id(user_id)
        if u:
            return u

    # 2) phoneNumber + password
    phone = data.get("phoneNumber")
    pwd   = data.get("password")
    if phone and pwd:
        u = find_user_by_phone_password(phone, pwd)
        if u:
            return u

    return None


# 创建投诉单
@app.route('/api/tickets', methods=['POST'])
def create_ticket():
    body = request.get_json(silent=True) or {}
    print("[/api/tickets] body:", body)

    # 关联用户
    user = _ensure_user_by_payload(body)
    if not user:
        return jsonify({"error": "Unauthorized or user not found"}), 401

    category = (body.get("category") or "other").lower()
    content  = body.get("content") or ""
    if not content.strip():
        return jsonify({"error": "content is required"}), 400

    data = _load_tickets()
    ticket_id = _gen_ticket_id()
    now = _now_iso()
    ticket = {
        "ticketId": ticket_id,
        "userId":   user["userId"],
        "username": user["username"],
        "category": category,
        "content":  content,
        "status":   "open",            # open -> processing -> resolved/closed
        "createdAt": now,
        "updatedAt": now
    }
    data["tickets"].append(ticket)
    _save_tickets(data)

    text = f"Ticket {ticket_id} has been created and is now open. We'll keep you updated."
    return jsonify({
        "ticketId": ticket_id,
        "status": "open",
        "text": text
    })


# 查询投诉单状态
@app.route('/api/tickets/<ticket_id>', methods=['GET'])
def get_ticket_status(ticket_id):
    data = _load_tickets()
    t = next((x for x in data["tickets"] if x["ticketId"] == ticket_id), None)
    if not t:
        return jsonify({"error": "ticket not found"}), 404
    text = f"Ticket {t['ticketId']} is currently {t['status']}."
    return jsonify({
        "ticketId": t["ticketId"],
        "status": t["status"],
        "text": text
    })


# 按用户列出工单（可用于“查看我的投诉”）
@app.route('/api/tickets/user/<user_id>', methods=['GET'])
def list_user_tickets(user_id):
    data = _load_tickets()
    items = [x for x in data["tickets"] if x["userId"] == user_id]
    text = f"You have {len(items)} tickets in total."
    return jsonify({
        "userId": user_id,
        "count": len(items),
        "tickets": items,
        "text": text
    })


if __name__ == '__main__':
    # 生产环境请改为 WSGI/反向代理，这里仅用于本地调试
    app.run(debug=True, host='0.0.0.0', port=5000)
