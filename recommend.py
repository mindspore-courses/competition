from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# 加载用户画像数据
with open('user_profiles.json', 'r', encoding='utf-8') as f:
    user_profiles = json.load(f)["users"]

# 产品推荐规则（可拓展）
def recommend_products(profile):
    risk = profile["riskLevel"]
    income = profile["incomeLevel"]
    experience = profile["investmentExperience"]

    # 规则引擎示例（可根据实际需求调整）
    if risk == "high":
        return [
            "Tech Growth Fund",
            "Crypto Index ETF",
            "AI Startups Portfolio"
        ]
    elif risk == "medium":
        if experience in ["advanced", "moderate"]:
            return [
                "Global Equity Fund",
                "Balanced Income Fund",
                "Emerging Market ETF"
            ]
        else:
            return [
                "Balanced Income Fund",
                "Index Bond Fund"
            ]
    else:  # risk = low
        return [
            "Government Bond Fund",
            "Fixed Income ETF",
            "Capital Protection Plan"
        ]

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get("userId")

    user = next((u for u in user_profiles if u["userId"] == user_id), None)
    if not user:
        return jsonify({"error": "User not found"}), 404

    products = recommend_products(user)
    summary = f"Hello {user['username']}, based on your profile (Risk: {user['riskLevel']}, Income: {user['incomeLevel']}), we recommend: {', '.join(products)}."

    return jsonify({
        "userId": user["userId"],
        "username": user["username"],
        "recommendedProducts": products,
        "summary": summary
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
