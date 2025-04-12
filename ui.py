from flask import Flask, render_template, request, jsonify
from prompt_engine import get_chat_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input", "")
    response = get_chat_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
