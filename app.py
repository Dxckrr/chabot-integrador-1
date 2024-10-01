from flask import Flask, jsonify, request
from flask_cors import CORS
from model.bioGPT import get_chatbot_response
# Sanavit's chatbot implementation bioGPT model 
app = Flask(__name__)

# Permitir CORS desde localhost:5173 y permitir métodos POST y OPTIONS
CORS(app, resources={r"/chatbot": {"origins": "http://localhost:5173", "methods": ["POST"]}})

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.json
    print(data)
    if 'message' not in data:
        return jsonify({'error': 'Missing message parameter'}), 400

    # input
    user_message = data['message']
    # bioGPT call
    response = get_chatbot_response(user_message)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
