from flask import Flask, jsonify, request
from model.bioGPT import get_chatbot_response
# Sanavit's chatbot implementation bioGPT model 
app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'Missing message parameter'}), 400

    # input
    user_message = data['message']
    
    # bioGPT call
    response = get_chatbot_response(user_message)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
