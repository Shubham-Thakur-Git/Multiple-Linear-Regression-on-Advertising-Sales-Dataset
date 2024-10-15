from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('lr3.pkl','rb'))
scaler = pickle.load(open('scalar.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Step 1: Extract features from the form submission
    int_features = [int(x) for x in request.form.values()]

    # Step 2: Scale the input features using the saved scaler
    int_features[0] = scaler.transform([int_features[0]])
    int_features[1] = scaler.transform([int_features[1]])

    final_features = [np.array(int_features)]  # Reshape input into 2D array (1 sample, 2 features)
    #final_features = np.array(int_features).reshape(-1,1)
    
    
    # Step 3: Make prediction using the scaled features
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Predicted Sales in million dollars is {output}')

if __name__ == "__main__":
    app.run(debug=True)
