import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

def load_model():
    try:
        # Load the model from the file using joblib
        model = load('model.joblib')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_label_encoder():
    try:
        # Load the model from the file using joblib
        label_encoder = load('label_encoder.joblib')
        return label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
def load_missing_info():
    try:
        # Load the model from the file using joblib
        missing_feature_info = load('missing_values.joblib')
        return missing_feature_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None    

def feature_list():
    try:
        # Load the features for the model prediction
        feature_info = load('feature_list.joblib')
        return feature_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None    

model = load_model()
label_encoder = load_label_encoder()
missing_feature_mapping = load_missing_info()
features = feature_list()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])
    
        # Find and replace junk value with null
        input_df.replace(' ?', np.nan, inplace=True)

        # Strip any unwanted spaces in nominal features
        input_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        
        for feature, value in missing_feature_mapping.items():
            input_df[feature].fillna(value, inplace=True)

        for column, encoding_mapping in label_encoder.items():
            input_df[column] = input_df[column].map(encoding_mapping).fillna(-1).astype(int)

        # Apply the same transformations used during training
        transformed_input = input_df[features]  # Apply your transformations based on the transformation_data

        # Make predictions using the model
        prediction = model.predict(transformed_input)

        if prediction:
            target_class = '50000+'
        else:
            target_class = '-50000'
            
        # Return the prediction as JSON
        return jsonify({'prediction': target_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
