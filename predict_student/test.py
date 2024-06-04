import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import sys

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            print("here")
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            print(preds)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

def main():
    # Read the sample data from test.csv
    csv_file_path = os.path.join("artifacts", "test.csv")
    df = pd.read_csv(csv_file_path)

    # Create an instance of PredictPipeline
    predict_pipeline = PredictPipeline()

    # Make predictions for each sample and compare with actual math scores
    predictions = []
    for i in range(len(df)):
        sample_data = CustomData(
            gender=df.iloc[i]['gender'],
            race_ethnicity=df.iloc[i]['race_ethnicity'],
            parental_level_of_education=df.iloc[i]['parental_level_of_education'],
            lunch=df.iloc[i]['lunch'],
            test_preparation_course=df.iloc[i]['test_preparation_course'],
            reading_score=df.iloc[i]['reading_score'],
            writing_score=df.iloc[i]['writing_score']
        )

        input_df = sample_data.get_data_as_data_frame()
        print(f"Input DataFrame for sample {i + 1}:")
        print(input_df)

        pred = predict_pipeline.predict(input_df)
        predictions.append(pred[0])

    # Compare predictions with actual math scores
    df['predicted_math_score'] = predictions
    print(df)

    # Evaluate the model's performance
    mae = mean_absolute_error(df['math_score'], df['predicted_math_score'])
    rmse = np.sqrt(mean_squared_error(df['math_score'], df['predicted_math_score']))
    r2 = r2_score(df['math_score'], df['predicted_math_score'])

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    main()
