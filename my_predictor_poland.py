import pickle
import pandas as pd

def wrangle(filepath):
    df = pd.read_csv(filepath)
    
    # Keep only Year 5 data 
    df = df[df["year"] == 5]
    
    # Drop 'class' and 'year' 
    df = df.drop(columns=["class", "year"], errors="ignore")
    
    # Name the index explicitly
    df.index.name = 'Company_ID'
    
    return df

def make_predictions(data_filepath, model_filepath):
    # Load and process test data
    X_test = wrangle(data_filepath)
    
    # Load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    
    # Generate predictions
    y_test_pred = model.predict(X_test)
    
    # Convert to Pandas Series
    y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="class")
    
    return y_test_pred
