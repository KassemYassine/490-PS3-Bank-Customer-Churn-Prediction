import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_and_split_data(file_path):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)
    # Select the first 500 entries
    df = df.head(500)
    
    # Drop the 'CustomerId' and 'Surname' columns as they are not needed
    df = df.drop(['CustomerId','Surname'], axis=1)
    
    # Convert 'Geography' to numerical values using get_dummies
    df = pd.get_dummies(df, columns=['Geography'])
    
    # Convert 'Gender' to binary (0 and 1) where Male = 0 and Female = 1
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    
    # Splitting the data into features and target variable
    X = df.drop('Exited', axis=1)  # Features
    y = df['Exited']  # Target variable
    
    # Split the dataset into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a StandardScaler object and fit to the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
