import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Memuat dataset dari file CSV"""
    df = pd.read_csv(filepath)
    print(f" Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_missing_values(df):
    """Mengecek dan melaporkan missing values"""
    missing = df.isnull().sum().sum()
    print(f" Total missing values: {missing}")
    return df


def remove_duplicates(df):
    """Menghapus data duplikat"""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f" Duplikat dihapus: {before - after} baris, sisa {after} baris")
    return df


def clean_gender(df):
    """Menghapus baris dengan gender 'Other' karena jumlahnya sangat sedikit"""
    before = df.shape[0]
    df = df[df['gender'] != 'Other']
    after = df.shape[0]
    print(f" Gender 'Other' dihapus: {before - after} baris, sisa {after} baris")
    return df


def handle_smoking_history(df):
    """Mengganti nilai 'No Info' dengan modus smoking_history"""
    modus = df[df['smoking_history'] != 'No Info']['smoking_history'].mode()[0]
    df['smoking_history'] = df['smoking_history'].replace('No Info', modus)
    print(f" 'No Info' diganti dengan modus: '{modus}'")
    return df


def encode_categorical(df):
    """Encoding kolom kategorikal menjadi numerik"""
    # Encoding gender
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    # Encoding smoking_history
    smoking_mapping = {
        'never': 0,
        'former': 1,
        'ever': 2,
        'not current': 3,
        'current': 4
    }
    df['smoking_history'] = df['smoking_history'].map(smoking_mapping)
    print(" Encoding kategorikal selesai")
    return df


def handle_outliers(df):
    """Menangani outlier pada kolom BMI menggunakan metode IQR"""
    Q1 = df['bmi'].quantile(0.25)
    Q3 = df['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    df['bmi'] = df['bmi'].clip(lower=batas_bawah, upper=batas_atas)
    print(f" Outlier BMI di-clip: [{batas_bawah:.2f}, {batas_atas:.2f}]")
    return df


def split_data(df):
    """Membagi dataset menjadi training dan testing"""
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" Split data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Menangani imbalanced data menggunakan SMOTE"""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f" SMOTE selesai: {X_train_smote.shape[0]} baris training")
    return X_train_smote, y_train_smote


def scale_features(X_train, X_test):
    """Standarisasi fitur menggunakan StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(" Standarisasi fitur selesai")
    return X_train_scaled, X_test_scaled, scaler


def save_results(X_train_scaled, y_train_smote, X_test_scaled, y_test, columns, output_dir):
    """Menyimpan hasil preprocessing ke file CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # Simpan training set
    train_df = pd.DataFrame(X_train_scaled, columns=columns)
    train_df['diabetes'] = y_train_smote.values
    train_df.to_csv(f'{output_dir}/diabetes_train.csv', index=False)

    # Simpan testing set
    test_df = pd.DataFrame(X_test_scaled, columns=columns)
    test_df['diabetes'] = y_test.values
    test_df.to_csv(f'{output_dir}/diabetes_test.csv', index=False)

    print(f"   Hasil disimpan di folder '{output_dir}'")
    print(f"   Training set: {train_df.shape}")
    print(f"   Testing set: {test_df.shape}")


def main():
    """Fungsi utama untuk menjalankan seluruh pipeline preprocessing"""
    print("="*50)
    print("PIPELINE PREPROCESSING DIABETES DATASET")
    print("="*50)

    # Path input dan output
    input_path = 'diabetes_raw/diabetes_prediction_dataset.csv'
    output_dir = 'preprocessing/diabetes_preprocessing'

    # Jalankan pipeline
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = clean_gender(df)
    df = handle_smoking_history(df)
    df = encode_categorical(df)
    df = handle_outliers(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_smote, X_test)

    save_results(X_train_scaled, y_train_smote, 
                 X_test_scaled, y_test, 
                 X_train.columns, output_dir)

    print("="*50)
    print("PREPROCESSING SELESAI!")
    print("="*50)


if __name__ == '__main__':
    main()