import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
def preprocessing():
    # VNPT churn
    '''
    dataset để dự đoán xem khách hàng có rời bỏ công ty cung cấp dịch vụ viễn thông trong quý 3 năm nay hay k
    Churn value: 0 = khách hàng ở lại với công ty, 1 = khách hàng rời bỏ công ty
    Churn score (0-100): score càng cao thể hiện mức độ khách hàng rời bỏ công ty càng cao
    CLTV : Giá trị trọn đời của khách hàng.Giá trị càng cao, khách hàng càng có giá trị.

    '''
    # df.isnull().sum()
    # dataset có 7043 sample nhưng có 5174 sample k có data trong Churn Reason => drop churn reason
    # churn label giống churn value lat long trùng latitude longitude => drop churn label, lat long
    # ngoài ra còn có 1 số cột k mang giá trị phân tích như cột id customer, state, country, count(toàn giá trị 1), Internet Service, Contract(month2month) => drop



    data = "../churn.csv"
    dt = pd.read_csv(data,engine='python')

    df = pd.DataFrame(dt)
    df = df.drop(columns=['Churn Reason', 'Churn Label', 'CustomerID', 'State', 'Country', 'Count', 'Lat Long',
                          'Internet Service', 'Contract'], axis=1)
    # print(df)

    '''object => number'''

    label_encoder = LabelEncoder()
    df['City'] = label_encoder.fit_transform(df['City'])
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['Senior Citizen'] = label_encoder.fit_transform(df['Senior Citizen'])
    df['Zip Code'] = label_encoder.fit_transform(df['Zip Code'])
    df['Partner'] = label_encoder.fit_transform(df['Partner'])
    df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
    df['Phone Service'] = label_encoder.fit_transform(df['Phone Service'])
    df['Multiple Lines'] = label_encoder.fit_transform(df['Multiple Lines'])
    df['Online Security'] = label_encoder.fit_transform(df['Online Security'])
    df['Online Backup'] = label_encoder.fit_transform(df['Online Backup'])
    df['Device Protection'] = label_encoder.fit_transform(df['Device Protection'])
    df['Tech Support'] = label_encoder.fit_transform(df['Tech Support'])
    df['Streaming TV'] = label_encoder.fit_transform(df['Streaming TV'])
    df['Streaming Movies'] = label_encoder.fit_transform(df['Streaming Movies'])
    df['Paperless Billing'] = label_encoder.fit_transform(df['Paperless Billing'])
    df['Payment Method'] = label_encoder.fit_transform(df['Payment Method'])
    df['Total Charges'] = df['Total Charges'].replace(' ', np.nan)
    df['Total Charges'] = df['Total Charges'].astype(float)  # object => float

    # print(df)
    #
    # thay giá trị nan bằng giá trị trung bình cho cột total charges
    df['Total Charges'].fillna(df['Total Charges'].mean(), inplace=True)
    #kiểm tra missing data = 0
    # df.isnull().sum()
    #tất cả các dữ liệu chữ đã đưuọc chuyển thành số
    # df.info()
    '''
    scaling: cột được trừ đi giá trị trung bình của nó và chia cho độ lệch chuẩn của nó
    '''

    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(df[['Latitude']])
    df['Latitude'] = scaled_column
    scaled_column = scaler.fit_transform(df[['Longitude']])
    df['Longitude'] = scaled_column
    scaled_column = scaler.fit_transform(df[['Monthly Charges']])
    df['Monthly Charges'] = scaled_column
    scaled_column = scaler.fit_transform(df[['Total Charges']])
    df['Total Charges'] = scaled_column
    scaled_column = scaler.fit_transform(df[['Churn Score']])
    df['Churn Score'] = scaled_column
    scaled_column = scaler.fit_transform(df[['CLTV']])
    df['CLTV'] = scaled_column
    scaled_column = scaler.fit_transform(df[['Tenure Months']])
    df['Tenure Months'] = scaled_column

    # print(df)

    '''
    mất cân bằng dữ liệu
    SMOTE tạo ra các mẫu nhân tạo cho lớp thiểu số bằng cách kết hợp các mẫu gần nhau
    '''

    # count = df['Churn Value'].value_counts()[1]
    # print("Số lượng giá trị bằng 1 trong cột: ", count)
    # có 1869 sample = 1 trong tổng số 7043 sample => mất cân bằng


    # Tạo một đối tượng SMOTE
    smote = SMOTE()

    # Chuẩn bị dữ liệu
    X = df.drop('Churn Value', axis=1)  # Xác định các đặc trưng (features)
    y = df['Churn Value']  # Xác định nhãn (labels)

    # Áp dụng SMOTE để oversampling
    X_smote, y_smote = smote.fit_resample(X, y)

    # # X_smote là tập dữ liệu đã được oversampling, y_smote là tập nhãn tương ứng
    # X_smote
    # # giá trị 1 và 0 trong cột Churn value bằng nhau = 5174
    count = y_smote.value_counts()
    # '''chọn ngưỡng là |0.15| chọn được 12 feature'''
    X_smote = X_smote.drop(columns=['City', 'Zip Code', 'Latitude', 'Longitude', 'Gender', 'Phone Service', 'Multiple Lines',
                              'Streaming TV', 'Streaming Movies', 'Payment Method', 'CLTV'], axis=1)
    # print(X_smote)
    df = pd.concat([ X_smote, y_smote],axis=1)
    # print (df)
    '''
    loai bo ngoai lai
    '''
    min_thresold, max_thresold = df['Tenure Months'].quantile([0.001, 0.999])
    df2 = df[(df['Tenure Months'] < max_thresold) & (df['Tenure Months'] > min_thresold)]
    min_thresold, max_thresold = df2['Monthly Charges'].quantile([0.001, 0.999])
    df3 = df2[(df2['Monthly Charges'] < max_thresold) & (df2['Monthly Charges'] > min_thresold)]
    min_thresold, max_thresold = df3['Total Charges'].quantile([0.001, 0.999])
    df4 = df3[(df3['Total Charges'] < max_thresold) & (df3['Total Charges'] > min_thresold)]
    min_thresold, max_thresold = df4['Churn Score'].quantile([0.001, 0.999])
    df5 = df4[(df4['Churn Score'] < max_thresold) & (df4['Churn Score'] > min_thresold)]
    # min_thresold, max_thresold = df5['CLTV'].quantile([0.001, 0.999])
    # df6 = df5[(df['CLTV'] < max_thresold) & (df['CLTV'] > min_thresold)]
    print(df5)
    X = df5.drop('Churn Value', axis=1)
    y = df5['Churn Value']
    return X, y