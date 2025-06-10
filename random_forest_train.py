import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


import joblib


#dataloader-
df_raw=pd.read_csv("logs/simulation_log.csv")

#print(df_raw.head())
#print(df_raw["Slice"].value_counts())

df=pd.get_dummies(df_raw,"Slice")
#print(df.head())

X=df[["Slice_eMBB","Slice_URLLC","Slice_mMTC","Asked","Admitted","Dropped"]]
y=df["SLA_violation"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=20)

clf = RandomForestClassifier(n_estimators=100, random_state=43)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

joblib.dump(clf, "models/rf_model.pkl")
