import pandas as pd
import json
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

#with open('musae_facebook_features.json') as json_file:
#    data = json.load(json_file)
#
#
#max_length = 0
#for key, value in data.items():
#    if isinstance(value, list):
#        array_length = len(value)
#        if array_length > max_length:
#            max_length = array_length
#
#print(max_length)
#
#for key, value in data.items():
#    copy = value.copy()
#    deficit = max_length - len(copy)
#    for i in range(0, deficit):
#        copy.append("?")
#    data[key] = copy
#
#sorted_json_data = {k: data[k] for k in sorted(data)}
#
#data = sorted_json_data
#
#with open('data.csv', 'w', newline='') as csvfile:
#    fieldnames = ['Node_ID', 'Profile_Name', 'Profile_Picture_URL', 'Profile_URL', 'Profile_Description/Bio',
#        'Profile_Location', 'Profile_Education', 'Profile_Work_Experience', 'Friend_Count', 'Friend_IDs',
#        'Circle_Name', 'Circle_Description', 'Circle_Privacy_Settings', 'Ego_Network_Size', 'Ego_Network_IDs',
#        'Mutual_Friends', 'Friendship_Start_Date', 'Last_Interaction_Date', 'Shared_Interests',
#        'Communication_Frequency', 'Common_Groups/Pages_Liked', 'Messages_Sent', 'Post_Frequency',
#        'Tagged_Photos_Count', 'Relationship_Status', 'Birthday', 'Languages_Spoken', 'User_Interactions',
#        'Social_Engagement_Level', 'Time_Zone', 'Last_Online_Status'
#    ]
#
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#    writer.writeheader()  # Write CSV header with field names
#
#    for key, value in sorted_json_data.items():
#        if len(value) == len(fieldnames):
#            row_dict = {fieldnames[i]: value[i] for i in range(len(fieldnames))}
#            writer.writerow(row_dict)
#
##imputer = SimpleImputer(strategy='mean')
##
##df.replace('?', np.nan, inplace=True)
##
##df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
##
##nan_values = df.isna().sum().sum()  # Sum of all NaN values
##
##print(f'there are {nan_values} nan values present')
#
#import numpy as np
#
#print(data["554"])
#
#df = pd.read_csv('data.csv')
#
#
#imputer = SimpleImputer(strategy='mean')
# 
#df.replace('?', np.nan, inplace=True)
# 
#df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# 
#nan_values = df.isna().sum().sum()  # Sum of all NaN values
# 
#print(f'there are {nan_values} nan values present')
#
#
##print(df.head())
#
##print(df.iloc[553])
#
#print(df.shape)
#
#
#target = pd.read_csv('musae_facebook_target.csv')
#
#df["class"] = target["page_type"]
#
#print(df.shape)
#
#df.to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')

label_encoder = LabelEncoder()

# Fit label encoder and transform data
encoded_data = label_encoder.fit_transform(df["class"])

df["class"] = encoded_data
