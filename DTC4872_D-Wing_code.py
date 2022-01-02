import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import random
random.seed(10)
np.random.seed(42)
warnings.filterwarnings('ignore')
user_input = input("Enter the path of your file: ")

assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
    
data = pd.read_csv(user_input)
test=data.copy()
data=data.rename(columns={"Seat Fare Type 1":"f1","Seat Fare Type 2":"f2"})
data=data.dropna(subset=["f1","f2"], how='all')
test=data.copy()
test["Service Date"]=pd.to_datetime(test["Service Date"],format='%d-%m-%Y %H:%M')
test["RecordedAt"]=pd.to_datetime(test["RecordedAt"],format='%d-%m-%Y %H:%M')
test["Service Date"]-test["RecordedAt"]
data["timediff"]=test["Service Date"]-test["RecordedAt"]
test["timediff"]=test["Service Date"]-test["RecordedAt"]
days=test["timediff"].dt.days
hours=test["timediff"].dt.components["hours"]
mins=test["timediff"].dt.components["minutes"]
test["abstimediff"]=days*24*60+hours*60+mins
test["f1"]=test["f1"].astype(str)
test["f1_1"]=test.f1.str.split(',')
#print(test)
test["f2"]=test["f2"].astype(str)
test["f2_1"]=test.f2.str.split(',')
test=test.reset_index(drop=True)
arr=[]
var=[]
for i in range(0,len(test["f1_1"])):
    if test["f1_1"][i][0]=='nan':
        arr.append(pd.to_numeric(test["f2_1"][i]).mean())
        var.append(pd.to_numeric(test["f2_1"][i]).std())
    #print(x)
    else:
        arr.append(pd.to_numeric(test["f1_1"][i]).mean())
        var.append(pd.to_numeric(test["f1_1"][i]).std())
test["meanfare"]=arr
test["devfare"]=var
test["abstimediff"]=(test["abstimediff"]-test["abstimediff"].mean())/test["abstimediff"].std()
test["meanfare"]=(test["meanfare"]-test["meanfare"].mean())/test["meanfare"].std()
test["is_type1"]=1
test.loc[test["f1"]=='nan',"is_type1"]=0
test["devfare"]=(test["devfare"]-test["devfare"].mean())/test["devfare"].std()
processed_data = test
#print(processed_data)

data = processed_data
data["is_weekend"]=0
data.loc[data["Service Date"].dt.dayofweek==5,"is_weekend"]=1
data.loc[data["Service Date"].dt.dayofweek==6,"is_weekend"]=1
data_copy=data.copy()
data=data.drop(["f1","f2","Service Date","RecordedAt","timediff","f1_1","f2_1"],axis=1)
data["maxtimediff"]=data["abstimediff"]
data=data_copy
data=data.drop(["f1","f2","Service Date","RecordedAt","timediff","f1_1","f2_1"],axis=1)
#print(data)
data=data.groupby("Bus").agg(['mean','max'])
data=data.drop([( 'is_weekend',  'max'),(   'is_type1',  'max'),],axis=1)
data=data.drop([(    'devfare',  'max'),(   'meanfare',  'max'),],axis=1)
data_copy=data.copy()
data=data_copy
data.columns = ['{}_{}'.format(x[0], x[1]) for x in data.columns]
#print(data)
data=data.reset_index()
X=data.drop("Bus",axis=1)

features = X
#data = features
#print(data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
model1 = KMeans(n_clusters=6) 
model1.fit(pca_result)
centroids1 = model1.cluster_centers_
labels = model1.labels_
bus = data["Bus"]
bus=pd.DataFrame(bus)
y=pd.concat((bus,pd.DataFrame(pca_result),pd.DataFrame(labels,columns = ["Cluster"])),axis=1)
y = y.rename(columns = {0:"pca1",1:"pca2"})
# print(y)
cluster=[]
for i in range(6):
    cluster.append(y[y["Cluster"]==i])

# print(labels)
    
X0=cluster[0][["pca1","pca2"]].to_numpy()
m0 = KMeans(n_clusters=2) 
m0.fit(X0)
X1=cluster[1][["pca1","pca2"]].to_numpy()
m1 = KMeans(n_clusters=7) 
m1.fit(X1)
X2=cluster[2][["pca1","pca2"]].to_numpy()
m2 = KMeans(n_clusters=6) 
m2.fit(X2)
X3=cluster[3][["pca1","pca2"]].to_numpy()
m3 = KMeans(n_clusters=3) 
m3.fit(X3)
X4=cluster[4][["pca1","pca2"]].to_numpy()
m4 = KMeans(n_clusters=2) 
m4.fit(X4)
X5=cluster[5][["pca1","pca2"]].to_numpy()
m5 = KMeans(n_clusters=6) 
m5.fit(X5)

def leader_follower(cluster):  #only bus and prob for a particular cluster sorted
    cluster["Follows"] = ""
    cluster["Confidence Score 1"] = ""
    cluster["Is followed by"] = ""
    cluster["Confidence Score 2"] = ""
    maxprob = cluster["Probability"][0]
    leader = cluster["Bus"][0]
    #confidence_score_1 = cluster["Probability"][0]
    cluster["Follows"][0] = "Independent"
    cluster["Confidence Score 1"][0] = 1-cluster["Probability"][0]
    #confidence_score_2 = 
    if len(cluster)==1:
        return cluster
    follower = cluster["Bus"][1]
    for i in range(1,len(cluster)):
        cluster["Follows"][i] = leader
        cluster["Confidence Score 1"][i] = cluster["Probability"][i]/cluster["Probability"][i-1]
        leader = cluster["Bus"][i]
        #confidence_score_1 = cluster["Probability"][i]
    for i in range(0,len(cluster)-1):
        cluster["Is followed by"][i] = follower
        follower = cluster["Bus"][i+1]
        cluster["Confidence Score 2"][i] = cluster["Probability"][i+1]/cluster["Probability"][i]
    #cluster["Is followed by"][len(cluster)-1] = ""
    #cluster["Confidence Score 2"][i]
            
    return cluster

def dist_from_own_centre(pca_result,centroids,labels):
    arr=np.zeros(len(labels))
    for i in range(len(labels)):
        arr[i]=1/((np.sum((pca_result[i] - centroids[labels[i]])**2))**0.5+1e-8)
    return arr
def dist_from_other_centre(pca_result,centroids,labels):
    arr=np.zeros(len(labels))
    for i in range(len(labels)):
        for j in range(len(centroids)):
            arr[i] += 1/((np.sum((pca_result[i] - centroids[j])**2))**0.5+1e-8)
    return arr

prob0 = dist_from_own_centre(X0,m0.cluster_centers_,m0.labels_)/dist_from_other_centre(X0,m0.cluster_centers_,m0.labels_)
cluster[0]["Probability"] = prob0
cluster[0]["labels"] = m0.labels_
output=[]
result=[]
for i in range(max(m0.labels_)+1):
    output.append(cluster[0][cluster[0]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
Y0 = result[0]
for i in range(1,len(result)):
    Y0 = pd.concat((Y0,result[i]))
Y0=Y0.set_index("index")
# print(Y0)
prob1 = dist_from_own_centre(X1,m1.cluster_centers_,m1.labels_)/dist_from_other_centre(X1,m1.cluster_centers_,m1.labels_)
cluster[1]["Probability"] = prob1
cluster[1]["labels"] = m1.labels_

output=[]
result=[]
for i in range(max(m1.labels_)+1):
    output.append(cluster[1][cluster[1]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
    
Y1 = result[0]
for i in range(1,len(result)):
    Y1 = pd.concat((Y1,result[i]))
    
Y1=Y1.set_index("index")

prob2 = dist_from_own_centre(X2,m2.cluster_centers_,m2.labels_)/dist_from_other_centre(X2,m2.cluster_centers_,m2.labels_)
cluster[2]["Probability"] = prob2
cluster[2]["labels"] = m2.labels_

output=[]
result=[]
for i in range(max(m2.labels_)+1):
    output.append(cluster[2][cluster[2]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
    
Y2 = result[0]
for i in range(1,len(result)):
    Y2 = pd.concat((Y2,result[i]))
    
Y2=Y2.set_index("index")

prob3 = dist_from_own_centre(X3,m3.cluster_centers_,m3.labels_)/dist_from_other_centre(X3,m3.cluster_centers_,m3.labels_)
cluster[3]["Probability"] = prob3
cluster[3]["labels"] = m3.labels_

output=[]
result=[]
for i in range(max(m3.labels_)+1):
    output.append(cluster[3][cluster[3]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
    
Y3 = result[0]
for i in range(1,len(result)):
    Y3 = pd.concat((Y3,result[i]))
    
Y3=Y3.set_index("index")

prob4 = dist_from_own_centre(X4,m4.cluster_centers_,m4.labels_)/dist_from_other_centre(X4,m4.cluster_centers_,m4.labels_)
cluster[4]["Probability"] = prob4
cluster[4]["labels"] = m4.labels_

output=[]
result=[]
for i in range(max(m4.labels_)+1):
    output.append(cluster[4][cluster[4]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
    
Y4 = result[0]
for i in range(1,len(result)):
    Y4 = pd.concat((Y4,result[i]))
    
Y4=Y4.set_index("index")

prob5 = dist_from_own_centre(X5,m5.cluster_centers_,m5.labels_)/dist_from_other_centre(X5,m5.cluster_centers_,m5.labels_)
cluster[5]["Probability"] = prob5
cluster[5]["labels"] = m5.labels_

output=[]
result=[]
for i in range(max(m5.labels_)+1):
    output.append(cluster[5][cluster[5]["labels"]==i])
    output[i] = output[i].sort_values("Probability",ascending = False)
    output[i] = output[i].reset_index()
    result.append(leader_follower(output[i]))
    
Y5 = result[0]
for i in range(1,len(result)):
    Y5 = pd.concat((Y5,result[i]))
    
Y5=Y5.set_index("index")

output = pd.concat((Y0,Y1,Y2,Y3,Y4,Y5))
output_copy = output.copy()
output = output.drop(columns=["pca1","pca2","Cluster","Probability","labels"])

output = output.reset_index("index")
output = output.sort_values("index")
output = output.set_index("index")
output = output.set_index("Bus")

output.to_csv("DTC4872_D-Wing_output.csv")

