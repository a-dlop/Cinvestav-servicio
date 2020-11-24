# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:06:34 2020

@author: Daniel
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import csv
import pickle

def importance(modelo,num, importype):
    """"Recibe modelo y nombre, regresa una gráfica de las features en orden de importancia de menor a mayor\n
        Tiene 5 criterios, el default weight. Los demás son gain,cover, total_gain, total_cover
    """
    
    xgb_fea_imp=pd.DataFrame(list(modelo.get_booster().get_score(importance_type=importype).items()),
    columns=['Caracteristica','importance by:'+str(importype)]).sort_values('importance by:'+str(importype), ascending=False)
    print('',xgb_fea_imp)
    #ax=xgb_fea_imp.plot.hist()    
    xgb_fea_imp.to_csv('xgb_feat_imp'+str(num)+'.csv')

    from xgboost import plot_importance
    
    ax = plot_importance(modelo,importance_type=importype,title='Feature importance by: '+importype )
    ax.figure.savefig('xgb_feat_imp'+str(num)+'by: '+str(importype)+'.png')

def Openfile(nombre,b):
    """La función toma el nombre de un archivo pkl y lo asigna a una variable dataframe de pandas
    Además agrega las etiquetas a la columna
    """
    data=pd.read_pickle(nombre)
    data=data[b]
    return  data
def classification(signal,bkgnd):
    """La función recibe dos dataframes y Clasifica las señales como ruido de fondo con 0 y señal por 1, regresa un dataframe con los 
    datos de ambos dataframes, concatenados"""
    bkgnd['signal/bkgnd'] = 0
    signal['signal/bkgnd'] = 1
    df=pd.concat([signal,bkgnd])
    return df
def sets(df,signal,bkgnd):
    """Recibe los datos del set de señal, fondo y el dataframe concatenado por la función classification\n
    Aplica train_test_split y crea 8 datasets 
    """
    train_x = df.drop(['signal/bkgnd'], axis=1) #features = all minus (signal/bkgnd and masses)
    train_y = df['signal/bkgnd'] 
    
    signal_x = signal.drop(['signal/bkgnd'], axis=1) 
    signal_y = signal['signal/bkgnd']
    
    bkgnd_x = bkgnd.drop(['signal/bkgnd'], axis=1)
    bkgnd_y = bkgnd['signal/bkgnd']

    train_signal_x, test_signal_x, train_signal_y, test_signal_y = train_test_split(signal_x, signal_y, 
                                                  test_size=0.5, 
                                                  random_state=1)
    train_bkgnd_x, test_bkgnd_x, train_bkgnd_y, test_bkgnd_y = train_test_split(bkgnd_x, bkgnd_y, 
                                                      test_size=0.5, 
                                                      random_state=1)
    
    test_x = test_signal_x.append(test_bkgnd_x)
    test_y = test_signal_y.append(test_bkgnd_y)
    
    train_x = train_signal_x.append(train_bkgnd_x)
    train_y = train_signal_y.append(train_bkgnd_y)
    return test_x,test_y,train_x,train_y,train_signal_x, test_signal_x, train_signal_y, test_signal_y,train_bkgnd_x, test_bkgnd_x, train_bkgnd_y, test_bkgnd_y
def prediction(model,test_signal_x,test_signal_y,test_bkgnd_x,test_bkgnd_y,train_signal_x,train_bkgnd_x):
    """Toma los sets obtenidos por la función set y aplica el modelo de clasificación deseado
    regresa cuatro sets con la señales reales y las señales de fondo predichas en sets de entrenamiento y prueba
    """
    predict_signal_test =  model.predict_proba(test_signal_x)[:,1]
    predict_signal_train =  model.predict_proba(train_signal_x)[:,1]
    
    predict_back_test = model.predict_proba(test_bkgnd_x)[:,1]
    predict_back_train = model.predict_proba(train_bkgnd_x)[:,1]
    return predict_signal_test,predict_signal_train,predict_back_test,predict_back_train

def signaloverfit(predict_signal_test,predict_signal_train):
    """Grafica los histogramas de la señal predicha y la señal de entrenamiento para observar si hay overfitting
    en los sets de señales
    """
    plt.figure(figsize=(10,7))
    plt.title('Signal overfit')
    m = plt.hist(predict_signal_test, bins=20, label='Test', color='olive')
    plt.hist(predict_signal_train, bins=m[1], label='Train', 
             histtype='stepfilled', facecolor='none', edgecolor='red',
            hatch='////')
    plt.legend(fontsize=13)
    plt.yscale('log')
    plt.show()
    
def overfitting(predict_signal_test,predict_signal_train,predict_back_test,predict_back_train):
    """"Grafica los sets de señales  y fondo para de prueba y entrenamiento"""
    plt.figure(figsize=(10,7))
    plt.title('Overfitting')
    m = plt.hist(predict_signal_test, bins=20, label='Test Signal', facecolor='lightgreen', edgecolor="darkgreen",
                 histtype='stepfilled', alpha=0.8)
    train = np.histogram(predict_signal_train, bins=m[1])
    bins_ = (train[1][1:]+train[1][:-1])/2
    plt.errorbar(bins_, train[0],yerr=np.sqrt(train[0]), 
                     fmt='s', marker='o' ,color='darkgreen',
                    barsabove=False, capsize=5, label='Train Signal')
    
    plt.hist(predict_back_test, bins=m[1], label='Test Background',
                 histtype='stepfilled', facecolor='none', edgecolor='crimson', hatch= '\\\\', alpha=0.8)
    train2 = np.histogram(predict_back_train, bins=m[1])
    bins_2 = (train2[1][1:]+train2[1][:-1])/2
    plt.errorbar(bins_2, train2[0],yerr=np.sqrt(train[0]), 
                     fmt='s', marker='o' ,color='red',
                    barsabove=False, capsize=5, label='Train Background')
    plt.legend(fontsize=13)
    plt.yscale('log')
    plt.ylim(1, 200000)
    plt.xlim(0, 1)
    plt.ylabel('Events', fontsize=13)
    plt.xlabel('XGBoost output', fontsize=13)
    plt.show()

    
def roc(test_x,test_y,train_x,train_y):
    """"
    Presenta la curva roc, que muestra la precisión del clasificador, entre mas  
    cercana sea el area 1 mejor será el clasificador """
    plt.figure(figsize=(10,7))
    plt.title('ROC curve', fontsize=12)
    model_predict = model.predict_proba(test_x)
    model_predict = model_predict[:,1]
    auc_score = roc_auc_score(test_y, model_predict) 
    fpr, tpr, _ = roc_curve(test_y, model_predict) #roc_curve(true binary labels, prediction scores)
    print('Test : ', auc_score)
    plt.plot(tpr, 1-fpr, label='Test   '+ str(round(auc_score, 4)), color='purple')
    
    model_predict = model.predict_proba(train_x)
    model_predict = model_predict[:,1]
    auc_score = roc_auc_score(train_y, model_predict)
    fpr, tpr, _ = roc_curve(train_y, model_predict)
    plt.plot(tpr, 1-fpr, label='Train   ' + str(round(auc_score,4)) , color='orange', linewidth=3)
    print('Train : ', auc_score)
    plt.legend(loc='best',fontsize=13)
    plt.ylabel('background rejection', fontsize=13)
    plt.xlabel('Signal efficiency', fontsize=13)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylim(0.7,1)
    plt.show() 

g=['Signal_2.pkl','leftSB_2.pkl']
k=['Signal_5.pkl','leftSB_5.pkl']
l=['Signal_10.pkl','leftSB_10.pkl']
b=['Bpt' ,'kpt' ,'PDL' ,'prob' ,'cosA' ,'signLxy']
files=[g,k,l]

signal=Openfile(l[0],b)
bkgnd=Openfile(l[1],b)
df=classification(signal,bkgnd)
test_x,test_y,train_x,train_y,train_signal_x, test_signal_x, train_signal_y, test_signal_y,train_bkgnd_x, test_bkgnd_x, train_bkgnd_y, test_bkgnd_y=sets(df,signal,bkgnd)
rocparams=list()

model = xgb.XGBClassifier(objective = 'binary:logistic', 
                          #learning_rate=1    
                          n_estimators= 200,
 max_depth= 7,
 min_child_weight= 4,
 gamma= 0,
 subsample= 1,
 colsample_bytree= 1,
 reg_alpha= 0.01,
 reg_lambda= 800)
model.fit(train_x, train_y)

predict_signal = model.predict(test_x)

predict_signal_test,predict_signal_train,predict_back_test,predict_back_train=prediction(model,test_signal_x,test_signal_y,test_bkgnd_x,test_bkgnd_y,train_signal_x,train_bkgnd_x)
print(np.round(accuracy_score(test_y,predict_signal)*100, 2),'%')


importanceby=['weight','gain','cover', 'total_gain', 'total_cover']
"""for i in importanceby:
    importance(model,1,i)"""
model_predict = model.predict_proba(test_x)
model_predict = model_predict[:,1]
auc_score = roc_auc_score(test_y, model_predict) 
fpr, tpr, _ = roc_curve(test_y, model_predict) #roc_curve(true binary labels, prediction scores)
print('Test : ', auc_score)    
 
model_predict = model.predict_proba(train_x)
model_predict = model_predict[:,1]
auc_score1 = roc_auc_score(train_y, model_predict)
fpr, tpr, _ = roc_curve(train_y, model_predict)
print('Train : ', auc_score1)
#rocparams.append([i*0.5,500*j,auc_score,auc_score1,np.round(accuracy_score(test_y,predict_signal)*100, 2)])
fields = ['L.rate', 'reg_lambda', 'auc test', 'auc train','accuracy']  

# data rows of csv file  
  

"""with open('signal_for_LSB_10_FEATURES.csv', 'w') as f: 

# using csv.writer method from CSV package 
    write = csv.writer(f) 

    write.writerow(fields) 
    write.writerows(rocparams)    """
signaloverfit(predict_signal_test,predict_signal_train)
overfitting(predict_signal_test,predict_signal_train,predict_back_test,predict_back_train)
roc(test_x,test_y,train_x,train_y)
model.save_model('model_10_leftSB.json')
model.save_model('model_10_leftSB.pkl')
pickle.dump(model, open("model_10_leftSB.dat", "wb"))
"""corr=df.corr()
c=list(b)
c.append('signal/bkgnd')
datar=corr.round(decimals=2)
data=datar.to_numpy()

fig, ax = plt.subplots()
im = ax.imshow(data,cmap='RdBu')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Coeficiente' ,rotation=-90, va="bottom")
ax.set_xticks(np.arange(len(c)))
ax.set_yticks(np.arange(len(c)))
# ... and label them with the respective list entries
ax.set_xticklabels(c)
ax.set_yticklabels(c)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(c)):
    for j in range(len(c)):
        text = ax.text(j, i, datar.to_numpy()[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Matriz de correlación")
fig.tight_layout()
plt.show()"""



        
        
