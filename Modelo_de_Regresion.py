import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')

#print(data.shape)
#print(data.head())
#print(data.columns.tolist())
#print(data.describe())

#Descargar para ver con otro programa que visualiza mejor los datos
#data.to_csv("CarPrice_Assignment.csv", index=False)

#Una vez analizado lo que se hara sera generar un modelo que pronostique el precio de los autos segun sus caracterisitcas
#No se hara EDA el ejercicio es solo hacer el modelo de regresion

#Algunas desiciones, Eliminar car_ID, symboling, CarName separaremos la marca de los nombres de los autos y quedarnos solo
#Con la marca puede inlfuir en el precio

#El labels sera "price"

data2 = data.drop(["car_ID", "symboling"], axis=1)



#Separamos la columna car_ID

data2[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data2["CarName"].str.split(" ",expand=True)

data3 = data2.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

print(data3.info())


data_gr=pd.DataFrame(data3.groupby("Brand")["price"].agg("mean"))

print(data_gr)

#Es una grafica que muestra los valores promedio de las marcas y tienen arto que ver en el precio final, puede que ocupen
#tambien caracterisitcas similares, se necesita un examinar aun mas las caracteristicas, si es que tienen caracterisiticas
#similares cada marcas esto podria eliminar a las marcas del modelo en general no puede hacer regresiones adecuacadas con
#caracterisitcas categoricas, aun asi no son tan representaticas en el calculo de la prediccion
data_gr.plot(kind="bar")
plt.show()

#Ahora pasar las variables categoricas que estan en tipo "object", pasarlas a "categories" para que el modelo las entienda
#como tal

#Las columnas que son categoricas son "Brand", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel",
#"enginelocation", "enginetype", "cylindernumber", "fuelsystem"


#print(data3.info())

#Ahora codigicar todas estas columnas a numeros para la regresion

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ohe = OneHotEncoder()
le = LabelEncoder()

#Muestra la cantidad de datos unicos y categorias tiene cada columna, debe ser mas de una para que tenga sentido
#num_uni_cols = (data3[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]].apply(lambda x: x.nunique()).sort_values(ascending=False))
#print(num_uni_cols)   

#print(data3.columns.tolist())

data_name_col = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]

"""
#Agregando las columnas codificadas al dataframe con LabelsEncoder
for col in data_name_col:
    
    data_le = le.fit_transform(data3[col])
    data_le = pd.DataFrame(data_le)
    data3[col] = data_le
               
#for col in data_name_col:
"""

#Para ocupar OneHotEncoder y ocupar ".categories_" se necesita que sean columnas tipo "category"
data3[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]]= data3[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]].astype("category")

#Codificando todas las columnas categoricas con OneHotEncoder
for col in data_name_col:

       data_ohe = ohe.fit_transform(data3[[col]].values.reshape(-1, 1)).toarray()
       data3 = pd.concat([data3.drop(col, axis = 1), pd.DataFrame(data_ohe, columns = ohe.categories_[0])], axis = 1)
       
#print(data3.columns.tolist())

#Ahora esta lista para el modelo de prediccion

X = data3.drop("price", axis=1)
y = data3["price"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Separar los datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])

#A probar los diferentes modelos para saber cual tiene mejor puntuacion R2


pipe_lm = Pipeline([("ss", StandardScaler()), ("model_lm", LinearRegression())]) 

pipe_lm.fit(X_train, y_train)

y_pred = pipe_lm.predict(X_test)

#El modelo se sobreajusto
print("R^2 on training  data lm ", pipe_lm.score(X_train, y_train))
print("R^2 on testing data lm", pipe_lm.score(X_test,y_test))
#Puntuacion no funciono
print("R^2 on predict data lm", r2_score(y_pred,y_test))
print(" ")


pipe_lr = Pipeline([("ss", StandardScaler()), ("model_lr", Ridge())]) 

pipe_lr.fit(X_train, y_train)

y_pred2 = pipe_lr.predict(X_test)

#Funciono bastante buena puntuacion
print("R^2 on training  data lr ", pipe_lr.score(X_train, y_train))
print("R^2 on testing data lr", pipe_lr.score(X_test,y_test))
#0.8937 buena
print("R^2 on predict data lr", r2_score(y_pred2,y_test))
print("MSE", mean_squared_error(y_pred2,y_test))
print(" ")

pipe_ll = Pipeline([("ss", StandardScaler()), ("model_ll", Lasso())]) 

pipe_ll.fit(X_train, y_train)

y_pred3 = pipe_ll.predict(X_test)

#Necesita mas iteraciones, pero obtuvo buen resultado
print("R^2 on training  data lm ", pipe_ll.score(X_train, y_train))
print("R^2 on testing data lm", pipe_ll.score(X_test,y_test))
#0.9088
print("R^2 on predict data lm", r2_score(y_pred3,y_test))
print("MSE", mean_squared_error(y_pred3,y_test))
print(" ")

pipe_en = Pipeline([("ss", StandardScaler()), ("model_ll", ElasticNet())]) 

pipe_en.fit(X_train, y_train)

y_pred4 = pipe_en.predict(X_test)

#Obtuvo la puntuacion mas pareja y la mejor en prueba
print("R^2 on training  data EN ", pipe_en.score(X_train, y_train))
print("R^2 on testing data EN", pipe_en.score(X_test,y_test))
#0.9079
print("R^2 on predict data EN", r2_score(y_pred4,y_test))
print("MSE", mean_squared_error(y_pred4,y_test))
print(" ")

###Intentando maximizar los resultados, polynomialfeature y viendo si la y_train sigue una distribucion normal
#Ademas de utilizar GridSearch

print(" ")

from scipy.stats import boxcox
from scipy.stats.mstats import normaltest

#data_plot = data3['price']
#plt.hist(data_plot)
#plt.show()

#print(normaltest(data_plot.values))

#data_plot_sq = np.sqrt(data3['price'])
#plt.hist(data_plot_sq)
#plt.show()

#print(normaltest(data_plot_sq.values))

#boxcox tambien se parece a una distribucion normal, pero menos que boxcox
#data_plot_log = np.log(data3['price'])
#plt.hist(data_plot_log)
#plt.show()

#print(normaltest(data_plot_log.values))

#boxcox es la que mas se parece a una distribucion normal
#data_plot_box = boxcox(data3['price'])[0]
#plt.hist(data_plot_box)
#plt.show()

#print(normaltest(pd.DataFrame(data_plot_box).values))

#Ninguna transformacion sigue una distribucion normal, no superan el p_value=0.05, no mejorara el rendimiento

 


###Ahora intentando mejorar el rendimiento con buscando hyperparametros

#pipe_lm = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("ss", StandardScaler()), ("model_lm", LinearRegression())]) 

#pipe_lm.fit(X_train, y_train)

#y_pred = pipe_lm.predict(X_test)

#param_grid = {
#    "polynomial__degree": [ 1, 2,3,4],
#    #"model__alpha":[0.00001, 0.0001,0.001,0.01,0.1,1,10,100]
#}

#search = GridSearchCV(pipe_lm, param_grid, n_jobs=2)
#search.fit(X_train, y_train)
#best=search.best_estimator_
#print(best.score(X_test,y_test))  

#print("best_score_: ",search.best_score_)
#print("best_params_: ",search.best_params_) 


#El modelo se sobreajusto
#print("R^2 on training  data lm ", pipe_lm.score(X_train, y_train))
#print("R^2 on testing data lm", pipe_lm.score(X_test,y_test))



pipe_lr = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)),("ss", StandardScaler()), ("model", Ridge(alpha=100))]) 

param_grid = {
    "polynomial__degree": [ 1, 2,3],
    "model__alpha":[0.01, 0.1, 1, 10,100,1000]
}


pipe_lr.fit(X_train, y_train)

y_pred2 = pipe_lr.predict(X_test)

#print(y_pred2)


#search = GridSearchCV(pipe_lr, param_grid, n_jobs=2)
#search.fit(X_train, y_train)
#best=search.best_estimator_
#print(best.score(X_test,y_test))  

###Encontreo el mejor parametro grado polinomial=2, alfa=100

#print("best_score_: ",search.best_score_)
#print("best_params_: ",search.best_params_) 

#Funciono bastante buena puntuacion el modelo con la prueba
print("R^2 on training  data lr ", pipe_lr.score(X_train, y_train))
print("R^2 on testing data lr", pipe_lr.score(X_test,y_test))
#La puntuacion de la prediccion 0.9094 mejoro un poco la puntuacion
print("R^2 on prediccion data lr", r2_score(y_pred2,y_test))
print("MSE", mean_squared_error(y_pred2,y_test))
print(" ")

#Poly = PolynomialFeatures(include_bias=False, degree=2)
#scaled = StandardScaler()
#X_train=scaled.fit_transform(X_train)
#X_train=Poly.fit_transform(X_train)

#Para graficar cual podria ser el mejor alfa
#alphas = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
#R_2=[]
#coefs = []
#for alpha in alphas:
#    ridge = Ridge(alpha=alpha)
#    ridge.fit(X_train, y_train)
#    coefs.append(abs(ridge.coef_))
#    R_2.append(ridge.score(X_train,y_train))

#ax = plt.gca()
#ax.plot(alphas, R_2)
#ax.set_xscale("log")
#plt.xlabel("alpha")
#plt.ylabel("$R^2$")
#plt.title("$R^2$ as a function of the regularization")
#plt.show() 


pipe_ls = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("ss", StandardScaler()), ("model", Lasso(alpha=100, max_iter=5000))]) 

pipe_ls.fit(X_train, y_train)

y_pred3 = pipe_ls.predict(X_test)

#search = GridSearchCV(pipe_ll, param_grid, n_jobs=2)
#search.fit(X_train, y_train)
#best=search.best_estimator_
#print(best.score(X_test,y_test))  
#Encontro mejor hiperparametros, grado polinomio=2, alfa=100
#print("best_score_: ",search.best_score_)
#print("best_params_: ",search.best_params_) 



#Necesita mas iteraciones, pero obtuvo buen resultado
print("R^2 on training  data ls ", pipe_ls.score(X_train, y_train))
print("R^2 on testing data ls", pipe_ls.score(X_test,y_test))
#0.9105
print("R^2 on testing data ls", r2_score(y_pred3,y_test))
print("MSE", mean_squared_error(y_pred3,y_test))
print(" ")

pipe_en = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("ss", StandardScaler()), ("model", ElasticNet(tol=0.2, max_iter=5000, l1_ratio=0.75, alpha=10))]) 

param_grid = {
    #"polynomial__degree": [ 1, 2,3],
    "alpha":[0.001, 0.1,1,10,100],
    "l1_ratio":[0.5,0.75, 1]
}

pipe_en.fit(X_train, y_train)

y_pred4 = pipe_en.predict(X_test)



#search = GridSearchCV(pipe_en, param_grid, n_jobs=2)
#search.fit(X_train, y_train)
#best=search.best_estimator_
#print(best.score(X_test,y_test))  
#Encontro como mejor parametros alfa=10, l1_ratio=0.75, grado polinomial=2
#print("best_score_: ",search.best_score_)
#print("best_params_: ",search.best_params_) 

#Obtuvo la puntuacion mas pareja
print("R^2 on training  data EN ", pipe_en.score(X_train, y_train))
print("R^2 on testing data EN", pipe_en.score(X_test,y_test))
#La mejor prediccion con 0.9281
print("R^2 on predict data EN", r2_score(y_pred4,y_test))
print("MSE", mean_squared_error(y_pred4,y_test))

def plot_coef(X,model,name=None):
    
    plt.bar(X.columns[2:],abs(model.coef_[2:]))
    plt.xticks(rotation=90)
    plt.ylabel("$coefficients$")
    plt.title(name)
    plt.show()
    print("R^2 on training  data ",model.score(X_train, y_train))
    print("R^2 on testing data ",model.score(X_test,y_test))

#Esto es para poder ver las caracterisitcas con efecto polinomicos no se puede ver
en= ElasticNet(tol=0.2, max_iter=5000, l1_ratio=0.75, alpha=10)   
en.fit(X_train, y_train)

y_pred4 = en.predict(X_test)   
        
#Muestra el valor de los coeficientes de cada columna
plt.bar(X.columns[2:], abs(en.coef_[2:]))
plt.xticks(rotation=90)
plt.ylabel("$coefficients$")
plt.title("name")
plt.show()

def get_R2_features(model,test=True): 
    #X: global  
    features=list(X)
    features.remove("three")
    
    R_2_train=[]
    R_2_test=[]

    for feature in features:
        model.fit(X_train[[feature]],y_train)
        
        R_2_test.append(model.score(X_test[[feature]],y_pred4))
        R_2_train.append(model.score(X_train[[feature]],y_train))
        
    plt.bar(features,R_2_train,label="Train")
    plt.bar(features,R_2_test,label="predict")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)),str(np.mean(R_2_test))) )
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)),str(np.max(R_2_test))) )

#Me llama la atencion las marcas no aportan a la prediccion, pero me gusta que se consideren en una gran cantidad de datos
#Pueden ser importantes
get_R2_features(en)


search = GridSearchCV(en, param_grid, n_jobs=2)
columns1=X.columns.tolist()[:12]

#Muestra como se estan comportando las regresiones en cada columna, en general es buena
for column in columns1:
    search.fit(X_train[[column]], y_train)
    x=np.linspace(X_test[[column]].min(), X_test[[column]].max(),num=100)
    plt.plot(x,search.predict(x.reshape(-1,1)),label="prediction")
    plt.plot(X_test[column],y_pred4,'ro',label="y")
    plt.xlabel(column)
    plt.ylabel("y")
    plt.legend()
    plt.show()










