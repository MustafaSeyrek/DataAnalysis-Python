import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class Analiz:
    def __init__(self, file):
        # Veri setinin yüklenmesi
        veriSeti = pd.read_csv(file)
        # Bağımlı ve bağımsız değişkenlerin oluşturulması
        X = veriSeti.values[:, 0:20]
        Y = veriSeti.values[:, 20]
        #Veri kümesinin eğitim ve test verileri olarak ayrılması
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,train_size=0.66,
                                            test_size=0.34, random_state=7)
        print("\033[94m Verilerin tür değişkenine göre dağılımı")
        print(veriSeti.groupby('Sonuc').size())
        # histogram graph
        veriSeti.hist()
        plt.show()
        # Modellerin listesinin olusturulmasi
        models = [
            ('Logistic Regression          ', LogisticRegression(max_iter=8000)),
            ('K-Nearest NNeighbors         ', KNeighborsClassifier()),
            ('Decision Tree                ', DecisionTreeClassifier()),
            ('Gaussian Naive Bayes         ', GaussianNB()),
            ('Support Vector Machine       ', SVC()),
            ('Multi-Layer Perceptron       ', MLPClassifier(max_iter=8000)),
            ('Random Forest                ', RandomForestClassifier(n_estimators=100))
        ]
        results = []
        names = []
        sonuc=0
        isim=""
        # Modeller için 'cross validation' sonuçlarının  yazdırılması
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
            results.append(cv_results)
            names.append(name)
            if sonuc < cv_results.mean():
                sonuc = cv_results.mean()
                isim = name
            strMeanValue = "{:.2%}".format(cv_results.mean()) # yüzde formatına dönüştürme
            print("\033[94m %s: \033[0m %s " % (name, strMeanValue))
        isim=isim.strip(" ") #düzenli yazdırmak için algoritma ismindeki boşluk silindi
        strSonuc = "{:.2%}".format(sonuc)# yüzde formatına dönüştürme
        print("\033[94m \n Seçilen Algoritma: \033[0m%s \033[94m Sonuç Değeri: \033[0m %s"%(isim,strSonuc))
a = Analiz("almanya analiz.csv")