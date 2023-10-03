import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Importar o conjunto de dados
df = pd.read_csv("./raw.githubusercontent.com_danielvieira95_IABD_master_Bases_de_dados_Atividade_Formativa_dados_produtos.csv")

# Visualizar as primeiras linhas do DataFrame
print(df.head())

# Ver informações sobre o conjunto de dados
df.info()

# Ver estatísticas descritivas das colunas numéricas
print(df.describe())

# Plotar um box plot das notas para verificar outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='product_name', y='rating', data=df)
plt.title('Box Plot das Notas de Avaliação por Produto')
plt.xlabel('Produto')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()

# Verificar valores ausentes
print(df.isnull().sum())

# Separar o conjunto de dados em features (X) e target (y)
X = df.drop('purchased', axis=1)
y = df['purchased']
# Codificar variáveis categóricas usando one-hot encoding
X = pd.get_dummies(X, columns=['product_name'], drop_first=False)
# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Criar um dicionário para armazenar os modelos
models = {}

# Lista de produtos únicos
unique_products = df['product_name'].unique()
print(unique_products)
# Iterar sobre os produtos e treinar um modelo para cada um
for product in unique_products:
    # Filtrar o conjunto de treinamento para o produto atual
    X_train_product = X_train[X_train['product_name_' + product] == 1]
    y_train_product = y_train[X_train['product_name_' + product] == 1]
    
    # Criar e treinar o modelo de Árvore de Decisão
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_product[['rating', 'rating_count']], y_train_product)
    
    # Armazenar o modelo no dicionário
    models[product] = model

# Dicionário para armazenar as métricas
metrics = {}

# Iterar sobre os produtos e avaliar os modelos
for product in unique_products:
    # Filtrar o conjunto de teste para o produto atual
    X_test_product = X_test[X_test['product_name_' + product] == 1]
    y_test_product = y_test[X_test['product_name_' + product] == 1]
    
    # Previsões do modelo
    y_pred = models[product].predict(X_test_product[['rating', 'rating_count']])
    
    # Calcular métricas
    accuracy = accuracy_score(y_test_product, y_pred)
    precision = precision_score(y_test_product, y_pred)
    recall = recall_score(y_test_product, y_pred)
    f1 = f1_score(y_test_product, y_pred)
    confusion = confusion_matrix(y_test_product, y_pred)
    
    # Armazenar métricas no dicionário
    metrics[product] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Confusion Matrix': confusion}

# Exibir métricas para cada produto
for product, metric in metrics.items():
    print(f"Produto: {product}")
    for metric_name, value in metric.items():
        print(f"{metric_name}: {value}")
    print("\n")

# Visualizar a estrutura da árvore de decisão (usando a estrutura de regras)

def plot_tree_structure(model, feature_names, class_names, product):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model,feature_names=feature_names, class_names=class_names, filled=True, ax=ax)
    plt.title(f"Estrutura da Árvore de Decisão {product}")
    plt.show()


# Plotar as estruturas da árvore de decisão
for key in models:
    plot_tree_structure(models[key], ['rating', 'rating_count'], ['Não Compra', 'Compra'], key)