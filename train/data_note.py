# %%
# !pip install matplotlib
# !pip install seaborn
# !pip install sklearn
# !pip install scipy
# !pip install causalinference
# !pip install causalnex==0.10.0

# %%
# from google.colab import drive
# drive.mount('/content/drive')


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import mlflow.sklearn
import mlflow
from urllib.parse import urlparse
# import warnings library
import warnings
# ignore all warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from preprocess import Preprocess
from ml import Ml

ml = Ml()
preprocess = Preprocess()



# %%
df = pd.read_csv('../Data/data.csv')


# %%
df = pd.DataFrame(df)


# %%
# setting index value with the id column value
# df.set_index('id', inplace=True)


# %%
# droping the unnamed value
df = df.drop("Unnamed: 32", axis=1)


# %%
df.head()


# %%
df.shape


# %%
df.isnull().sum()


# %%
df.columns.values
# 'area_worst', 'smoothness_worst','compactness_worst', 'concavity_worst',


# %%
df.get('diagnosis').value_counts()


# %%
# y includes our labels and x includes our features
y = df.diagnosis                          # M or B
list = ['id', 'diagnosis']
x = df.drop(list, axis=1)
x.head()


# %%
ax = sns.countplot(y, label="Count")       # M = 212, B = 357
B, M = y.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)


# %%
x.describe()


# %% [markdown]
# > > Visualization
# 

# %%
# first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
data = pd.melt(data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features", y="value", hue="diagnosis",
               data=data, split=True, inner="quart")
plt.xticks(rotation=90)


# %% [markdown]
# In texture_mean feature, median of the Malignant and Benign looks like separated so it can be good for classification. However, in fractal_dimension_mean feature, median of the Malignant and Benign does not looks like separated so it does not gives good information for classification
# 

# %%
# Second ten features
data = pd.concat([y, data_n_2.iloc[:, 10:20]], axis=1)
data = pd.melt(data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features", y="value", hue="diagnosis",
               data=data, split=True, inner="quart")
plt.xticks(rotation=90)


# %%
# the last ten features
data = pd.concat([y, data_n_2.iloc[:, 20:31]], axis=1)
data = pd.melt(data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features", y="value", hue="diagnosis",
               data=data, split=True, inner="quart")
plt.xticks(rotation=90)


# %%
# with violin plot
plt.figure(figsize=(10, 10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# %% [markdown]
# Using joint plot we can see how much concavity_worst and concave points_worst are correlated with each other.
# 

# %%
from scipy.stats import stats
sns.jointplot(x.loc[:, 'concavity_worst'],
              x.loc[:, 'concave points_worst'], kind="reg")
r, p = stats.pearsonr(x.loc[:, 'concavity_worst'],
                      x.loc[:, 'concave points_worst'])
print('Personr:', r)
print('p-value:', p)


# %% [markdown]
# 0.86 is looks enough to say that they are correlated.
# 

# %%
sns.set(style="white")
df = x.loc[:, ['radius_worst', 'perimeter_worst', 'area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# %% [markdown]
# And in the above pair grid plot we can see radius_worst, perimeter_worst and area_worst are correlated. We definitely will use these discoveries for feature selection.
# 

# %%
# correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)


# %% [markdown]
# > > Feature Extraction
# > > With correlation and random classfication
# 

# %% [markdown]
# As it can be seen in map heat figure above radius_mean, perimeter_mean and area_mean are correlated with each other so we will use only area_mean. If you ask how i choose area_mean as a feature to use, well actually there is no correct answer, I just look at swarm plots and area_mean looks like clear for me but we cannot make exact separation among other correlated features without trying. So lets find other correlated features and look accuracy with random forest classifier.
# 
# Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefore I only choose concavity_mean. Apart from these, radius_se, perimeter_se and area_se are correlated and I only use area_se. radius_worst, perimeter_worst and area_worst are correlated so I use area_worst. Compactness_worst, concavity_worst and concave points_worst so I use concavity_worst. Compactness_se, concavity_se and concave points_se so I use concavity_se. texture_mean and texture_worst are correlated and I use texture_mean. area_worst and area_mean are correlated, I use area_mean.
# 

# %%
drop_list1 = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se', 'radius_worst',
              'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
x_1 = x.drop(drop_list1, axis=1)        # do not modify x, we will use it later
x_1.head()


# %%
# correlation map
f, ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)


# %% [markdown]
# After drop correlated features, as it can be seen in the above correlation matrix, there are no more correlated features. Actually, I know and you see there is correlation value 0.9 but lets see together what happen if we do not drop it. To see if the right features are picked let see the data we select with random forest and check the result
# 

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(
    x_1, y, test_size=0.3, random_state=42)

# random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(x_train, y_train)

ac = accuracy_score(y_test, clf_rf.predict(x_test))
print('Accuracy is: ', ac)
cm = confusion_matrix(y_test, clf_rf.predict(x_test))
sns.heatmap(cm, annot=True, fmt="d")


# %% [markdown]
# Accuracy is almost 96% which is good but still need some work, so lets' go to the next method to get better result
# 

# %% [markdown]
# > > Univairate Feature Extraction
# > > With correlation and random classfication
# 

# %% [markdown]
# In univariate feature selection, we will use SelectKBest that removes all but the k highest scoring features.
# In this method we need to choose how many features we will use. For example, will k (number of features) be 5 or 10 or 15? The answer is only trying or intuitively. I do not try all combinations but I only choose k = 5 and find best 5 features.
# 

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)


# %%
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)


# %%
x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
# random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(x_train_2, y_train)
ac_2 = accuracy_score(y_test, clf_rf_2.predict(x_test_2))
print('Accuracy is: ', ac_2)
cm_2 = confusion_matrix(y_test, clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2, annot=True, fmt="d")


# %% [markdown]
# Accuracy is almost 96% and as it can be seen in confusion matrix, we make few wrong prediction. What we did up to now is that we choose features according to correlation matrix and according to selectkBest method. Although we use 5 features in selectkBest method accuracies look similar. Now lets see other feature selection methods to find better results.
# 

# %% [markdown]
# > > Recursive feature elimination (RFE) with random forest
# 

# %% [markdown]
# Basically, it uses one of the classification methods (random forest in our example), assign weights to each of features. Whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features
# 
# Like previous method, we will use 5 features. However, the 5 features will be choose by RFE method instead of our judgement choice.
# 

# %%
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)


# %%
print('Chosen best 5 feature by rfe:', x_train.columns[rfe.support_])


# %% [markdown]
# lets see how many feature we need to use with rfecv method.
# 

# %% [markdown]
# > > Recursive feature elimination with cross validation and random forest classification
# 

# %% [markdown]
# Now we will not only find best features but we also find how many features do we need for best accuracy.
# 

# %%
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,
              scoring='accuracy')  # 5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])


# %% [markdown]
# Finally, we find best 12 features that are texture_mean, area_mean, concavity_mean, texture_se, area_se, concavity_se, symmetry_se, smoothness_worst, concavity_worst, symmetry_worst and fractal_dimension_worst for best classification. Lets look at best accuracy with plot.
# 

# %%
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# %% [markdown]
# > > Tree based feature selection and random forest classification¶
# 

# %% [markdown]
# In random forest classification method there is a featureimportances attributes that is the feature importances (the higher, the more important the feature). !!! To use feature_importance method, in training data there should not be correlated features. Random forest choose randomly at each iteration, therefore sequence of feature importance list can change.
# 

# %%
clf_rf_5 = RandomForestClassifier()
clr_rf_5 = clf_rf_5.fit(x_train, y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
        color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# %% [markdown]
# As seen in plot above, after 7 best features importance of features decrease. Therefore we can focus these 5 features. I give importance to understand features and find best of them.
# 

# %%
new_list2 = ['area_mean', 'concavity_mean', 'area_se',
             'concavity_worst', 'symmetry_worst', 'texture_mean', 'concavity_se']
x_2 = x_1[new_list2].copy()
x_2.head()


# %% [markdown]
# > > Feature Extraction with PCA
# 

# %% [markdown]
# principle component analysis (PCA) for feature extraction. Before PCA, we need to normalize data for better performance of PCA
# 

# %%
# split data train 70 % and test 30 %
from sklearn.decomposition import PCA
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)
# normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

pca = PCA()
pca.fit(x_train_N)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')


# %% [markdown]
# According to variance ration, 3 component can be chosen. In conclustion, I tried to show importance of feature selection and data visualization. Default data includes 33 feature but after feature selection we drop this number from 33 to 5 with accuracy 95%. In this kernel we just tried basic things, I am sure with these data visualization and feature selection methods, we can easily ecxeed the % 95 accuracy.
# 

# %% [markdown]
# > > Data scaling - for selected feature
# 

# %%
x_2['diagnosis'] = y    # do not modify x, we will use it later
x_2['diagnosis'] = x_2['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
x_2.head()


# %%
# s1 = [x_2, np.nan]
# pd.get_dummies(s1)


# %%
import os,sys 
sys.path.append(os.path.abspath(os.path.join('../script/')))
from func import dataHandler
# cas = dataHandler()

# %%
scale_train_list = x_2.columns.to_list()
dataHandler.standardize_columns(x_2, scale_train_list)


# %% [markdown]
# > > Causal Inference
# 

# %% [markdown]
# These features describe characteristics of the cell nuclei present in the image and can be used to build a model to predict whether a tumor is benign or malignant.
# 

# %%
# !pip install causalnex
# !apt install libgraphviz-dev
# !pip install pygraphviz


# %%
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE


def vis_sm(sm):
    viz = plot_structure(sm, graph_attributes={"scale": "2.0",
                                               'size': 2.5}, all_node_attributes=NODE_STYLE.WEAK,
                         all_edge_attributes=EDGE_STYLE.WEAK)
    return Image(viz.draw(format='png'))


sm = from_pandas(x_2.iloc[:, :8], tabu_parent_nodes=['diagnosis'],)
vis_sm(sm)


# %%
sm.remove_edges_below_threshold(0.8)
vis_sm(sm)


# %% [markdown]
# **Comparison** with Jaccard Similarity Index
# 

# %%
x_selected = x_2.copy()
# x_selected['diagnosis'] = y    # do not modify x, we will use it later
# x_selected['diagnosis'] = x_selected['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
# # x_selected.head()
# scale_train_list = x_selected.columns.to_list()
# standardize_columns(x_selected, scale_train_list)


# %%
def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return "{0:.0%}".format(round(len(i) / (len(g) + len(h) - len(i)), 3))


# %% [markdown]
# **Causal graph with 60% data**
# 

# %%
portion = int(x_selected.shape[0]*.6)
x_portion = x_selected.head(portion)
sm2 = from_pandas(x_portion, tabu_parent_nodes=['diagnosis'],)
sm2.remove_edges_below_threshold(0.8)
sm2 = sm2.get_largest_subgraph()
vis_sm(sm2)


# %% [markdown]
# **Causal graph with 70% data**
# 

# %%
portion = int(x_selected.shape[0]*.7)
x_portion = x_selected.head(portion)
sm3 = from_pandas(x_portion, tabu_parent_nodes=['diagnosis'],)
sm3.remove_edges_below_threshold(0.8)
sm3 = sm3.get_largest_subgraph()
vis_sm(sm3)


# %%
jaccard_similarity(sm2.edges, sm3.edges)


# %% [markdown]
# The Jaccard similarity b/n the casual graph of 60% and 70% of the data is 89%.
# 

# %% [markdown]
# **Causal graph with 80% data**
# 

# %%
portion = int(x_selected.shape[0]*.8)
x_portion = x_selected.head(portion)
sm4 = from_pandas(x_portion, tabu_parent_nodes=['diagnosis'],)
sm4.remove_edges_below_threshold(0.8)
sm4 = sm4.get_largest_subgraph()
vis_sm(sm4)


# %%
jaccard_similarity(sm3.edges, sm4.edges)


# %% [markdown]
# The Jaccard similarity b/n the casual graph of 70% and 80% of the data is 90%.
# 

# %% [markdown]
# **Causal graph with 90% data**
# 

# %%
portion = int(x_selected.shape[0]*.9)
x_portion = x_selected.head(portion)
sm5 = from_pandas(x_portion, tabu_parent_nodes=['diagnosis'],)
sm5.remove_edges_below_threshold(0.8)
sm5 = sm5.get_largest_subgraph()
vis_sm(sm4)


# %%
jaccard_similarity(sm4.edges, sm5.edges)


# %% [markdown]
# The Jaccard similarity b/n the casual graph of 80% and 90% of the data is 90%.
# 

# %% [markdown]
# **Causal graph with 100% data**
# 

# %%
portion = int(x_selected.shape[0]*1.0)
x_portion = x_selected.head(portion)
sm6 = from_pandas(x_portion, tabu_parent_nodes=['diagnosis'],)
sm6.remove_edges_below_threshold(0.8)
sm6 = sm6.get_largest_subgraph()
vis_sm(sm6)


# %%
jaccard_similarity(sm5.edges, sm6.edges)


# %% [markdown]
# The Jaccard similarity b/n the casual graph of 90% and 100% of the data is 100%.
# 

# %% [markdown]
# we can conclude that everytime we increase the fraction of the data and the compare the jaccard similarity it is more than 80% in all comparision so we can conclude this causal graph is stable
# 

# %% [markdown]
# **Markov blanket**
# To select only variables that point directly to the target variable, in these case the target variable is the diagnosis as used above
# 

# %%
from causalnex.network import BayesianNetwork
from causalnex.utils.network_utils import get_markov_blanket
bn = BayesianNetwork(sm)
blanket = get_markov_blanket(bn, 'diagnosis')
# list(blanket.structure.edges)
print(blanket.edges)


# %% [markdown]
# so the nearest variables to the target variable from the feature we extracted and used so far are concavity_mean,convaity_worst and concavity_se

# %%
from causalnex.discretiser.discretiser_strategy import (
    DecisionTreeSupervisedDiscretiserMethod)
features = ['area_mean', 'concavity_mean', 'area_se', 'concavity_worst',
            'symmetry_worst', 'texture_mean', 'concavity_se', 'diagnosis']

def discretised(features,data):
    tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(
        mode='single',
        tree_params={'max_depth': 3, 'random_state': 27},
    )
    tree_discretiser.fit(
        feat_names=features,
        dataframe=data,
        target_continuous=True,
        target='diagnosis',
    )
    discretised_data = data.copy()
    for col in features:
        discretised_data[col] = tree_discretiser.transform(data[[col]])
    return discretised_data
discretised_data = discretised(features,x_selected)


# %%
from sklearn.metrics import recall_score, precision_score
def bayesian_model(discretised_data, sm):
    train, test = train_test_split(
        discretised_data, train_size=0.8, test_size=0.2, random_state=27)
    bn = BayesianNetwork(sm)
    bn = bn.fit_node_states(discretised_data)
    bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

    pred = bn.predict(test, 'diagnosis')
    true = np.where(test['diagnosis'] == 0, 1, 0)
    pred = np.where(pred == 0, 1, 0)

    print('Recall: {:.2f}'.format(recall_score(y_true=true, y_pred=pred)))
    print('F1: {:.2f} '.format(f1_score(y_true=true, y_pred=pred)))
    print('Accuracy: {:.2f} '.format(accuracy_score(y_true=true, y_pred=pred)))
    print('Precision: {:.2f} '.format(precision_score(y_true=true, y_pred=pred)))

bayesian_model(discretised_data, sm)


# %%
graph_list = ['concavity_mean','concavity_worst','concavity_se','diagnosis']
x_3 = x_selected[graph_list].copy()
x_3.head()

# %%
feat = ['concavity_mean','concavity_worst','concavity_se','diagnosis']
discretised_graph = discretised(feat, x_3)
discretised_graph 

# %%
# gm = from_pandas(x_3.iloc[:, :8], tabu_parent_nodes=['diagnosis'],)
# bayesian_model(discretised_graph, sm)

# %%

from sklearn.linear_model import LogisticRegression

train, test = train_test_split(x_3, train_size=0.8, test_size=0.2, random_state=7)

clf = LogisticRegression()
clf.fit(train.drop(columns=['diagnosis']), train['diagnosis'])

pred = clf.predict(test.drop(columns=['diagnosis']))
true = test['diagnosis']

print('Recall: {:.2f}'.format(recall_score(y_true=true, y_pred=pred)))
print('F1: {:.2f} '.format(f1_score(y_true=true, y_pred=pred)))
print('Accuracy: {:.2f} '.format(accuracy_score(y_true=true, y_pred=pred)))
print('Precision: {:.2f} '.format(precision_score(y_true=true, y_pred=pred)))

# %% [markdown]
# Accuracy of 87% come using the variables selected by the causual graph




mlflow.set_experiment('data analysis')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mlflow.log_param('data_url', x_2)
    mlflow.log_param('input_rows', x_2.shape[0])
    mlflow.log_param('input_cols', x_2.shape[1])
    mlflow.log_param('model_type','Logistic Regression')
    mlflow.log_param('model_parameters', 'n_estimators=100, max_depth=10')

    # use logistic regression
    logistic_regression_model = LogisticRegression(random_state=0)

    logistic_regression_result = ml.cross_validation(logistic_regression_model, X, y, 5)

    # Write scores to file
    with open("train/logistic_metrics.txt", 'w') as outfile:
        outfile.write(
            f"Training data accuracy: {logistic_regression_result['Training Recall scores'][0]}")
        outfile.write(
            f"Validation data accuracy: {logistic_regression_result['Validation f1 scores'][0]}")
        outfile.write(
            f"Validation data accuracy: {logistic_regression_result['Validation Accuracy scores'][0]}")
        outfile.write(
            f"Validation data accuracy: {logistic_regression_result['Validation precision scores'][0]}")


    # Plot accuacy results to cml

    # Plot Accuracy Result
    model_name = "Logistic Regression"
    ml.plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
                logistic_regression_result["Training Accuracy scores"],
                logistic_regression_result["Validation Accuracy scores"],
                'train/logistic_accuracy.png')

    # Precision Results

    # Plot Precision Result
    ml.plot_result(model_name, "Precision", "Precision scores in 5 Folds",
                logistic_regression_result["Training Precision scores"],
                logistic_regression_result["Validation Precision scores"],
                'train/logistic_preicision.png')

    # Recall Results plot

    # Plot Recall Result
    ml.plot_result(model_name, "Recall", "Recall scores in 5 Folds",
                logistic_regression_result["Training Recall scores"],
                logistic_regression_result["Validation Recall scores"],
                'train/logistic_recall.png')


    # f1 Score Results

    # Plot F1-Score Result
    ml.plot_result(model_name, "F1", "F1 Scores in 5 Folds",
                logistic_regression_result["Training F1 scores"],
                logistic_regression_result["Validation F1 scores"],
                'train/logistic_f1_score.png')
