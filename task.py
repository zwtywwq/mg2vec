import argparse
import numpy as np
import pandas as pd
import sys

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# n_w_t = np.loadtxt('./model/facebook/final_n_w_t.txt')
# n_b = np.loadtxt('./model/facebook/final_n_b.txt')


def relu(matrix):
    return np.maximum(0, matrix)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1608)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--train_size', type=float, default=0.2)
    parser.add_argument('--data_file', required=True)
    # parser.add_argument('--output_file', default='../data/dblp/csv/metrics_mg-ord2.xlsx')

    return parser.parse_args()


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop(['Id'], axis=1)
    return df


def balanced_sample(df, train_size_per_class, seed):
    df = df.sample(frac=1, random_state=seed)
    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)
    for label in df['Label'].unique():
        df_label = df.loc[df['Label'] == label]
        df_train = df_train.append(df_label.iloc[:train_size_per_class, :], ignore_index=True)
        df_test = df_test.append(df_label.iloc[train_size_per_class:, :], ignore_index=True)

    return df_train, df_test


def split_data(df, train_size, seed):
    if train_size <= 0:
        raise argparse.ArgumentTypeError("train_size not in range (0.0, infitity)")

    if train_size < 1.0:
        df_train, df_test = model_selection.train_test_split(df, train_size=train_size, test_size=None,
                                                             random_state=seed, stratify=df['Label'].values)
    else:
        df_train, df_test = balanced_sample(df, int(train_size), seed + 5)

    ncol = len(df.columns)
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    # X_train = relu(np.dot(X_train, n_w_t))
    # X_test = relu(np.dot(X_test, n_w_t))

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


#############################

args = parse_args()
df = load_data(args.data_file)

classes, counts = np.unique(df['Label'].values, return_counts=True)
print(classes)
print(counts)
df_metrics = pd.DataFrame(columns=['MicroF', 'MacroF', 'Accuracy'])
print('instances =', len(df), ', features =', len(df.columns) - 1, ', classes =', len(classes))

for split in range(args.num_splits):
    print('*', end='')
    sys.stdout.flush()

    X_train, y_train, X_test, y_test = split_data(df, args.train_size, args.seed + split * 7)

    clf = SVC(gamma='auto').fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred_test)
    f_micro = metrics.f1_score(y_test, pred_test, labels=np.delete(classes, np.argmax(counts)), average='micro')
    f_macro = metrics.f1_score(y_test, pred_test, average='macro')
    CM = confusion_matrix(y_test, pred_test, labels=classes)
    # print(CM)
    df_metrics.loc[split] = [f_micro, f_macro, accuracy]
print(df_metrics.iloc[:args.num_splits, :])
df_metrics.loc["Sd"] = df_metrics.iloc[:args.num_splits, :].std()
df_metrics.loc["Mean"] = df_metrics.iloc[:args.num_splits, :].mean()
print()
print(df_metrics.iloc[-1:, :])
# df_metrics.to_excel(args.output_file)
