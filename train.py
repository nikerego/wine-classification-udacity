from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from sklearn.utils import resample


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to use in the optimization problem.")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Solver:", args.solver)

    # Data from UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    # Data from UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    # Read data
    red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

    # Set target classes (0: White Wine, 1: Red Wine)
    white['y'] = 0
    red['y'] = 1

    # Combine into a single DataFrame & shuffle
    wine_df = pd.concat([red, white], axis=0)
    wine_df = wine_df.sample(frac=1)

    # Upsample to balance dataset (Number of Instances: red wine - 1599; white wine - 4898)
    def upsample_classes(data, target):
        
        lst = list(data[target].unique())
        
        classes = []
        for c in lst:
            classes.append(data[data[target]==c])
        
        length = 0
        class_lab = None
        for c in classes:
            if len(c)>length:
                length=len(c)
                class_lab = c
        class_lab = class_lab[target].unique()[0]
        
        regroup = pd.concat(classes)
        maj_class = regroup[regroup[target]==class_lab]

        lst.remove(class_lab)
        
        new_classes=[]
        for i in lst:
            new_classes.append(resample(data[data[target]==i],replace=True, n_samples=len(maj_class)))

        minority_classes = pd.concat(new_classes)
        upsample = pd.concat([regroup[regroup[target]==class_lab],minority_classes])

        return upsample

    wine_df_balanced = upsample_classes(wine_df, 'y')

    y = wine_df_balanced.pop('y')
    x = wine_df_balanced

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=666)

    # Train Logistic Regression Model
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
