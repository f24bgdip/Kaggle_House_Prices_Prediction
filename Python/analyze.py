import numpy as np
import scipy.stats as stats
from scipy.stats import kendalltau, pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import power_transform
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy


def check_data(df):
    print(df.shape)
    print(df.info())
    print(df.head())
    print(df.describe())


def check_missing_data(df, plotting=False):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent],
                             axis=1,
                             keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data['Total'] > 0]

    if plotting:
        if not missing_data.empty:
            missing_data['Total'].plot.bar()
            print(missing_data)

    return missing_data


def check_missing_data(df, plotting=False):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent],
                             axis=1,
                             keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data['Total'] > 0]

    if plotting:
        if not missing_data.empty:
            missing_data['Total'].plot.bar()
            print(missing_data)

    return missing_data


def divide_numerical_categorical(df, *exclude_features):
    a = df.loc[:, ~df.columns.isin(exclude_features)]

    numerical = [f for f in a.columns if a.dtypes[f] != 'object']
    print(numerical)
    print(len(numerical))

    categorical = [f for f in a.columns if a.dtypes[f] == 'object']
    print(categorical)
    print(len(categorical))

    return numerical, categorical


def classify_samples_by_dtypes(df, exclusion_columns=[]):
    return {
        'int': df.select_dtypes(include=['int64']).columns.drop(exclusion_columns, errors='ignore').values,
        'float': df.select_dtypes(include=['float64']).columns.drop(exclusion_columns, errors='ignore').values,
        'numeric': df.select_dtypes(include=['int64', 'float64']).columns.drop(exclusion_columns, errors='ignore').values,
        'categorical': df.select_dtypes(include=['object']).columns.drop(exclusion_columns, errors='ignore').values,
        'datetime': df.select_dtypes(include=['datetime64']).columns.drop(exclusion_columns, errors='ignore').values,
    }


# Transform
def gaus_transform(df, features):
    names = list(map(lambda str: str+'_gaus', features))

    for i, c in enumerate(features):
        df[names[i]] = scale(df[c].values)

    return names


def gaus_inverse(df, feature, mean, std):
    df[feature] = df[feature]*std + mean

    return df


def log_transform(df, features):
    features_log = []

    for c in features:
        name = c+'_log'
        if df[c].min() > 1:
            df[name] = np.log(df[c].values)
        else:
            df[name] = np.log1p(df[c].values)

        features_log.append(name)

    return features_log


def log_inverse(df, feature):
    df[feature] = np.expm1(df[feature])

    return df


def quadratic(df, features):
    df[features+'_quad'] = df[features]**2


def johnson_for_features(df, features):
    features_john = []

    for c in features:
        name = c+'_john'
        gamma, eta, epsilon, lbda = stats.johnsonsu.fit(df[c].values)
        yt = gamma + eta*np.arcsinh((df[c].values-epsilon)/lbda)
        df[name] = yt
        features_john.append(name)

    return features_john, yt, gamma, eta, epsilon, lbda


def johnson(y):
    gamma, eta, epsilon, lbda = stats.johnsonsu.fit(y.values)
    yt = gamma + eta*np.arcsinh((y.values-epsilon)/lbda)

    return yt, gamma, eta, epsilon, lbda


def johnson_inverse(y, gamma, eta, epsilon, lbda):

    return lbda*np.sinh((y-gamma)/eta) + epsilon


def power_transform_box_cox(df, features):
    features_box_cox = []

    for c in features:
        name = c+'_bocox'
        df[name] = power_transform(df[c], method='box-cox')
        features_box_cox.append(name)

    return features_box_cox


def power_transform_yeo_johnson(df, features):
    features_yeo_johnson = []

    for c in features:
        name = c+'_yeoj'
        df[name] = power_transform(df[c], method='yeo-johnson')
        features_box_cox.append(name)

    return features_yeo_johnson


def qt_transform(df, features):
    features_qt = []
    qt = QuantileTransformer
    for c in features:
        name = c+'_qt'
        df[name] = power_transform(df[c], method='yeo-johnson')
        features_box_cox.append(name)

    return features_yeo_johnson


# Test
def transform_to_various_distribution(X):
    distributions = [
        #('UnscaledData',
        #    lambda x: x),
        ('LogTransform',
            np.log1p),
        ('StandardScaling',
            StandardScaler().fit_transform),
        ('Min-maxScaling',
            MinMaxScaler().fit_transform),
        ('Max-absScaling',
            MaxAbsScaler().fit_transform),
        ('RobustScaling',
            RobustScaler(quantile_range=(25, 75)).fit_transform),
        ('Yeo-Johnson',
            PowerTransformer(method='yeo-johnson').fit_transform),
        #('Box-Cox',
        #    PowerTransformer(method='box-cox').fit_transform),
        ('GaussianPDF',
            QuantileTransformer(output_distribution='normal').fit_transform),
        ('UniformPDF',
            QuantileTransformer(output_distribution='uniform').fit_transform),
        ('Normalizing',
            Normalizer().fit_transform),
    ]

    X_transformed = pd.DataFrame(index=range(len(X)))
    for i in range(len(distributions)):
        title, transformer = distributions[i]

        if title == 'Box-Cox' and np.any(np.isin(X, 0)):
            # ValueError: The Box-Cox transformation can only be applied to strictly positive data
            continue

        # Transform, and drop row which is infinite to avoid error.
        X_transformed[title] = transformer(X)

    return X_transformed


def test_distribution_of_target(y):
    print("Skewness: %f" % y.skew())
    print("Kurtosis: %f" % y.kurt())

    plt.figure(1)
    plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=stats.johnsonsu)

    plt.figure(2)
    plt.title('Normal')
    sns.distplot(y, kde=False, fit=stats.norm)

    plt.figure(3)
    plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=stats.lognorm)


def test_symmetry(df, features):
    skewness = []
    kurtosis = []

    for c in features:
        skewness.append(df[c].skew())
        kurtosis.append(df[c].kurt())

    symmetry = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis},
                            index=df[features].columns)

    return symmetry


def distplot_features(df, features):
    f = pd.melt(df, value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


def boxplot_features(df, target, features):
    f = pd.melt(df, id_vars=[target], value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, height=5)
    g = g.map(boxplot, "value", target)


def pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts = ts.groupby('time').mean()
    ts.plot(ax=ax)
    plt.xticks(rotation=90)


def pairplot_features(df, target, features):
    f = pd.melt(df, id_vars=[target], value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, height=5)
    g = g.map(pairplot, "value", target)


def OLS_regression_results(X, y):
    model = sm.OLS(y, sm.add_constant(X))
    result = model.fit()
    print(result.summary())


def test_normality(df, features):
    normal = pd.DataFrame(df[features])
    normal = normal.apply(lambda x: stats.shapiro(x.fillna(0))[1] < 0.01)

    print(normal)
    print('Do quantitative variables have normal distribution? {0}'.format(not normal.any()))

    return not normal.any()


def anova(df, target, features):
    anv = pd.DataFrame()
    anv['features'] = features
    pvals = []
    for c in features:
        samples = []
        for cls in df[c].unique():
            s = df[df[c] == cls][target].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals

    return anv.sort_values('pval')


def disparity_each_features(df, target, features):
    a = anova(df, target, features)
    a['disparity'] = np.log(1./a['pval'].values)
    sns.barplot(data=a, x='features', y='disparity')
    x = plt.xticks(rotation=90)


def encode(df, features):
    ordering = pd.DataFrame()
    ordering['val'] = df[features].unique()
    ordering.index = ordering.val
    # ordering['spmean'] = df[[features, 'SalePrice']].groupby(features).mean()['SalePrice']
    # ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()

    for cat, o in ordering.items():
        df.loc[df[features] == cat, features+'_E'] = o


def encode_qualitative(df, qualitative):
    qual_encoded = []

    for q in qualitative:
        encode(df, q)
        qual_encoded.append(q+'_E')

    print(qual_encoded)

    return qual_encoded


def barplot_spearman(df, target, features):
    spr = pd.DataFrame()
    spr['features'] = features
    spr['spearman'] = [df[c].corr(df[target], 'spearman') for c in features]
    spr = spr.sort_values('spearman', ascending=False)
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='features', x='spearman', orient='h')


def countplot_features(df, features):
    f = pd.melt(df, value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.countplot, "value")


# House Prices: EDA to ML (Beginner)
def plot_corr_matrix(df, target, features):
    features_copied = features.copy()
    features_copied.extend([target])
    nr_c = len(features_copied)
    df_copied = df[features_copied].copy()

    corr = df_copied.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, target)[target].index
    cm = np.corrcoef(df_copied[cols].values.T)

    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


def heatmap_features(df):
    plt.figure(1)
    corr = df[quantitative+['SalePrice']].corr()
    sns.heatmap(corr)

    plt.figure(2)
    corr = df[qual_encoded+['SalePrice']].corr()
    sns.heatmap(corr)

    plt.figure(3)
    corr = pd.DataFrame(np.zeros([len(quantitative)+1, len(qual_encoded)+1]),
                        index=quantitative+['SalePrice'],
                        columns=qual_encoded+['SalePrice'])
    for q1 in quantitative+['SalePrice']:
        for q2 in qual_encoded+['SalePrice']:
            corr.loc[q1, q2] = df[q1].corr(df[q2])
    sns.heatmap(corr)


def price_seqments(df, features):
    standard = df[df['SalePrice'] < 200000]
    pricey = df[df['SalePrice'] >= 200000]

    diff = pd.DataFrame()
    diff['feature'] = features
    diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean()) for f in features]

    sns.barplot(data=diff, x='feature', y='difference')
    x = plt.xticks(rotation=90)


def clustering_features(df, features):
    model = TSNE(n_components=2, random_state=0, perplexity=50)
    X = df[features].fillna(0.).values
    tsne = model.fit_transform(X)

    std = StandardScaler()
    s = std.fit_transform(X)

    pca = PCA(n_components=30)
    pca.fit(s)
    pc = pca.transform(s)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pc)

    fr = pd.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)

    print(np.sum(pca.explained_variance_ratio_))


def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)

    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))


# PCA
# PCA (Principal Component Analysis)
def my_pca():
    Bsmt = df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']]

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(Bsmt)

    # Add new column
    df.loc[:, 'PCABsmtSF'] = X_pca

    LivArea = df[['1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']]

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(LivArea)

    print(pca.components_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(X_pca)

    # Add new column
    df.loc[:, 'PCALivArea'] = X_pca


# VIF
# https://www.kaggle.com/sidshady/ml-basics-vif-and-classifiers
def calculate_vif(X, thresh=5.0):
    # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
    dropped = True

    while dropped:
        variables = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True

    return X


def calculate_vif_2(X, thresh=5.0):
    variables = range(X.shape[1])
    dropped = True

    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])

    return X[variables]


def calculate_correlation_coefficients(df, target, features):
    spr = pd.DataFrame()
    spr['features'] = features
    spr['kendalltau_rho'] = [0]*len(features)
    spr['kendalltau_p_val'] = [0]*len(features)    
    spr['pearsonr_rho'] = [0]*len(features)
    spr['pearsonr_p_val'] = [0]*len(features)
    spr['spearman_rho'] = [0]*len(features)
    spr['spearman_p_val'] = [0]*len(features)

    for i, c in enumerate(features):
        spr['kendalltau_rho'].iloc[i], spr['kendalltau_p_val'].iloc[i] = kendalltau(df[c], df[target])

    for i, c in enumerate(features):
        spr['pearsonr_rho'].iloc[i], spr['pearsonr_p_val'].iloc[i] = pearsonr(df[c], df[target])

    for i, c in enumerate(features):
        spr['spearman_rho'].iloc[i], spr['spearman_p_val'].iloc[i] = spearmanr(df[c], df[target])

    return spr


def select_features_with_pearsonr(df, target, features):
    calculate_correlation_coefficients(df, target, features)
    spr_checked = spr.query('(pearsonr_rho > -0.7 and 0.7 < pearsonr_rho) and (-0.05 < pearsonr_p_val < 0.05)')

    return spr_checked['features']

