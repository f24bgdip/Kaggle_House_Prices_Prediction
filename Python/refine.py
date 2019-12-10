# Abstraction
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer


# Drop
def treat_outliers(train, test):
    # Id: 1299. Why is this house so cheap? I want to buy this one.
    # Id: 524. This one is cheap too.
    train = train.drop(index=[
        1299, 524,  # Outliers between GrLivArea and SalePrice
        935, 314    # Outliers between  and SalePrice
    ], errors='ignore')

    test.loc[test.GarageYrBlt > 2200, 'GarageYrBlt'] = test.loc[test.GarageYrBlt > 2200, 'GarageYrBlt'].apply(lambda x: x - 100)

    return train, test


def filling_missing_value_to_0(df, features):
    print(features)

    # fillna with 0
    for c in features:
        df[c].fillna(0, inplace=True)


def filling_missing_value_to_mean(df, features):
    print(features)

    # fillna with mean
    for c in features:
        df[c].fillna(df[c].mean(), inplace=True)


def filling_missing_value_to_Missing(df, features):
    print(features)

    # Replace 'NaN' with 'Missing' because 'None' is existing.
    for c in features:
        df[c].fillna('Missing', inplace=True)


def imputation_strategy_base(classified_features):

    imputation_strategy = np.array([
        ['num_nan_zero',    np.nan, 'constant',         0,         classified_features['numeric']],
        ['num_nan_mean',    np.nan, 'mean',             None,      np.array([], dtype='object')],
        ['cat_nan_none',    np.nan, 'constant',         'None',    np.array([], dtype='object')],
        ['cat_nan_miss',    np.nan, 'constant',         'Missing', classified_features['categorical']],
        ['nan_most',        np.nan, 'most_frequent',    None,      np.array([], dtype='object')],
        ['date_nan_dummy',  np.nan, 'mean',             None,      classified_features['datetime']],
    ])

    return imputation_strategy


# %%
def imputation_strategy(classified_features):

    cate_features_be_none = np.array([
        'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MasVnrType',
        'PoolQC'], dtype='object')

    cate_features_be_missing = classified_features['categorical'][np.isin(classified_features['categorical'], ['Electrical', 'MSZoning', *cate_features_be_none], invert=True)]

    imputation_strategy = np.array([
        ['num_nan_zero',    np.nan, 'constant',         0,         classified_features['numeric']],
        ['num_nan_mean',    np.nan, 'mean',             None,      np.array([], dtype='object')],
        ['cat_nan_none',    np.nan, 'constant',         'None',    cate_features_be_none],
        ['cat_nan_miss',    np.nan, 'constant',         'Missing', cate_features_be_missing],
        ['nan_most',        np.nan, 'most_frequent',    None,      np.array(['Electrical', 'MSZoning'], dtype='object')],
        ['date_nan_dummy',  np.nan, 'mean',             None,      classified_features['datetime']],
    ])

    return imputation_strategy


# %%
def impute_features(df, imputation_strategy):
    # imputation transformer
    imputers = ColumnTransformer([
        (name, SimpleImputer(missing_values=mis_value, strategy=strategy, fill_value=value), npary) for name, mis_value, strategy, value, npary in imputation_strategy])

    # impute
    imputed_ndarray = imputers.fit_transform(df)

    # convert from ndarray to dataframe
    chain = itertools.chain(*list([f for _, _, _, _, f in imputation_strategy]))
    imputed_df = pd.DataFrame(imputed_ndarray, columns=list(chain))
    # imputed_df[features['numeric']] = imputed_df[features['numeric']].astype('float64')

    return imputed_df


# %%
# Replace and combine levels in each feature.
# Before encoding the category values, I replaced them with words for analysis.
def dimensionality_reduction_strategy():
    strategy = pd.DataFrame([
                    [['BsmtCond'],                                                                                       ['Po'],                                 ['None']],
                    [['BldgType'],                                          [['2fmCon', 'Duplex', 'Twnhs'], ['TwnhsE', '1Fam']],                      ['Low', 'Moderate']],
    [['Condition1', 'Condition2'],                    [['Artery', 'Feedr'], ['RRAe', 'RRAn', 'RRNe', 'RRNn'], ['PosN', 'PosA']],        ['Road', 'Railway', 'PosFeature']],
                  [['Electrical'],                                              [['FuseA', 'FuseF', 'FuseP', 'Mix'], ['SBrkr']],                      ['Low', 'Moderate']],
  [['Exterior1st', 'Exterior2nd'],                          [['AsbShng', 'AsphShn', 'BrkComm', 'Brk Cmn', 'CBlock', 'Missing'],
                                       ['BrkFace', 'HdBoard', 'MetalSd', 'Plywood', 'Stucco', 'Wd Sdng', 'WdShing', 'Wd Shng'],
                                                                ['CemntBd', 'CmentBd', 'ImStucc', 'Other', 'Stone', 'VinylSd']],              ['Low', 'Moderate', 'High']],
                       [['Fence'],                                                         [['MnWw', 'GdWo', 'MnPrv', 'GdPrv']],                                ['Fence']],
                  [['Foundation'],                                 [['Slab'], ['BrkTil', 'CBlock'], ['Stone', 'Wood', 'PConc']],              ['Low', 'Moderate', 'High']],
                  [['Functional'],                                    [['Maj1', 'Maj2', 'Sev', 'Sal'], ['Min1', 'Min2', 'Mod']],                      ['Damaged', 'Used']],
                  [['GarageCond'],                                                                             [['Po'], ['Ex']],                           ['None', 'Gd']],
                  [['GarageType'],                [['CarPort', 'None'], ['2Types', 'Detchd', 'Basment'], ['Attchd', 'BuiltIn']],              ['Low', 'Moderate', 'High']],
                     [['Heating'],                                        [['Grav', 'Floor', 'OthW', 'Wall'], ['GasA', 'GasW']],                      ['Low', 'Moderate']],
                  [['HouseStyle'],         [['1.5Unf'], ['1Story', '1.5Fin', '2.5Unf', 'SFoyer', 'SLvl'], ['2.5Fin', '2Story']],              ['Low', 'Moderate', 'High']],
                   [['LandSlope'],                                                                             [['Mod', 'Sev']],                                ['Slope']],
                   [['LotConfig'],                                                                             [['FR2', 'FR3']],                             ['Frontage']],
                    [['LotShape'],                                                                      [['IR1', 'IR2', 'IR3']],                                ['IR123']],
                  [['MasVnrType'],                                                 [['BrkCmn', 'None'], ['BrkFace'], ['Stone']],              ['Low', 'Moderate', 'High']],
                 [['MiscFeature'],                                                   [['TenC', 'Shed', 'Othr', 'Gar2', 'Elev']],                          ['MiscFeature']],
                    [['MSZoning'],                                                                               [['RM', 'RH']],                                 ['RM&H']],
                [['Neighborhood'],                                    [['Missing'], ['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide'],
                                         ['Edwards', 'OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill'], ['NAmes', 'Mitchel'],
                                                ['SawyerW', 'NWAmes'], ['Blmngtn', 'CollgCr', 'Gilbert', 'Crawfor', 'ClearCr'],
                                                           ['Somerst', 'Timber', 'Veenker'], ['NoRidge', 'NridgHt', 'StoneBr']],                 [0, 1, 2, 3, 4, 5, 6, 7]],
                    [['RoofMatl'],      [['ClyTile', 'CompShg', 'Metal', 'Roll', 'Tar&Grv'], ['Membran', 'WdShake', 'WdShngl']],                      ['Low', 'Moderate']],
                   [['RoofStyle'],                                 [['Gambrel'], ['Gable'], ['Flat', 'Hip', 'Mansard', 'Shed']],              ['Low', 'Moderate', 'High']],
                    [['SaleType'],            [['ConLD', 'ConLw', 'COD', 'Oth'], ['ConLI', 'CWD', 'WD', 'VWD'], ['Con', 'New']],              ['Low', 'Moderate', 'High']],
               [['SaleCondition'],                      [['AdjLand'], ['Abnorml', 'Family'], ['Alloca', 'Normal'], ['Partial']],     ['Low', 'LowMid', 'HighMid', 'High']],
                   [['Utilities'],                                                    [['NoSewr', 'NoSeWa', 'ELO'], ['AllPub']],                      ['Low', 'Moderate']],
                [['BedroomAbvGr'],                                                                          [[0], [5, 6, 7, 8]],                                   [1, 4]],
                [['BsmtFullBath'],                                                                                  [[2, 3, 4]],                                      [1]],
                [['BsmtHalfBath'],                                                                                  [[2, 3, 4]],                                      [1]],
                  [['Fireplaces'],                                                                                          [4],                                      [3]],
                  [['GarageCars'],                                                                                     [[4, 5]],                                      [3]],
                    [['HalfBath'],                                                                                          [2],                                      [1]],
                [['KitchenAbvGr'],                                                                                   [[0], [3]],                                   [1, 2]],
                 [['OverallQual'],                                                                                          [1],                                      [2]],
                      [['PoolQC'],                                                                               [[2, 3, 4, 5]],                                      [1]],
                [['TotRmsAbvGrd'],                                                                       [[1, 2], [13, 14, 15]],                                  [3, 12]],
    ], columns=['features', 'to_reps', 'values'])

    return strategy


# %%
def reduce_dimensions(df, strategy):
    if isinstance(strategy, pd.DataFrame):
        strategy = strategy.to_numpy()

    for f_list, to_rep_list, v_list in strategy:
        existance = df.columns.isin(f_list)
        # print('exist:{} / features:{} {} / to_rep:{} {} / v:{} {}'.format(np.any(exist), f_list, type(f_list), to_rep_list, type(to_rep_list), v_list, type(v_list)))
        for to_rep, v in zip(*[to_rep_list, v_list]):
            if any(existance):
                # print('feature:{}{} / to_rep:{}{} / v:{}{}'.format(f_list, type(f_list), to_rep, type(to_rep), v, type(v)))
                df.loc[:, existance] = df.loc[:, existance].replace(to_rep, v)
                # print(df.loc[:, existance])


def convert_dict_multiplekeys_singlekey(dict_multiplekeys):
    dict_singlekey = dict()
    for kset, values in dict_multiplekeys.items():
        if isinstance(kset, tuple):
            for k in kset:
                dict_singlekey[k] = values
        else:
            dict_singlekey[kset] = values

    return dict_singlekey


def remove_missing_features(df, strategy_dict):
    for k in strategy_dict.keys():
        if k in df.columns:
            pass
        else:
            del strategy_dict[k]


def onehot_encoding_strategy(df):
    exclusion_columns = ['Neighborhood']
    category = df.select_dtypes(include=['object']).columns.drop(exclusion_columns, errors='ignore').values
    onehot_strategy = ['MSSubClass', 'OverallQual', 'MoSold', 'YrSold', *category]

    return onehot_strategy


def ordinaly_encoding_strategy(df):
    ordinal_features_multiplekeys = {
        ('Alley', 'Street'): ['Missing', 'Grvl', 'Pave'],
        'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
        ('BsmtCond', 'BsmtQual', 'GarageCond', 'GarageQual', 'PoolQC'): ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ('BsmtFinType1', 'BsmtFinType2'): ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        ('ExterCond', 'ExterQual', 'FireplaceQu', 'HeatingQC', 'KitchenQual'): ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'CentralAir': ['Missing', 'N', 'Y'],
        ('Condition1', 'Condition2'): ['Missing', 'Road', 'Railway', 'Norm', 'PosFeature'],
        ('BldgType', 'Electrical', 'Heating', 'RoofMatl', 'Utilities'): ['Missing', 'Low', 'Moderate'],
        ('Exterior1st', 'Exterior2nd', 'Foundation', 'GarageType', 'HouseStyle', 'MasVnrType', 'RoofStyle', 'SaleType'): ['Missing', 'Low', 'Moderate', 'High'],
        'Fence': ['Missing', 'Fence'],
        'Functional': ['Missing', 'Damaged', 'Used', 'Typ'],
        'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
        'LandContour': ['Missing', 'Bnk', 'Lvl', 'Low', 'HLS'],
        'LandSlope': ['Missing', 'Gtl', 'Slope'],
        'LotShape': ['Missing', 'Reg', 'IR123'],
        'LotConfig': ['Missing', 'Inside', 'Frontage', 'Corner', 'CulDSac'],
        'MiscFeature': ['Missing', 'MiscFeature'],
        'MSZoning': ['Missing', 'C (all)', 'RM&H', 'RL', 'FV'],
        'PavedDrive': ['Missing', 'N', 'P', 'Y'],
        'SaleCondition': ['Missing', 'Low', 'LowMid', 'HighMid', 'High']
    }
    ordinal_features_singlekey = convert_dict_multiplekeys_singlekey(ordinal_features_multiplekeys)
    remove_missing_features(df, ordinal_features_singlekey)

    return ordinal_features_singlekey


# %%
# approval
def encode_categories(df, ordinal_features):
    category_encorders = ColumnTransformer([
        *[(name, OrdinalEncoder(categories=[category], dtype='float64'), [name]) for name, category in ordinal_features.items()]
        # ('date', )
    ])

    encoded_ndarray = category_encorders.fit_transform(df[[*ordinal_features]])
    chain = itertools.chain([*ordinal_features])
    encoded_df = pd.DataFrame(encoded_ndarray, columns=list(chain), dtype='float64')
    df[list([*ordinal_features])] = encoded_df

    return df


# %%
def scaling_strategy(df, classified_features):
    # date_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    quantitative_features = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    quantitative_featues_addition = ['TotalSF', 'Total_sqr_footage', 'Total_porch_sf', 'Area']
    quantitative_features = [*quantitative_features, *quantitative_featues_addition]

    drop_features = drop_strategy(df)
    qualitative_featues = df.columns.drop([*quantitative_features, *drop_features], errors='ignore').to_list()

    strategy = {
        'minmaxscaling': qualitative_featues,
        'standardscaling': quantitative_features,
        'powerscaling': []}

    return strategy


# %%
# approval
def scaled_features(df, strategy):
    scalers = ColumnTransformer([
        ('normalization', MinMaxScaler(), strategy['minmaxscaling']),
        ('standardization', StandardScaler(), strategy['standardscaling']),
        ('powerscaling', PowerTransformer(), strategy['powerscaling'])
    ])

    scaled_ndarray = scalers.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_ndarray, columns=[*strategy['minmaxscaling'], *strategy['standardscaling'], *strategy['powerscaling']], dtype='float64')

    return scaled_df


def drop_strategy(df):
    # LotConfig.
    # Fence. I couldn't find this value.
    # MiscFeature. MiscVal means this value.

    columns = ['Id', 'SalePrice']
    onehot_features = onehot_encoding_strategy(df)
    manually_drop_features = [
        'MSSubClass',
        'BsmtFinSF1', 'BsmtFinSF2''BsmtUnfSF',
        'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath',
        'Neighborhood',
        'Utilities',
        'YrSold', 'MoSold',
        'MiscVal']
    addition = [0, 'BuiltPeriod', 'DataFile', 'District']
    return list([*columns, *onehot_features, *manually_drop_features, *addition])


# %%
def check_facilityies(df):
    df['HasBasement'] = df['TotalBsmtSF'].apply(lambda x: 'Y' if x > 0 else 'N')
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 'Y' if x > 0 else 'N')
    df['HasMasVnr'] = df['MasVnrArea'].apply(lambda x: 'Y' if x > 0 else 'N')
    df['HasPorch'] = df['OpenPorchSF'].apply(lambda x: 'Y' if x > 0 else 'N')
    df['HasPool'] = df['PoolArea'].apply(lambda x: 'Y' if x > 0 else 'N')
    df['IsNew'] = df['SaleType'].apply(lambda x: 'Y' if x == 'New' else 'N')


# %%
def additional_featues(df):
    df['YrSold'] = df['YrSold'].astype('int64')
    df['MoSold'] = df['MoSold'].astype('int64')

    # if not 'DateSold' in df.columns:
    #     date_time = pd.to_datetime(df['YrSold'].astype(str)+'/'+df['MoSold'].astype(str), format='%Y/%m', errors='ignore')
    #     date_str = date_time.map(lambda x: x.strftime('%Y%m'))
    #     df.insert(loc=df.columns.get_loc('MoSold'), column='DateSold', value=date_str)
    #     df['DateSold'] = df['DateSold'].astype('int64')

    addition = pd.DataFrame()
    addition['SaleSeason'] = df['MoSold'].apply(lambda x: 1 if (x == 6 | x == 7) else 0)

    year_bins = [1850, 1950, 1975, 2000, 2025]
    addition['BuiltPeriod'] = pd.cut(df.YearBuilt, year_bins)

    addition['Antique'] = pd.Series(np.zeros(len(df)), dtype='int64')
    addition.loc[186, 'Antique'] = 1

    addition['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    addition['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    addition['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])
    addition['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    addition['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    # Area
    Basement = df['BsmtFinSF1']*0.8+df['BsmtFinSF2']*0.8+df['BsmtUnfSF']*0.6
    LiveArea = df['1stFlrSF']+df['2ndFlrSF']+df['WoodDeckSF']*0.8+df['OpenPorchSF']*0.8+df['EnclosedPorch']*0.8+df['3SsnPorch']*0.9+df['ScreenPorch']*0.9+df['PoolArea']*0.8
    addition['Area'] = Basement + LiveArea + df['GarageArea']

    return addition


def up_low_in_neighborhood(df):
    mask_low = get_overallqual_mask(df)

    up_low = pd.DataFrame(np.zeros(shape=(len(df), 1)), columns=['District'], dtype='object')
    up_low.loc[mask_low, 'District'] = df.loc[mask_low, 'Neighborhood'].apply(lambda x: x + '_Lower')
    up_low.loc[~mask_low, 'District'] = df.loc[~mask_low, 'Neighborhood'].apply(lambda x: x + '_Upper')

    return up_low


def up_low_in_neighborhood_old(df):
    mask_low = get_overallqual_mask(df)

    up_low = pd.DataFrame(np.zeros(shape=(len(df), 2)), columns=['LowerDistrict', 'UpperDistrict'], dtype='object')
    up_low.loc[mask_low, 'LowerDistrict'] = df.loc[mask_low, 'Neighborhood'].apply(lambda x: 'Lower_' + x)
    up_low.loc[np.any([up_low.LowerDistrict == 0.0], axis=0), 'LowerDistrict'] = 'Upper'
    up_low.loc[~mask_low, 'UpperDistrict'] = df.loc[~mask_low, 'Neighborhood'].apply(lambda x: 'Upper_' + x)
    up_low.loc[np.any([up_low.UpperDistrict == 0.0], axis=0), 'UpperDistrict'] = 'Lower'

    return up_low


def to_object(df):
    df['MSSubClass'] = df['MSSubClass'].astype('object')
    df['OverallQual'] = df['OverallQual'].astype('object')
    df['OverallCond'] = df['OverallCond'].astype('object')
    df['BsmtFullBath'] = df['BsmtFullBath'].astype('object')
    df['BsmtHalfBath'] = df['BsmtHalfBath'].astype('object')
    df['FullBath'] = df['FullBath'].astype('object')
    df['HalfBath'] = df['HalfBath'].astype('object')
    df['Fireplaces'] = df['Fireplaces'].astype('object')
    df['GarageCars'] = df['GarageCars'].astype('object')
    df['YrSold'] = df['YrSold'].astype('object')
    df['MoSold'] = df['MoSold'].astype('object')


def preprocessing_data(df):
    quadratic(df, 'OverallQual')
    quadratic(df, 'YearBuilt')
    quadratic(df, 'YearRemodAdd')
    quadratic(df, 'TotalBsmtSF')
    quadratic(df, '2ndFlrSF')
    quadratic(df, 'Neighborhood_E')
    quadratic(df, 'RoofMatl_E')
    quadratic(df, 'GrLivArea')

    qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
           '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

    df['HasBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasMasVnr'] = df['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasWoodDeck'] = df['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasPorch'] = df['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['IsNew'] = df['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

    boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr',
               'HasWoodDeck', 'HasPorch', 'HasPool', 'IsNew']

    return qdr, boolean


def preprocessing_data_with_patsy(df):
    y, X = patsy.dmatrices(
        "SalePrice ~ \
            GarageCars + \
            np.log1p(BsmtFinSF1) + \
            ScreenPorch + \
            Condition1_E + \
            Condition2_E + \
            WoodDeckSF + \
            np.log1p(LotArea) + \
            Foundation_E + \
            MSZoning_E + \
            MasVnrType_E + \
            HouseStyle_E + \
            Fireplaces + \
            CentralAir_E + \
            BsmtFullBath + \
            EnclosedPorch + \
            PavedDrive_E + \
            ExterQual_E + \
            bs(OverallCond, df=7, degree=1) + \
            bs(MSSubClass, df=7, degree=1) + \
            bs(LotArea, df=2, degree=1) + \
            bs(FullBath, df=3, degree=1) + \
            bs(HalfBath, df=2, degree=1) + \
            bs(BsmtFullBath, df=3, degree=1) + \
            bs(TotRmsAbvGrd, df=2, degree=1) + \
            bs(LandSlope_E, df=2, degree=1) + \
            bs(LotConfig_E, df=2, degree=1) + \
            bs(SaleCondition_E, df=3, degree=1) + \
            OverallQual + np.square(OverallQual) + \
            GrLivArea + np.square(GrLivArea) + \
            Q('1stFlrSF') + np.square(Q('1stFlrSF')) + \
            Q('2ndFlrSF') + np.square(Q('2ndFlrSF')) +  \
            TotalBsmtSF + np.square(TotalBsmtSF) +  \
            KitchenAbvGr + np.square(KitchenAbvGr) +  \
            YearBuilt + np.square(YearBuilt) + \
            Neighborhood_E + np.square(Neighborhood_E) + \
            Neighborhood_E:OverallQual + \
            MSSubClass:BldgType_E + \
            ExterQual_E:OverallQual + \
            PoolArea:PoolQC_E + \
            Fireplaces:FireplaceQu_E + \
            OverallQual:KitchenQual_E + \
            GarageQual_E:GarageCond + \
            GarageArea:GarageCars + \
            Q('1stFlrSF'):TotalBsmtSF + \
            TotRmsAbvGrd:GrLivArea",
        df.to_dict('list'))

    return X, y


#%%
def get_housestyle_mask(df):
    mask_double = np.any([df.HouseStyle.isin(['2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])], axis=0)

    return np.reshape(mask_double, (len(df.index)))


def split_by_housestyle(train, test=None):
    mask_double_train = get_housestyle_double(train)
    mask_double_test = get_housestyle_double(test)

    train_double = train.loc[mask_double_train]
    train_single = train.loc[~mask_double_train]
    if test == None:
        test_double = None
        test_single = None
    else:
        test_double = test.loc[mask_double_test]
        test_single = test.loc[~mask_double_test]

    return train_single, train_double, test_single, test_double


def get_saleprice_mask(df):
    mask_cheap = np.any([
        [df.MSZoning == 'C (all)'],
        [df.Neighborhood.isin(['BrDale', 'MeadowV'])],
        [df.OverallQual < 3],
        [df.ExterQual.isin(['None', 'Po', 'Fa'])],
        [df.BsmtExposure == 'Missing'],
        [df.Heating == 'Grav'],
        [df.KitchenQual.isin(['None', 'Po', 'Fa'])],
        [df.GarageType == 'Missing'],
        [df.GarageQual.isin(['None', 'Po', 'Fa'])],
        [df.SaleCondition == 'AdjLand']], axis=0)

    return np.reshape(mask_cheap, (len(df.index)))


def split_by_saleprice(train, test=None):
    mask_cheap_train = get_saleprice_mask(train)
    mask_cheap_test = get_saleprice_mask(test)

    train_cheap = train.loc[mask_cheap]
    train_pricy = train.loc[~mask_cheap]
    if test == None:
        test_cheap = None
        test_pricy = None
    else:
        test_cheap = test.loc[mask_cheap]
        test_pricy = test.loc[~mask_cheap]

    return train_cheap, train_pricy, test_cheap, test_pricy


def get_overallqual_mask(df):
    mask_low = np.any([df.OverallQual < 6], axis=0)

    return np.reshape(mask_low, (len(df.index)))


def split_by_overallqual(train, test=None):
    mask_low = get_low_overallqual_mask(df)

    train_low = train.loc[mask_low]
    train_high = train.loc[~mask_low]
    if test == None:
        test_low = None
        test_high = None
    else:
        test_low = test.loc[mask_cheap]
        test_high = test.loc[~mask_cheap]


