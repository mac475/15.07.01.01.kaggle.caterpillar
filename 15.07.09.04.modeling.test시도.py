# coding: utf-8
from IPython.core.display import HTML
styles = open("../styles/custom.css", "r").read()
HTML( styles )

mode = False

import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import grid_search

df_train = pd.read_csv( './dataset/train_set.csv' )    # data를 읽어들인다.
df_test = pd.read_csv( './dataset/test_set.csv' )    # data를 읽어들인다.

def extract_year_month_day_from_quote_date( p_df ) :
    p_df[ 'quote_date' ] = pd.to_datetime( p_df[ 'quote_date' ] )    # string을 datetime으로 형변환
    p_df[ 'year' ] = p_df[ 'quote_date' ].dt.year    # 연도
    p_df[ 'month' ] = p_df[ 'quote_date' ].dt.month    # 월
    p_df[ 'day' ] = p_df[ 'quote_date' ].dt.day    # 일

    return p_df

le_supplier = preprocessing.LabelEncoder()
le_bracket_pricing = preprocessing.LabelEncoder()

le_material_id = preprocessing.LabelEncoder()
le_end_a_1x = preprocessing.LabelEncoder()
le_end_a_2x = preprocessing.LabelEncoder()
le_end_x_1x = preprocessing.LabelEncoder()
le_end_x_2x = preprocessing.LabelEncoder()
le_end_a = preprocessing.LabelEncoder()
le_end_x = preprocessing.LabelEncoder()

le_component_id = preprocessing.LabelEncoder()    # component의 경우, master dataset이 별도로 존재하므로
df_components = pd.read_csv( './dataset/components.verified.csv' )
le_component_id.fit( df_components[ 'component_id' ] )
df_components = None

le_spec_id = preprocessing.LabelEncoder()    # spec의 경우, spec meta dataset을 별도로 생성했음
df_spec_meta = pd.read_csv( './dataset/spec_meta.csv' )
le_spec_id.fit( df_spec_meta[ 'spec' ].append( pd.Series( [ '-1' ] ) ) )
df_spec_meta = ''

def executeLabelEncoding( p_df, is_init ) :
    if is_init == True :    # training dataset인 경우, label encoder 생성 및 fitting 수행
        p_df[ 'bracket_pricing' ] = le_bracket_pricing.fit_transform( p_df[ 'bracket_pricing' ] )
        p_df[ 'supplier' ] = le_supplier.fit_transform( p_df[ 'supplier' ] )
        p_df[ 'material_id' ] = le_material_id.fit_transform( p_df[ 'material_id' ] )
        p_df[ 'end_a_1x' ] = le_end_a_1x.fit_transform( p_df[ 'end_a_1x' ] )
        p_df[ 'end_a_2x' ] = le_end_a_2x.fit_transform( p_df[ 'end_a_2x' ] )
        p_df[ 'end_x_1x' ] = le_end_x_1x.fit_transform( p_df[ 'end_x_1x' ] )
        p_df[ 'end_x_2x' ] = le_end_x_2x.fit_transform( p_df[ 'end_x_2x' ] )
        p_df[ 'end_a' ] = le_end_a.fit_transform( p_df[ 'end_a' ] )
        p_df[ 'end_x' ] = le_end_x.fit_transform( p_df[ 'end_x' ] )
    else :
        p_df[ 'bracket_pricing' ] = le_bracket_pricing.transform( p_df[ 'bracket_pricing' ] )
        p_df[ 'supplier' ] = le_supplier.transform( p_df[ 'supplier' ] )
        p_df[ 'material_id' ] = le_material_id.transform( p_df[ 'material_id' ] )
        p_df[ 'end_a_1x' ] = le_end_a_1x.transform( p_df[ 'end_a_1x' ] )
        p_df[ 'end_a_2x' ] = le_end_a_2x.transform( p_df[ 'end_a_2x' ] )
        p_df[ 'end_x_1x' ] = le_end_x_1x.transform( p_df[ 'end_x_1x' ] )
        p_df[ 'end_x_2x' ] = le_end_x_2x.transform( p_df[ 'end_x_2x' ] )
        p_df[ 'end_a' ] = le_end_a.transform( p_df[ 'end_a' ] )
        p_df[ 'end_x' ] = le_end_x.transform( p_df[ 'end_x' ] )

    for i in range( 1, 9 ) :    # bill_of_materials에서 merge된 component_id_1~8을 label encoding 수행
        comp_str = 'component_id_' + str( i )
        p_df[ comp_str ] = le_component_id.transform( p_df[ comp_str ] )

    for i in range( 1, 11 ) :    # specs에서 merge된 spec1~10을 label encoding 수행
        spec_str = 'spec' + str( i )
        p_df[ spec_str ] = le_spec_id.transform( p_df[ spec_str ] )
    return p_df

df_train[ 'id' ] = '99999'    # test와 join위해 feature 추가 : 99999는 train dataset이다
df_merged = df_train.append( df_test )    # train과 test df를 merge한다.

df_tube_bill_specs_end = pd.read_csv( './dataset/tube_full_tmp.csv',
                                  dtype = { 'component_id_8' : str, 'spec9' : str, 'spec10' : str,
                                            'component_id_7' : str, 'spec8' : str, '69' : str } )

df_merged = df_merged.merge( df_tube_bill_specs_end, how = 'inner', on = 'tube_assembly_id' )

def process_nulls( p_df ) :
    for i in range( 1, 9 ) :    # df_merged내의 null을 -1으로 채워둔다
        comp_str = 'component_id_' + str( i )
        quan_str = 'quantity_' + str( i )

        print( comp_str, quan_str )

        p_df[ comp_str ].fillna( '-1', inplace = True )
        p_df[ quan_str ].fillna( 0, inplace = True )

    for i in range( 1, 11 ) :    # df_merged내의 null을 -1으로 채워둔다
        spec_str = 'spec' + str( i )

        print( spec_str )

        p_df[ spec_str ].fillna( '-1', inplace = True )

    p_df.fillna( -1, inplace = True )

    return p_df

df_merged = process_nulls( df_merged )    # merge된 df내의 null 처리
df_merged = executeLabelEncoding( df_merged, is_init = True )    # label encoding을 수행한다
df_merged = ''

df_train.drop( 'id', axis = 1, inplace = True )    # 일단, 필요없는 것들 제거 : df_train을 원상 복구해둠
df_test = ''    # 일단, 필요없는 것들 제거
df_merged = ''    # 일단, 필요없는 것들 제거


list_for_remove = []    # 전역변수의 선언
def executeFeatureRemoval( p_df ) :
    # 제거하고자 하는 feature list
    global list_for_remove    # 전역변수 명시
    list_for_remove = [    # 82.7% 정확도 feature list : default
                        'tube_assembly_id',
#                         'supplier',

                        'quote_date',
#                         'annual_usage',
#                         'min_order_quantity',

    #                     'bracket_pricing',
    #                     'quantity',

#                         'year',
#                         'month',
#                         'day',

#                         'material_id',
#                         'diameter',
#                         'wall',
#                         'length',

#                         'num_bends',
#                         'bend_radius',

#                         'end_a_1x',
#                         'end_a_2x',
#                         'end_x_1x',
#                         'end_x_2x',
#                         'end_a',
#                         'end_x',

#                         'num_boss',
#                         'num_bracket',
#                         'other',

#                         'component_id_1',
#                         'quantity_1',
#                         'component_id_2',
#                         'quantity_2',
#                         'component_id_3',
#                         'quantity_3',
#                         'component_id_4',
#                         'quantity_4',

                        'component_id_5',
                        'quantity_5',
                        'component_id_6',
                        'quantity_6',
                        'component_id_7',
                        'quantity_7',
                        'component_id_8',
                        'quantity_8',

#                         'comp_type_count',
#                         'comp_total_count',

#                         'tube_volume',

#                         'spec1',
                        'spec2',
                        'spec3',
                        'spec4',
                        'spec5',
                        'spec6',
                        'spec7',
                        'spec8',
                        'spec9',
                        'spec10',
#                         'spec_type_count',

#                         'forming_x',
#                         'forming_y',

                        'component_id_x', 'component_id_y',
                        '69_x', '69_y',
                      ]

    return p_df.drop( list_for_remove, axis = 1, inplace = False )

df_train = extract_year_month_day_from_quote_date( df_train )    # feature 처리를 수행
df_train_merged = df_train.merge( df_tube_bill_specs_end, how = 'inner', on = 'tube_assembly_id' )
df_train_merged = process_nulls( df_train_merged )
df_train_merged = executeLabelEncoding( df_train_merged, is_init = False )    # label encoding을 수행한다
df_train_merged = executeFeatureRemoval( df_train_merged )    # feature removal 처리를 수행
df_test = pd.read_csv( './dataset/test_set.csv' )    # data를 읽어들인다.
df_test = extract_year_month_day_from_quote_date( df_test )    # feature 처리를 수행
df_result = pd.DataFrame( df_test[ 'id' ], columns = ['id'] )    # 결과용 dataframe을 생성
df_test.drop( 'id', axis = 1, inplace = True )    # id feature는 일단 제거
df_test_merged = df_test.merge( df_tube_bill_specs_end, how = 'inner', on = 'tube_assembly_id' )
df_test_merged = process_nulls( df_test_merged )
df_test_merged = executeLabelEncoding( df_test_merged, is_init = False )    # label encoding을 수행한다
df_test_merged = executeFeatureRemoval( df_test_merged )    # feature removal 처리를 수행

X = df_train_merged.drop( 'cost', axis = 1, inplace = False )    # X를 확보
y = np.log1p( df_train_merged[ 'cost' ] )   # y를 확보

df_train = ''
df_train_merged = ''

cv_cnt = 10    # cv : cross validation 횟수
n_jobs_cnt = 6

if mode == True :
    model_list = [
                   RandomForestRegressor( max_depth = 100, n_estimators = 500, n_jobs = n_jobs_cnt ),    # 제출용
                 ]
else :
    model_list = [
                   RandomForestRegressor( n_jobs = n_jobs_cnt ),    # 테스트용
                 ]

get_ipython().run_cell_magic('time', '', "import time\nfor model in model_list :\n    params = {    # 현재, 최적의 params 조건은 max_depth = 100, n_estimators = 500임\n#                'max_depth' : (10,30,50,100,200),    # RandomForest\n#                'n_estimators' : (10,20,50,100,200,500),    # RandomForest\n             }\n\n    gs = grid_search.GridSearchCV( model,\n                                   param_grid = params,\n#                                    n_jobs = n_jobs_cnt,\n                                   cv = cv_cnt,\n#                                    scoring = scorer\n                                 )\n    gs.fit( X, y )\n    \n    \n    print( 'model : ', str( model ).split( '(' )[0] )\n    print( 'best_score : ', gs.best_score_ )\n    \n    print( '=================' )\n    print( 'best model : ', gs.best_estimator_ )\n    print( '=================' )\n    \n    df_feature_importance = pd.DataFrame( X.columns.values, columns = [ 'features' ] )\n    df_feature_importance[ 'importance' ] = gs.best_estimator_.feature_importances_\n#     print( df_feature_importance.sort( 'importance', ascending = False  ) )    \n    \n    \n    y_pred = gs.best_estimator_.predict( df_test_merged )    # prediction 수행\n#     df_result[ 'cost' ] = y_pred\n    df_result[ 'cost' ] = np.expm1( y_pred )\n#     df_result[ 'cost' ] = df_result[ 'cost' ]\n    \n    now = time.strftime( '%Y%m%d%H%M%S' )    # 현재시각을 확보\n    model_name = str( model ).split( '(' )[0]    # 파일생성용\n    file_timestamp = now[2:4] + now[4:6] + now[6:8] + now[8:10] + now[10:12] + now[12:14]\n    accuracy = '{0:.1f}%'.format( gs.best_score_ * 100 )    # latitude에 대한 예측률 저장 : 파일명 활용용도임\n    df_result.to_csv( path_or_buf = './result.to.submit/' + file_timestamp + '.' + model_name +\n                      '.' + accuracy + '.result().csv', sep = ',', index = False )")

list_features = df_test_merged.columns.values.tolist()

with open( 'performance.condition.history.txt', 'a' ) as history_file :    # 모델의 이력을 logging
    history_file.write( '======================================================================\n' )
    history_file.write( '0. Description :\n' )
    if mode == True :
        history_file.write( '\tmode : Production mode\n' )
    else :
        history_file.write( '\tmode : Test mode\n' )
    history_file.write( '1. File name :\n\t' + file_timestamp + '.' + model_name + '.' + accuracy + '\n' )
    history_file.write( '2. Model information :\n\t' + str( gs.best_estimator_ ) + '\n' )
    history_file.write( '3. Applied features :\n' )
    for feature in list_features :
        history_file.write( '\t' + feature + '\n' )
    history_file.write( '4. Removed features :\n' )
    for feature in list_for_remove :
        history_file.write( '\t' + feature + '\n' )
    history_file.write( '5. Feature importance :\n\t' )
    history_file.write( df_feature_importance.sort( 'importance', ascending = False  ).to_string() )
    history_file.write( '\n' )
    history_file.close()

df_result.head( 3 )
df_result_desc = df_result.describe()    # 결과의 overview 확인
df_result_desc.drop( 'id', axis = 1, inplace = True )
df_result_desc.describe()

X = ''    # X 초기화
y = ''

df_train = ''
df_train_merged = ''

df_test = ''
df_test_merged = ''

df_components = ''
df_merged = ''

df_result_desc = ''
df_result = ''
df_tmp = ''
df_feature_importance = ''

df_tube_bill = ''