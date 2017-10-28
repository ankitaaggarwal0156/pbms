import pandas as pd
from sklearn.model_selection import KFold
#from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#import matplotlib.pyplot as plt
#import sklearn.model_selection

def getFile(fileName):
    
    #edge case
    assert (len(fileName)>4 and fileName[-3:]=='csv')
    fileDataFrame=None
    try:
        fileDataFrame = pd.read_csv(fileName)
    except Exception as techLog:
        raise IOError('FILE READ FAIL- '+techLog.message)
    
    #edge case
    assert fileDataFrame.shape[1]>18
    
    #rename columns - so that code holds valid even if header changes in data file
    fileDataFrame.columns = ['Masked FIN','Age','Sequence No.','SURG_PROCEDURE','Duration of Surgery','SURGICAL_SPECIALTY',
                             'Surgeon Hash Name','PATIENT_TYPE','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count',
                             'SN - BM - PRBC Ordered','Allogeneic Blood Transfusion','SN - BM - Red Blood Cells',
                             'SN - BM - Fresh Frozen Plasma','SN - BM - Platelets','SN - BM - Cryoprecipitate','ResultsBeforeSurgery',
                             'ResultAfterSurgery','EBL']
    
    return fileDataFrame

def __cleanseDF(subsetDF, cleansedFileName):
    global surgery_avg_rbc,patType_avg_rbc 
    
    def dfApplyLogic_allogenic(row):    
        setAllogenicNo= row['SN - BM - Red Blood Cells']==0 and row['SN - BM - Fresh Frozen Plasma']==0 and \
                        row['SN - BM - Platelets']==0 and  row['SN - BM - Cryoprecipitate']==0   
        if setAllogenicNo: return 'No' 
        else: return 'Yes' 
    
    def __convert_time_to_sec(row):
        time_splits=row['Duration of Surgery'].split(':')
        time_in_sec= 0
        for pos in range(len(time_splits)): time_in_sec+=int(time_splits[-pos-1]) * pow(60,(pos))
        return time_in_sec
    
    #hide comment
    pd.options.mode.chained_assignment = None
    
    #replace NaN with 0 for 4 fields
    subsetDF[['SN - BM - Red Blood Cells','SN - BM - Fresh Frozen Plasma', 'SN - BM - Platelets', 'SN - BM - Cryoprecipitate']]=\
    subsetDF[['SN - BM - Red Blood Cells','SN - BM - Fresh Frozen Plasma', 'SN - BM - Platelets', 'SN - BM - Cryoprecipitate']].replace(pd.np.nan,0)
    
    #decide which rows to keep
    retainRows= (subsetDF['SN - BM - Platelets'] <= 75.0)           & (subsetDF['SN - BM - Platelets'] >= 0.0) &            \
                (subsetDF['SN - BM - Red Blood Cells'] <= 75.0)     & (subsetDF['SN - BM - Red Blood Cells'] >= 0.0) &      \
                (subsetDF['SN - BM - Fresh Frozen Plasma'] <= 75.0) & (subsetDF ['SN - BM - Fresh Frozen Plasma'] >= 0.0) & \
                (subsetDF['ResultsBeforeSurgery'] <= 20.0)          & (subsetDF['ResultsBeforeSurgery'] >= 3.0)             
    subsetDF = subsetDF[retainRows]
    
    #populate 1 field based on final values in 4 fields
    subsetDF['Allogeneic Blood Transfusion'] = subsetDF.apply(axis=1, func = dfApplyLogic_allogenic)
    
    subsetDF['Duration of Surgery']= subsetDF.apply(axis=1, func = __convert_time_to_sec)
    
    
    subsetDF = subsetDF.dropna(subset = ['SURG_PROCEDURE','ResultsBeforeSurgery','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count',
                                         'PATIENT_TYPE', 'SN - BM - Red Blood Cells','Age','SN - BM - PRBC Ordered',
                                         'SN - BM - Platelets', 'SN - BM - Fresh Frozen Plasma', 'SN - BM - Cryoprecipitate'])
    specialCharFilter = (subsetDF['SN - BM - Pre-Op INR']=='.') | (subsetDF['SN - BM - Pre-Op Platelet Count']=='.')
    subsetDF = subsetDF[~ specialCharFilter]
    subsetDF['SN - BM - Pre-Op INR']=subsetDF['SN - BM - Pre-Op INR'].astype('float64')
    subsetDF['SN - BM - PRBC Ordered']=subsetDF['SN - BM - PRBC Ordered'].astype('float64')
    subsetDF['SN - BM - Pre-Op Platelet Count']=subsetDF['SN - BM - Pre-Op Platelet Count'].astype('float64')
    
    retain_rows = (subsetDF ['SN - BM - Red Blood Cells'] <= subsetDF ['SN - BM - PRBC Ordered'])
    subsetDF = subsetDF[retain_rows]
    mean_attr = 'SN - BM - Red Blood Cells'
    surgery_avg_rbc = subsetDF.groupby('SURG_PROCEDURE')[mean_attr].agg(pd.np.mean)
    patType_avg_rbc = subsetDF.groupby('PATIENT_TYPE')[mean_attr].agg(pd.np.mean)
    surgery_avg_rbc = surgery_avg_rbc.to_dict()
    patType_avg_rbc = patType_avg_rbc.to_dict()
    
    subsetDF['SURG_PROCEDURE']= subsetDF['SURG_PROCEDURE'].apply(lambda x: surgery_avg_rbc[x])
    subsetDF['PATIENT_TYPE']= subsetDF['PATIENT_TYPE'].apply(lambda x: patType_avg_rbc[x])
    
    
    subsetDF.to_csv(cleansedFileName, index =False)
    print 'Data Cleaned'
    return subsetDF
    
def correlation(fileDataFrame):
    # find correlation matrix among following columns:
    # age, <Surgical Procedure ?>, Duration of Surgery, <Surgeon Name?>, <Patient Type ?>, 
    # SM-BM-PreOp-INR, SN-BM-PreOp Platelet Count, SN - BM - PRBCs Ordered
    # <Allogeneic Blood Transfusion?> , SN-BM-Red Blood Cells, SN-BM-Fresh Frozen Plasma
    # SN-BM-Platelets, SN-BM-Cryoprecipitate, ResultsBeforeSurgery, ResultAfterSurgery, EBL
    
    corr_AttributesList = ['Age', 'SN - BM - Pre-Op INR', 'SN - BM - Pre-Op Platelet Count', 'SN - BM - PRBC Ordered', 'SN - BM - Red Blood Cells', 
                  'SN - BM - Fresh Frozen Plasma', 'SN - BM - Platelets', 'SN - BM - Cryoprecipitate', 'ResultsBeforeSurgery','EBL',
                  'Duration of Surgery', 'SURG_PROCEDURE', 'PATIENT_TYPE']
    subsetDF = fileDataFrame[corr_AttributesList]
    
    correlation_df = subsetDF.corr(method='pearson')
    #print correlation_df
    correlation_df.to_excel('Correlation results information.xls')
    return correlation_df, corr_AttributesList


def __runMLAlgorithm( kf, predictors, to_be_predicted, subsetDF, alg ):

    train_predictors = subsetDF[predictors]    
    train_target = subsetDF[to_be_predicted]
    alg.fit(train_predictors, train_target) # model

    return alg
            
def machine_Learning(subsetDF, subset_AttributesList):
    global measured_cols, predictors
    ############### Machine Learning #######################
    predictors = ['Age', 'SN - BM - Pre-Op INR', 'SN - BM - Pre-Op Platelet Count',
                   'SURG_PROCEDURE', 'PATIENT_TYPE','ResultsBeforeSurgery']
    print "\nRunning Random Forest Regressor...\n"
    measured_cols = ['SN - BM - Red Blood Cells','SN - BM - Platelets','SN - BM - Fresh Frozen Plasma','SN - BM - Cryoprecipitate']
    rbc = 'SN - BM - Red Blood Cells'
    platelets = 'SN - BM - Platelets'
    ffp = 'SN - BM - Fresh Frozen Plasma'
    cryo = 'SN - BM - Cryoprecipitate'
    blood_type = 'Allogeneic Blood Transfusion'
    measured_cols = [rbc, platelets, ffp, cryo]
    
    kf = KFold(n_splits= 3, random_state=1)
    model_dict = {}
    for to_be_predicted in measured_cols:
        alg = RandomForestRegressor()
        p1=__runMLAlgorithm( kf, predictors, to_be_predicted, subsetDF, alg)
        #print p1.feature_importances_.tolist()
        model_dict[to_be_predicted] = p1
    
    alg2 = RandomForestClassifier()
    p5=__runMLAlgorithm(kf, predictors, blood_type, subsetDF, alg2)
    model_dict[blood_type] = p5
    measured_cols.append(blood_type)
    return model_dict

def __getResults(predictors, subsetDF,alg):
    test_predictions = alg.predict(subsetDF[predictors])
    return test_predictions

def readTestFile(fileName):
    global surg_proc_column, patient_t_column  
    def __fillSurgProc(x):
        if x in surgery_avg_rbc.keys():
            return surgery_avg_rbc[x]
        return 0.0
    
    def __fillPatType(x):
        if x in patType_avg_rbc.keys():
            return patType_avg_rbc[x]
        return 0.0
    
    assert (len(fileName)>4 and fileName[-3:]=='csv')
    fileDataFrame=None
    try:
        fileDataFrame = pd.read_csv(fileName)
    except Exception as techLog:
        raise IOError('FILE READ FAIL- '+techLog.message)
    
    #edge case
    assert fileDataFrame.shape[1]>10
    surg_proc_column = fileDataFrame['SURG_PROCEDURE'].copy()
    patient_t_column = fileDataFrame['PATIENT_TYPE'].copy()
        
    #rename columns - so that code holds valid even if header changes in data file
    fileDataFrame.columns = ['Masked FIN','Age','Sequence No.','SURG_PROCEDURE','Duration of Surgery','SURGICAL_SPECIALTY',
                             'Surgeon Hash Name','PATIENT_TYPE','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count',
                             'ResultsBeforeSurgery']
    fileDataFrame['SURG_PROCEDURE']= fileDataFrame['SURG_PROCEDURE'].apply(lambda x: __fillSurgProc(x))
    fileDataFrame['PATIENT_TYPE']= fileDataFrame['PATIENT_TYPE'].apply(lambda x: __fillPatType(x))
    return fileDataFrame

def process_SinglePatient(patientDetails):
    def __fillSurgProc(x):
        if x in surgery_avg_rbc.keys():
            return surgery_avg_rbc[x]
        return 0.0
    
    def __fillPatType(x):
        if x in patType_avg_rbc.keys():
            return patType_avg_rbc[x]
        return 0.0
    
    patientDetails[1] = __fillSurgProc(patientDetails[1])
    patientDetails[2] = __fillPatType(patientDetails[2])
    all_columns = ['Age','SURG_PROCEDURE','PATIENT_TYPE','SN - BM - Pre-Op INR','SN - BM - Pre-Op Platelet Count',
                             'ResultsBeforeSurgery']
    fileDF= pd.DataFrame(columns = all_columns)
    fileDF = fileDF.append(pd.Series(patientDetails,index = all_columns),ignore_index = True)
    return fileDF
    
def get_Model_Predictions(model_Dictionary,fileName, returnFileName, patientDetails):       
    global surg_proc_column, patient_t_column  
    
    #print alg.feature_importances_.tolist()
    proj_columns = ['Projected RBC','Projected Platelets','Projected FFP','Projected Cryoprecipitate','Projected Allogeneic Blood Transfusion']
    print "Predicting..."
    if patientDetails:
        return_Results = []
        #print patientDetails
        subsetDF = process_SinglePatient(patientDetails)
        for to_be_predicted,proj_col in zip(measured_cols,proj_columns):
            return_Results.append(__getResults(predictors, subsetDF, model_Dictionary[to_be_predicted])[0])
        return return_Results
        
    else:
        subsetDF = readTestFile(fileName)
        for to_be_predicted,proj_col in zip(measured_cols,proj_columns):
            subsetDF[proj_col] = __getResults(predictors, subsetDF, model_Dictionary[to_be_predicted])    
        subsetDF['SURG_PROCEDURE'] = surg_proc_column
        subsetDF['PATIENT_TYPE'] = patient_t_column
        subsetDF.to_csv(returnFileName ,index= False)

def fill_missing_cells(corr_df, corr_AttributesList, fileDataFrame_copy, upload_file_name):
    

    #===========================================================================
    # corr_AttributesList = ['Age', 'SN - BM - Pre-Op INR', 'SN - BM - Pre-Op Platelet Count', 'SN - BM - PRBC Ordered', 
    #                        'SN - BM - Red Blood Cells', 'SN - BM - Fresh Frozen Plasma', 'SN - BM - Platelets', 
    #                        'SN - BM - Cryoprecipitate', 'ResultsBeforeSurgery','Duration of Surgery']
    #===========================================================================
  

    # TO DO - CONVERT DURATION BACK TO MINUTE FORMAT, AFTER COMPLETING
    def __fill_across_1_row(row):
        
        # TO DO - CONVERT DURATION BACK TO MINUTE FORMAT, AFTER COMPLETING
        try:
            for focus_col in corr_AttributesList:
                if pd.np.isnan( row[focus_col] ):
                    
                    num_recom = 0;  den_recom = 0
                    for other_col in corr_AttributesList:
    
                        if (focus_col != other_col) and not pd.np.isnan( row[other_col] ):
                            
                            if max_of_columns[other_col] == 0 : continue
                            nor_val_other_col = ( row[other_col] /max_of_columns[other_col]  )*   max_of_columns[focus_col]                
                            nor_mean_other_col = ( mean_of_columns[other_col] / max_of_columns[other_col] ) *  max_of_columns[focus_col]   
                                              
                            num_recom += ( nor_val_other_col - nor_mean_other_col) * corr_df.at[focus_col, other_col]
                            den_recom +=  abs( corr_df.at[focus_col, other_col] )   
                               
                            
                    # fit-in components of recommendation in a single value  
                    row[focus_col] = mean_of_columns[focus_col] + float(num_recom)/den_recom
                    
                    if row[focus_col] <0: 
                        row[focus_col]=0
                    try:
                        if focus_col =='Age': 
                            row[focus_col]   = int(row[focus_col])
                        else: row[focus_col] = round(row[focus_col], 4)   
                    except ValueError:
                        pass            
        except Exception as e:
            print 'FAILURE ! Line 257 - '+e
            pass
        return row  
      
    # 1st convert time to seconds format, and numbers
    def __convert_time_to_sec(time_in_str):
        time_splits=time_in_str.split(':')
        time_in_sec= 0
        for pos in range(len(time_splits)): time_in_sec+=int(time_splits[-pos-1]) * pow(60,(pos))
        return time_in_sec
    
    def __convert_time_to_str(time_in_num):
        time_in_str = ''
        if time_in_num<=0:
            time_in_str = '0:00'
        else:    
            while time_in_num>0:
                time_in_num, remainder = divmod(time_in_num, 60)
                if remainder>9: time_in_str= str(int(remainder))+':'+time_in_str
                else:           time_in_str= '0'+str(int(remainder))+':'+time_in_str
            time_in_str=time_in_str[:-1]
            if len(time_in_str)==2: time_in_str='0:'+time_in_str    
        
        return time_in_str
    
    def dfApplyLogic_allogenic(row):    
        setAllogenicNo= row['SN - BM - Red Blood Cells']==0 and row['SN - BM - Fresh Frozen Plasma']==0 and \
                        row['SN - BM - Platelets']==0 and  row['SN - BM - Cryoprecipitate']==0   
        if setAllogenicNo: return 'No' 
        else: return 'Yes' 
    ##################################################################################################
    print 'Beginning to fill missing cells...'
    
    corr_AttributesList.remove('SURG_PROCEDURE')
    corr_AttributesList.remove('PATIENT_TYPE')
    
    # convert time to numbers
    fileDataFrame_copy['Duration of Surgery'] = fileDataFrame_copy['Duration of Surgery'].apply( func = __convert_time_to_sec)
    
        
    #all those columns with Non-numeric data type, force conversion with NaN                                                                                        
    dataTypesInfo = fileDataFrame_copy.dtypes 
    for column in corr_AttributesList:
        if dataTypesInfo[column] == 'object':   
            fileDataFrame_copy[column] = pd.to_numeric( fileDataFrame_copy[column], errors='coerce')
    
    #get mean of all available columns        
    mean_of_columns = fileDataFrame_copy.mean(axis=0, skipna = True)
    med_of_columns = fileDataFrame_copy.median(axis=0, skipna = True)
    max_of_columns = med_of_columns * 2
    
    # recommendation engine
    fileDataFrame_copy[corr_AttributesList] = fileDataFrame_copy[corr_AttributesList].apply(axis=1, func = __fill_across_1_row)

    
    # convert time back to xx:xx
    fileDataFrame_copy['Duration of Surgery'] = fileDataFrame_copy['Duration of Surgery'].apply( func = __convert_time_to_str)
    fileDataFrame_copy['Allogeneic Blood Transfusion'] = fileDataFrame_copy.apply(axis=1, func = dfApplyLogic_allogenic)
    
    fileDataFrame_copy.to_csv(upload_file_name, index = False)
    print 'Recommended File : '+upload_file_name
    return True



####################################################################################################################
####################################################################################################################
####################################################################################################################



def api_build_model_on_train_data(fileName):    
    fileDataFrame = getFile(fileName)
    subsetDF = __cleanseDF(fileDataFrame, 'Cleansed_Intermediate_File.csv')
    corr_df,corr_AttributesList = correlation( subsetDF )    
    model_Dictionary = machine_Learning(subsetDF, corr_AttributesList)
    
    return model_Dictionary



def api_use_model_on_test_data(model_Dictionary, fileName, returnFileName, patientDetails):
    
    if patientDetails:
        results = get_Model_Predictions(model_Dictionary, fileName,  returnFileName, patientDetails)
        return results
    else:
        get_Model_Predictions(model_Dictionary, fileName,  returnFileName, patientDetails)
        return returnFileName
    

def api_recommend_and_fill_missing_cells(fileName, upload_file_name):
    
 
    fileDataFrame = getFile(fileName) # raw user file

    subsetDF = __cleanseDF(fileDataFrame, 'Cleansed_Intermediate_File.csv')
    corr_df,corr_AttributesList = correlation( subsetDF )
    try:
        corr_AttributesList.remove('EBL')
    except ValueError: pass
    fill_missing_cells(corr_df, corr_AttributesList, fileDataFrame, upload_file_name)
    
    return upload_file_name
    
    
    