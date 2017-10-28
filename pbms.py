import os
from werkzeug import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from API_ModelPredictorTrainTest import api_build_model_on_train_data, api_use_model_on_test_data,api_recommend_and_fill_missing_cells


#global
model_Dictionary = None
uploadedFilePath = None

# Initialize the Flask application

app = Flask(__name__)
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'temp/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png','csv'])
# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    
    global uploadedFilePath,model_Dictionary
    
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        print filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploadedFilePath="temp/"+filename
        
        print request.form.get('match')
        
        if request.form.get('match'):
            # recommendation logic
           fn_ext = uploadedFilePath.rsplit('.',1)
           fn_ext.insert(1,'_Filled.')
           filled_file_to_generate = reduce ( lambda x,y:x+y, fn_ext)
           api_recommend_and_fill_missing_cells( uploadedFilePath, filled_file_to_generate)
           
           return redirect(url_for('uploaded_file',
                                filename=filled_file_to_generate[5:])) 
        else:
            return redirect(url_for('index')) 
    if uploadedFilePath:
        model_Dictionary = api_build_model_on_train_data(uploadedFilePath)
        print "trained"
        uploadedFilePath=None
    #uploadedFilePath = uploadedFilePath[5:]
    return redirect(url_for('index'))


#Executes when prediction for more than 1 patient    
@app.route('/test', methods=['POST'])
def test():
    global model_Dictionary,uploadedFilePath
    
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        print filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploadedFilePath="temp/"+filename
    
    if model_Dictionary==None :
        
        model_Dictionary=api_build_model_on_train_data("temp/Permenant Train File.csv")
        
    if uploadedFilePath:
        
        returnFileName = 'temp/Output.csv'
        PatientData=None
        op_file_name = api_use_model_on_test_data(model_Dictionary,uploadedFilePath, returnFileName, PatientData)
        print "op",op_file_name
        print "tested"
        op_file_name = op_file_name[5:]
        uploadedFilePath=None
        return redirect(url_for('uploaded_file',
                                filename=op_file_name))
    return redirect(url_for('index'))


#Executes when prediction for 1 patient    
@app.route('/test2', methods=['GET','POST'])
def test2():
    if request.method == 'POST':
        global model_Dictionary,uploadedFilePath
        
        #uploadedFilePath=None because we will pass list of user eneterd value instead of file
        uploadedFilePath=None
            
        if model_Dictionary==None :
            model_Dictionary=api_build_model_on_train_data("temp/Permenant Train File.csv")
        
        Age=request.form.get('Age')
        Surg_proc=request.form.get('Surg_proc')
        Patient_type=request.form.get('Patient_type')
        INR=request.form.get('INR')
        platelet=request.form.get('platelet')
        ResultbeforeSurgery=request.form.get('ResultbeforeSurgery')
            
        PatientDetails=[]
        PatientDetails.append(int(Age))
        PatientDetails.append(str(Surg_proc))
        PatientDetails.append(str(Patient_type))
        PatientDetails.append(float(INR))
        PatientDetails.append(float(platelet))
        PatientDetails.append(float(ResultbeforeSurgery))
        
        returnFileName = 'temp/output.csv'
        
        
        
        predictionList = api_use_model_on_test_data(model_Dictionary,uploadedFilePath, returnFileName, PatientDetails)
        predictionList = list(predictionList)
        
        #['Projected Allogeneic Blood Transfusion','Projected Cryoprecipitate','Projected FFP','Projected Platelets','Projected RBC']
        
        #print "predictionList",predictionList
        #print "tested for 1 patient"
        
        #return redirect(url_for('index',list=predictionList))
        return render_template( "index1.html",predictionList=predictionList)

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    
    app.run(debug=True)
    print "Hello"