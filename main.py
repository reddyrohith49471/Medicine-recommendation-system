from flask import Flask, request, render_template, jsonify,render_template_string,url_for  # Import jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

html_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Health Care Center</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9" crossorigin="anonymous">
  </head>

  <style>

        .logo {
            width: 50px;
            height: 50px;
            color: black;
            margin-top: 0;
            margin-left: 2px;
        }

        .myimg {
            width: 50px;
            height: 50px;
            border: 2px solid black;
            border-radius: 25px;
        }




    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <!-- Logo at the top-left corner -->
            <div class="logo">
                <img class="myimg" src="{{ url_for('static', filename='img.png') }}" alt="">
            </div>

            <a class="navbar-brand" href="#">Health Center</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarSupportedContent">
                <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                    <li class="nav-item">
                        <a class="nav-link active" aria-current="page" href="#">Home</a>
                    </li>


                </ul>

            </div>
        </div>
    </nav>




<!-- main form of page -->
<h1 class="mt-4 my-4 text-center text-green">Health Care Center</h1>
<div class="container my-4 mt-4" style="background: black; color: white; border-radius: 15px; padding: 40px;">
    <form action="/predict" method="post">
        <div class="form-group">
            <label for="symptoms">Select Symptoms:</label>
            <input type="text" class="form-control", id="symptoms" name="symptoms" placeholder="type symptoms such as itching, sleeping, aching etc">

        </div>
        <br>
        <button type="button" id="startSpeechRecognition" class="btn btn-primary" style="margin-left:3px;border:1px solid white; border-radius:20px;">
            Start Speech Recognition
        </button>
        <br>

        <!-- Display the transcribed text here -->
<!--        <div name="mysysms" id="transcription"></div>-->

<!--        {% if message %}-->
<!--        <p>{{ message }}</p>-->
<!--        {% endif %}-->
        <br>

        <button type="submit" class="btn btn-danger btn-lg" style="width: 100%; padding: 14px; margin-bottom: 5px;">Predict</button>
    </form>
</div>








<!--{% if predicted_disease %}-->

<!-- Results -->
<h1 class="text-center my-4 mt-4">Our AI System Results</h1>
<div class="container">

    <div class="result-container">
        <!-- Buttons to toggle display -->
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#diseaseModal" style="padding:4px;  margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px; border-radius:5px; background:#F39334;color:black;">Disease</button>
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#descriptionModal" style="padding:4px; margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px; border-radius:5px; background:#268AF3 ;color:black;">Description</button>
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#precautionModal" style="padding:4px; margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px; border-radius:5px; background:#F371F9 ;color:black;">Precaution</button>
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#medicationsModal" style="padding:4px; margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px;border-radius:5px; background:#F8576F ;color:black;">Medications</button>
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#workoutsModal" style="padding:4px; margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px; border-radius:5px; background:#99F741 ;color:black;">Workouts</button>
        <button class="toggle-button" data-bs-toggle="modal" data-bs-target="#dietsModal" style="padding:4px; margin: 5px 40px 5px 0; font-size:20px;font-weight:bold; width:140px; border-radius:5px; background:#E5E23D;color:black;">Diets</button>
    </div>
</div>

<!--{% endif %}-->

<!-- Disease Modal -->
    <div class="modal fade" id="diseaseModal" tabindex="-1" aria-labelledby="diseaseModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;"> <!-- Set header background color inline -->
                    <h5 class="modal-title" id="diseaseModalLabel">Predicted Disease</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body" style="background-color: #modal-body-color;"> <!-- Set modal body background color inline -->
                    <p>{{ predicted_disease }}</p>
                </div>
            </div>
        </div>
    </div>


    <!-- Description Modal -->
    <div class="modal fade" id="descriptionModal" tabindex="-1" aria-labelledby="descriptionModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;">
                    <h5 class="modal-title" id="descriptionModalLabel">Description</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>{{ dis_des }}</p>
                </div>
            </div>
        </div>
    </div>

<!-- Precaution Modal -->
    <div class="modal fade" id="precautionModal" tabindex="-1" aria-labelledby="precautionModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;">
                    <h5 class="modal-title" id="precautionModalLabel">Precaution</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <ul>
                        {% for i in my_precautions %}
                            <li>{{ i }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>




    <!-- Medications Modal -->
    <div class="modal fade" id="medicationsModal" tabindex="-1" aria-labelledby="medicationsModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;">
                    <h5 class="modal-title" id="medicationsModalLabel">Medications</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <ul>
                        {% for i in medications %}
                            <li>{{ i }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- Workouts Modal -->
    <div class="modal fade" id="workoutsModal" tabindex="-1" aria-labelledby="workoutsModalLabel" aria-hidden="true">
        <div class="modal-dialog" >
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;">
                    <h5 class="modal-title" id="workoutsModalLabel">Workouts</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <ul>
                        {% for i in workout %}
                            <li>{{ i }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- Diets Modal -->
    <div class="modal fade" id="dietsModal" tabindex="-1" aria-labelledby="dietsModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header" style="background-color: #020606; color:white;">
                    <h5 class="modal-title" id="dietsModalLabel">Diets</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <ul>
                        {% for i in my_diet %}
                            <li>{{ i }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>



        <script>
    const startSpeechRecognitionButton = document.getElementById('startSpeechRecognition');
    const transcriptionDiv = document.getElementById('transcription');

    startSpeechRecognitionButton.addEventListener('click', startSpeechRecognition);

    function startSpeechRecognition() {
        const recognition = new webkitSpeechRecognition(); // Use webkitSpeechRecognition for compatibility

        recognition.lang = 'en-US'; // Set the language for recognition

        recognition.onresult = function (event) {
            const result = event.results[0][0].transcript;
            transcriptionDiv.textContent = result;
        };

        recognition.onend = function () {
            console.log('Speech recognition ended.');
        };

        recognition.start();
    }
</script>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-HwwvtgBNo3bZJJLYd8oVXjrBZt8cqVSpeBNS5n7C8IVInixGAoxmnlMuBnhbgrkm" crossorigin="anonymous"></script>
</body>

</html>
"""

sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

svc = pickle.load(open('svc.pkl','rb'))



def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]




@app.route("/")
def index():
    return render_template_string(html_template)

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template_string(html_template, message=message)
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template_string(html_template, predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template_string(html_template)






if __name__ == '__main__':

    app.run(debug=True)
