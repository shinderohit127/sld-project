from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
import google.generativeai as genai
import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin
cred = credentials.Certificate('firebase-service-account.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-pro')

# SLD Categories and thresholds
SLD_CATEGORIES = {
    'dyslexia': {'threshold': 0.65, 'description': 'Reading difficulties'},
    'dyscalculia': {'threshold': 0.65, 'description': 'Mathematical difficulties'},
    'dysgraphia': {'threshold': 0.65, 'description': 'Writing difficulties'},
    'dyspraxia': {'threshold': 0.65, 'description': 'Motor coordination difficulties'}
}

# Question categories mapping (from research paper)
PARENT_QUESTION_CATEGORIES = {
    'reading': list(range(1, 11)),
    'writing': list(range(11, 21)),
    'math': list(range(21, 31)),
    'attention': list(range(31, 41)),
    'memory': list(range(41, 51)),
    'motor_skills': list(range(51, 61)),
    'social': list(range(61, 66))
}

TEACHER_QUESTION_CATEGORIES = {
    'reading': list(range(1, 10)),
    'writing': list(range(10, 19)),
    'math': list(range(19, 28)),
    'attention': list(range(28, 37)),
    'behavior': list(range(37, 47)),
    'social': list(range(47, 53))
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    """Register a new user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'parent')
        name = data.get('name')
        
        # Create Firebase user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )
        
        # Store additional user data in Firestore
        user_data = {
            'uid': user.uid,
            'email': email,
            'role': role,
            'profile': {
                'name': name,
                'phone': data.get('phone', '')
            },
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        
        db.collection('users').document(user.uid).set(user_data)
        
        return jsonify({
            'success': True,
            'uid': user.uid,
            'message': 'User registered successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/children', methods=['POST'])
def add_child():
    """Add a new child profile"""
    try:
        data = request.json
        auth_token = request.headers.get('Authorization')
        
        # Verify token
        decoded_token = auth.verify_id_token(auth_token.replace('Bearer ', ''))
        parent_uid = decoded_token['uid']
        
        child_data = {
            'parentId': parent_uid,
            'name': data.get('name'),
            'age': data.get('age'),
            'grade': data.get('grade'),
            'dateOfBirth': data.get('dateOfBirth'),
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        
        # Validate age (8-12 years as per research)
        if not (8 <= child_data['age'] <= 12):
            return jsonify({
                'success': False,
                'error': 'Child age must be between 8 and 12 years'
            }), 400
        
        doc_ref = db.collection('children').add(child_data)
        child_id = doc_ref[1].id
        
        return jsonify({
            'success': True,
            'childId': child_id,
            'message': 'Child profile created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/assessment/create', methods=['POST'])
def create_assessment():
    """Create a new assessment"""
    try:
        data = request.json
        auth_token = request.headers.get('Authorization')
        
        decoded_token = auth.verify_id_token(auth_token.replace('Bearer ', ''))
        user_uid = decoded_token['uid']
        
        assessment_data = {
            'childId': data.get('childId'),
            'createdBy': user_uid,
            'status': 'pending',
            'parentResponses': {},
            'teacherResponses': {},
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = db.collection('assessments').add(assessment_data)
        assessment_id = doc_ref[1].id
        
        return jsonify({
            'success': True,
            'assessmentId': assessment_id
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/assessment/<assessment_id>/submit', methods=['POST'])
def submit_responses(assessment_id):
    """Submit questionnaire responses"""
    try:
        data = request.json
        auth_token = request.headers.get('Authorization')
        
        decoded_token = auth.verify_id_token(auth_token.replace('Bearer ', ''))
        user = db.collection('users').document(decoded_token['uid']).get().to_dict()
        
        role = user.get('role')
        responses = data.get('responses')
        
        # Update assessment with responses
        assessment_ref = db.collection('assessments').document(assessment_id)
        
        if role == 'parent':
            assessment_ref.update({
                'parentResponses': responses,
                'status': 'parent_completed',
                'updatedAt': firestore.SERVER_TIMESTAMP
            })
        elif role == 'teacher':
            assessment_ref.update({
                'teacherResponses': responses,
                'updatedAt': firestore.SERVER_TIMESTAMP
            })
            
            # Check if both parent and teacher completed
            assessment = assessment_ref.get().to_dict()
            if assessment.get('parentResponses') and assessment.get('teacherResponses'):
                assessment_ref.update({'status': 'ready_for_analysis'})
        
        return jsonify({
            'success': True,
            'message': f'{role.capitalize()} responses submitted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def calculate_sld_probabilities(parent_responses, teacher_responses):
    """
    Calculate SLD probabilities based on research paper methodology
    Using simplified logistic regression approach
    """
    # Combine responses (117 total: 65 parent + 52 teacher)
    all_responses = list(parent_responses.values()) + list(teacher_responses.values())
    
    # Category-wise analysis
    probabilities = {
        'dyslexia': 0.0,
        'dyscalculia': 0.0,
        'dysgraphia': 0.0,
        'dyspraxia': 0.0
    }
    
    # Dyslexia - related to reading questions
    reading_parent = sum([parent_responses.get(f'q{i}', 0) for i in PARENT_QUESTION_CATEGORIES['reading']])
    reading_teacher = sum([teacher_responses.get(f'q{i}', 0) for i in TEACHER_QUESTION_CATEGORIES['reading']])
    probabilities['dyslexia'] = min((reading_parent + reading_teacher) / 20, 1.0)
    
    # Dyscalculia - related to math questions
    math_parent = sum([parent_responses.get(f'q{i}', 0) for i in PARENT_QUESTION_CATEGORIES['math']])
    math_teacher = sum([teacher_responses.get(f'q{i}', 0) for i in TEACHER_QUESTION_CATEGORIES['math']])
    probabilities['dyscalculia'] = min((math_parent + math_teacher) / 20, 1.0)
    
    # Dysgraphia - related to writing questions
    writing_parent = sum([parent_responses.get(f'q{i}', 0) for i in PARENT_QUESTION_CATEGORIES['writing']])
    writing_teacher = sum([teacher_responses.get(f'q{i}', 0) for i in TEACHER_QUESTION_CATEGORIES['writing']])
    probabilities['dysgraphia'] = min((writing_parent + writing_teacher) / 20, 1.0)
    
    # Dyspraxia - related to motor skills
    motor_parent = sum([parent_responses.get(f'q{i}', 0) for i in PARENT_QUESTION_CATEGORIES['motor_skills']])
    probabilities['dyspraxia'] = min(motor_parent / 10, 1.0)
    
    return probabilities

@app.route('/api/assessment/<assessment_id>/analyze', methods=['POST'])
def analyze_assessment(assessment_id):
    """Analyze assessment using Gemini AI and generate results"""
    try:
        assessment_ref = db.collection('assessments').document(assessment_id)
        assessment = assessment_ref.get().to_dict()
        
        if not assessment:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        
        parent_responses = assessment.get('parentResponses', {})
        teacher_responses = assessment.get('teacherResponses', {})
        
        if not parent_responses or not teacher_responses:
            return jsonify({
                'success': False,
                'error': 'Both parent and teacher responses required'
            }), 400
        
        # Calculate probabilities
        probabilities = calculate_sld_probabilities(parent_responses, teacher_responses)
        
        # Determine overall risk level
        max_prob = max(probabilities.values())
        if max_prob >= 0.7:
            risk_level = 'high'
        elif max_prob >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate AI-powered recommendations using Gemini
        prompt = f"""
        You are an educational psychologist assistant. Based on the following screening results 
        for a child aged 8-12, provide brief, actionable recommendations:
        
        Probabilities:
        - Dyslexia (Reading): {probabilities['dyslexia']:.2%}
        - Dyscalculia (Math): {probabilities['dyscalculia']:.2%}
        - Dysgraphia (Writing): {probabilities['dysgraphia']:.2%}
        - Dyspraxia (Motor): {probabilities['dyspraxia']:.2%}
        
        Overall Risk Level: {risk_level.upper()}
        
        Provide 3-5 specific, practical recommendations for parents and teachers.
        Keep each recommendation to 1-2 sentences.
        """
        
        response = model.generate_content(prompt)
        recommendations = response.text.split('\n')
        recommendations = [r.strip() for r in recommendations if r.strip() and len(r.strip()) > 10]
        
        # Store results
        results = {
            'overallRisk': risk_level,
            'probabilities': probabilities,
            'recommendations': recommendations,
            'generatedAt': datetime.now().isoformat(),
            'requiresProfessionalAssessment': max_prob >= 0.65
        }
        
        assessment_ref.update({
            'results': results,
            'status': 'analyzed',
            'analyzedAt': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/assessment/<assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
    """Get assessment details"""
    try:
        auth_token = request.headers.get('Authorization')
        decoded_token = auth.verify_id_token(auth_token.replace('Bearer ', ''))
        
        assessment = db.collection('assessments').document(assessment_id).get()
        
        if not assessment.exists:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        
        return jsonify({
            'success': True,
            'assessment': assessment.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/assessments/child/<child_id>', methods=['GET'])
def get_child_assessments(child_id):
    """Get all assessments for a child"""
    try:
        auth_token = request.headers.get('Authorization')
        decoded_token = auth.verify_id_token(auth_token.replace('Bearer ', ''))
        
        assessments = db.collection('assessments')\
            .where('childId', '==', child_id)\
            .order_by('createdAt', direction=firestore.Query.DESCENDING)\
            .stream()
        
        results = [{'id': doc.id, **doc.to_dict()} for doc in assessments]
        
        return jsonify({
            'success': True,
            'assessments': results
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)