import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')  # Use environment variable or fallback
    DATABASE = 'queries.db'
    THRESHOLD = 0.3  # Similarity threshold for chatbot responses
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
def init_db():
    with sqlite3.connect(Config.DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS queries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            question TEXT,
                            email TEXT,
                            escalated INTEGER DEFAULT 0,
                            resolution TEXT,
                            flag TEXT)''')
        conn.commit()

init_db()

# Load CSV dataset for chatbot responses
df = pd.read_csv(r"D:\myproject\chatbot capstone\4\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['response'])

# Database connection utility
def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['user_query']
    user_email = request.form['email']

    # Perform TF-IDF vectorization and similarity matching
    user_query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_query_vec, X)
    most_similar_idx = similarities.argmax()

    # Escalate query if similarity is below threshold
    if similarities[0][most_similar_idx] < Config.THRESHOLD:
        answer = "Query not found. Your request has been escalated to the support team."
        with sqlite3.connect(Config.DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO queries (question, email, escalated) VALUES (?, ?, ?)',
                           (user_query, user_email, 1))
            conn.commit()
    else:
        answer = df.iloc[most_similar_idx]['response']

    return render_template('index.html', response=answer)

@app.route('/staff')
def staff():
    # Fetch escalated queries
    conn = get_db_connection()
    queries = conn.execute('SELECT * FROM queries WHERE escalated = 1').fetchall()
    conn.close()
    return render_template('staff.html', queries=queries)

@app.route('/resolve_query/<int:query_id>', methods=['POST'])
def resolve_query(query_id):
    resolution = request.form['resolution']
    flag = request.form['flag']

    # Update query resolution
    conn = get_db_connection()
    conn.execute('UPDATE queries SET escalated = 0, resolution = ?, flag = ? WHERE id = ?',
                 (resolution, flag, query_id))
    conn.commit()
    conn.close()

    return redirect(url_for('staff'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
