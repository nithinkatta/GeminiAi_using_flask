from flask import Flask, render_template, request
import google.generativeai as genai

app = Flask(__name__)


GOOGLE_API_KEY = 'AIzaSyBaj3yH1P6dNttlssEt_aAptFpLed4PoKw'  
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        s = request.form['input_text'] # ex:- diabetes, hypertension, etc
    
        text_input = "give me diet of a " + s + " in json format consisting of names only without any extra description "
        if s == "":
            return render_template('index.html', generated_text="Please enter a valid input")
        response = model.generate_content(text_input)
        generated_text = response.text
        return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
