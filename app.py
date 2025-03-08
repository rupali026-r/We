import streamlit as st
import time  # For typing effect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from fuzzywuzzy import process
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import base64

# Ensure consistent language detection
DetectorFactory.seed = 0

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="Carewise AI", layout="wide", page_icon="⚕")

# Custom CSS to set a background image
def set_bg(image_path):
    """Encodes image to Base64 and sets it as background."""
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error setting background image: {e}")

# ✅ Set Background Image (Change the path as needed)
set_bg(r"/Users/snigdhatalasila/Documents/ai_bg.png")

class AISymptomChecker:
    def __init__(self):
        self.knowledge_base = {
        "fever": {
            "common_causes": ["flu", "cold", "infection"],
            "home_remedies": [
                "Drink plenty of fluids (water, herbal teas).",
                "Take a lukewarm bath.",
                "Rest and avoid overexertion."
            ],
            "medicines": ["Paracetamol (acetaminophen)", "Ibuprofen"],
            "recovery_time": "3-5 days",
            "related_diseases": ["Influenza", "COVID-19", "Malaria", "Dengue", "Typhoid"],
            "famous_hospitals": [
                "Apollo Hospitals, Jubilee Hills",
                "https://www.google.com/maps/place/Apollo+Hospitals,+Jubilee+Hills"
            ]
        },
        "cough": {
            "common_causes": ["cold", "flu", "allergies"],
            "home_remedies": [
                "Gargle with warm salt water.",
                "Drink honey and lemon tea.",
                "Use a humidifier to keep the air moist."
            ],
            "medicines": ["Dextromethorphan (cough suppressant)", "Guaifenesin (expectorant)"],
            "recovery_time": "7-10 days",
            "related_diseases": ["Bronchitis", "Pneumonia", "Asthma", "Tuberculosis", "COVID-19"],
            "famous_hospitals": [
                "Yashoda Hospitals, Somajiguda",
                "https://www.google.com/maps/place/Yashoda+Hospitals,+Somajiguda"
            ]
        },
        "headache": {
            "common_causes": ["stress", "migraine", "dehydration"],
            "home_remedies": [
                "Apply a cold or warm compress to your forehead.",
                "Practice relaxation techniques (e.g., deep breathing).",
                "Stay hydrated and avoid caffeine."
            ],
            "medicines": ["Ibuprofen", "Aspirin", "Acetaminophen"],
            "recovery_time": "A few hours to 1-2 days",
            "related_diseases": ["Migraine", "Tension Headache", "Cluster Headache", "Sinusitis", "Hypertension"],
            "famous_hospitals": [
                "Continental Hospitals, Gachibowli",
                "https://www.google.com/maps/place/Continental+Hospitals"
            ]
        },
        "sore throat": {
            "common_causes": ["cold", "flu", "strep throat"],
            "home_remedies": [
                "Gargle with warm salt water.",
                "Drink warm liquids like tea or broth.",
                "Suck on throat lozenges or hard candy."
            ],
            "medicines": ["Paracetamol", "Ibuprofen", "Throat numbing sprays"],
            "recovery_time": "5-7 days",
            "related_diseases": ["Strep Throat", "Tonsillitis", "Pharyngitis", "COVID-19", "Mononucleosis"],
            "famous_hospitals": [
                "KIMS Hospitals, Secunderabad",
                "https://www.google.com/maps/place/KIMS+Hospitals,+Secunderabad"
            ]
        },
        "nausea": {
            "common_causes": ["food poisoning", "motion sickness", "pregnancy", "acid reflux"],
            "home_remedies": [
                "Sip on ginger tea or chew ginger candies.",
                "Eat small, bland meals (e.g., crackers, toast).",
                "Avoid strong smells and fatty foods."
            ],
            "medicines": ["Dimenhydrinate (Dramamine)", "Meclizine (Bonine)", "Antacids (for acid reflux)"],
            "recovery_time": "A few hours to 1-2 days",
            "related_diseases": ["Gastroenteritis", "Pregnancy", "Migraine", "Food Poisoning", "Acid Reflux"],
            "famous_hospitals": [
                "AIG Hospitals, Gachibowli",
                "https://www.google.com/maps/place/AIG+Hospitals"
            ]
        },
        "rashes": {
            "common_causes": ["allergic reaction", "eczema", "contact dermatitis"],
            "home_remedies": [
                "Apply a cold compress to reduce itching.",
                "Use over-the-counter hydrocortisone cream.",
                "Avoid scratching the affected area."
            ],
            "medicines": ["Antihistamines (e.g., Benadryl)", "Hydrocortisone cream", "Calamine lotion"],
            "recovery_time": "3-7 days",
            "related_diseases": ["Eczema", "Psoriasis", "Allergic Reaction", "Measles", "Chickenpox"],
            "famous_hospitals": [
                "Rainbow Children's Hospital, Banjara Hills",
                "https://www.google.com/maps/place/Rainbow+Children's+Hospital"
            ]
        },
        "eye infection": {
            "common_causes": ["bacterial infection", "viral infection", "allergies"],
            "home_remedies": [
                "Apply a warm compress to the affected eye.",
                "Avoid wearing contact lenses until the infection clears.",
                "Keep the eye clean and avoid rubbing it."
            ],
            "medicines": ["Antibiotic eye drops", "Antihistamine eye drops", "Artificial tears"],
            "recovery_time": "5-10 days",
            "related_diseases": ["Conjunctivitis", "Blepharitis", "Keratitis", "Stye", "Uveitis"],
            "famous_hospitals": [
                "LV Prasad Eye Institute, Banjara Hills",
                "https://www.google.com/maps/place/LV+Prasad+Eye+Institute"
            ]
        },
        "ear infection": {
            "common_causes": ["bacterial infection", "viral infection", "fluid buildup"],
            "home_remedies": [
                "Apply a warm compress to the affected ear.",
                "Use over-the-counter pain relievers.",
                "Keep the ear dry and avoid inserting objects."
            ],
            "medicines": ["Antibiotics (e.g., Amoxicillin)", "Pain relievers (e.g., Ibuprofen)", "Ear drops"],
            "recovery_time": "7-14 days",
            "related_diseases": ["Otitis Media", "Otitis Externa", "Swimmer's Ear", "Mastoiditis", "Labyrinthitis"],
            "famous_hospitals": [
                "Global Hospitals, Lakdi-ka-pul",
                "https://www.google.com/maps/place/Global+Hospitals,+Lakdi-ka-pul"
            ]
        },
        "fatigue": {
            "common_causes": ["lack of sleep", "stress", "anemia"],
            "home_remedies": [
                "Get adequate sleep (7-9 hours per night).",
                "Eat a balanced diet rich in vitamins and minerals.",
                "Exercise regularly to boost energy levels."
            ],
            "medicines": ["Multivitamins", "Iron supplements (if anemic)", "Caffeine (in moderation)"],
            "recovery_time": "Varies (depends on cause)",
            "related_diseases": ["Chronic Fatigue Syndrome", "Anemia", "Hypothyroidism", "Depression", "Sleep Apnea"],
            "famous_hospitals": [
                "Care Hospitals, Banjara Hills",
                "https://www.google.com/maps/place/Care+Hospitals,+Banjara+Hills"
            ]
        },
        "body pains": {
            "common_causes": ["overexertion", "flu", "fibromyalgia"],
            "home_remedies": [
                "Apply a warm compress to the affected area.",
                "Take a warm bath with Epsom salts.",
                "Practice gentle stretching or yoga."
            ],
            "medicines": ["Ibuprofen", "Acetaminophen", "Muscle relaxants"],
            "recovery_time": "2-5 days",
            "related_diseases": ["Fibromyalgia", "Arthritis", "Lupus", "Influenza", "Chronic Fatigue Syndrome"],
            "famous_hospitals": [
                "Omega Hospitals, Gachibowli",
                "https://www.google.com/maps/place/Omega+Hospitals"
            ]
        },
        "shortness of breath": {
            "common_causes": ["asthma", "anxiety", "respiratory infection"],
            "home_remedies": [
                "Practice deep breathing exercises.",
                "Avoid triggers like smoke or allergens.",
                "Stay hydrated and rest."
            ],
            "medicines": ["Inhalers (for asthma)", "Antihistamines (for allergies)", "Bronchodilators"],
            "recovery_time": "Varies (depends on cause)",
            "related_diseases": ["Asthma", "COPD", "Pneumonia", "Heart Failure", "Anxiety Disorders"],
            "famous_hospitals": [
                "Star Hospitals, Banjara Hills",
                "https://www.google.com/maps/place/Star+Hospitals"
            ]
        },
        "stomach ache": {
            "common_causes": ["indigestion", "food poisoning", "irritable bowel syndrome"],
            "home_remedies": [
                "Drink peppermint or ginger tea.",
                "Avoid spicy or fatty foods.",
                "Apply a warm compress to the stomach."
            ],
            "medicines": ["Antacids", "Pepto-Bismol", "Loperamide (for diarrhea)"],
            "recovery_time": "1-3 days",
            "related_diseases": ["Gastritis", "Irritable Bowel Syndrome", "Appendicitis", "Ulcerative Colitis", "Food Poisoning"],
            "famous_hospitals": [
                "AIG Hospitals, Gachibowli",
                "https://www.google.com/maps/place/AIG+Hospitals"
            ]
        }
    }

        self.symptoms = list(self.knowledge_base.keys())
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
        self.classifier = MultinomialNB()
        self.train_classifier()

    def train_classifier(self):
        training_data = [
            "I have a high fever", "I am running a fever", "I feel hot and have a fever",
            "I am coughing a lot", "I have a persistent cough", "My cough won't go away",
            "My head hurts", "I have a splitting headache", "I feel a throbbing pain in my head",
            "My throat is sore", "I have a scratchy throat", "It hurts when I swallow",
            "I feel nauseous", "I have an upset stomach", "I feel like throwing up",
            "I have rashes on my skin", "My skin is itchy and red", "I have a rash that won't go away",
            "My eyes are red and itchy", "I have an eye infection", "My eyes are watery and painful",
            "My ears hurt", "I have an ear infection", "I feel pain in my ear",
            "I feel extremely tired", "I have no energy", "I am always fatigued",
            "My body aches all over", "I have muscle pain", "I feel pain in my joints",
            "I am having trouble breathing", "I feel short of breath", "I can't catch my breath",
            "My stomach hurts", "I have a stomach ache", "I feel pain in my abdomen"
        ]
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11]  # Corresponding symptom indices
        X = self.vectorizer.fit_transform(training_data)
        self.classifier.fit(X, labels)

    def predict_symptom(self, user_input):
        input_vector = self.vectorizer.transform([user_input])
        prediction = self.classifier.predict(input_vector)
        return self.symptoms[prediction[0]]

    def find_closest_symptom(self, user_input):
        closest_symptom, score = process.extractOne(user_input, self.symptoms)
        if score > 70:
            return closest_symptom
        return None

    def analyze_sentiment(self, text):
        # Override sentiment analysis for medical symptoms
        negative_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "hurt", "pain", "unwell", "sick",
                             "rash", "itchy", "red", "eye infection", "ear infection", "fatigue", "body pain", "shortness of breath", "stomach ache"]
        if any(keyword in text.lower() for keyword in negative_keywords):
            return "NEGATIVE", -1  # Force negative sentiment for symptoms

        # Use TextBlob for general sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return "POSITIVE", polarity
        elif polarity < 0:
            return "NEGATIVE", polarity
        return "NEUTRAL", polarity

    def suggest_treatment(self, user_input, src_lang):
        if not user_input.strip():
            return "Please enter a valid symptom."

        try:
            # Translate input to English
            user_input_en = GoogleTranslator(source=src_lang, target='en').translate(user_input)

            # Analyze sentiment
            sentiment, score = self.analyze_sentiment(user_input_en)

            # Predict the symptom
            predicted_symptom = self.predict_symptom(user_input_en)

            # Fallback to fuzzy matching
            if predicted_symptom not in self.knowledge_base:
                predicted_symptom = self.find_closest_symptom(user_input_en)

            if predicted_symptom and predicted_symptom in self.knowledge_base:
                result = self.knowledge_base[predicted_symptom]
                
                # Format the response with HTML and CSS
                response = f"""
                <div style="font-size: 20px; font-weight: bold;">Symptom: {predicted_symptom}</div>
                <div style="font-size: 18px; font-weight: bold;">Common Causes:</div>
                <div style="font-size: 16px;">{', '.join(result['common_causes'])}</div>
                <br>
                <div style="font-size: 18px; font-weight: bold;">Home Remedies:</div>
                <ul>
                """
                for remedy in result["home_remedies"]:
                    response += f"<li style='font-size: 16px;'>{remedy}</li>"
                response += """
                </ul>
                <div style="font-size: 18px; font-weight: bold;">Suggested Medicines:</div>
                <ul>
                """
                for medicine in result["medicines"]:
                    response += f"<li style='font-size: 16px;'>{medicine}</li>"
                response += f"""
                </ul>
                <div style="font-size: 18px; font-weight: bold;">Estimated Recovery Time: {result['recovery_time']}</div>
                <br>
                <div style="font-size: 18px; font-weight: bold;">Related Diseases:</div>
                <div style="font-size: 16px;">{', '.join(result['related_diseases'])}</div>
                <br>
                <div style="font-size: 18px; font-weight: bold;">Famous Hospital in Hyderabad:</div>
                <div style="font-size: 16px;">
                    <a href="{result['famous_hospitals'][1]}" target="_blank">{result['famous_hospitals'][0]}</a>
                </div>
                """

                # Add empathetic response based on sentiment
                if sentiment == "NEGATIVE":
                    response = f"<div style='font-size: 16px;'>I'm sorry to hear that you're feeling unwell.</div>{response}"
                elif sentiment == "POSITIVE":
                    response = f"<div style='font-size: 16px;'>Glad to hear you're feeling better.</div>"
                else:
                    response = f"<div style='font-size: 16px;'>Let's find a solution for your symptom.</div>{response}"

                # Translate response back to the user's language
                response_translated = GoogleTranslator(source='en', target=src_lang).translate(response)
                return response_translated
            else:
                return GoogleTranslator(source='en', target=src_lang).translate(
                    "Sorry, I couldn't understand your symptom. Please consult a doctor for further evaluation."
                )
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Function to simulate typing effect
def type_text(text, delay=0.01):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(typed_text, unsafe_allow_html=True)
        time.sleep(delay)
    return typed_text

# Function to provide personalized health recommendations
def personalized_recommendations():
    st.subheader("🌟 Personalized Health Recommendations")
    st.write("Provide details about your frequent symptoms, their frequency, and any allergies to receive personalized health recommendations.")

    # Input fields
    frequent_symptoms = st.multiselect(
        "Select your frequent symptoms:",
        ["Fever", "Cough", "Headache", "Sore Throat", "Nausea", "Rashes", "Eye Infection", "Ear Infection", "Fatigue", "Body Pains", "Shortness of Breath", "Stomach Ache"]
    )
    symptom_frequency = st.selectbox(
        "How often do you experience these symptoms?",
        ["Daily", "Weekly", "Monthly", "Occasionally"]
    )
    allergies = st.text_input("Do you have any allergies? (e.g., peanuts, pollen, dust)")

    if st.button("Get Recommendations"):
        if not frequent_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Generate personalized recommendations
            recommendations = generate_recommendations(frequent_symptoms, symptom_frequency, allergies)
            st.success("Here are your personalized health recommendations:")
            st.markdown(recommendations, unsafe_allow_html=True)

# Function to generate personalized recommendations
def generate_recommendations(frequent_symptoms, symptom_frequency, allergies):
    recommendations = "<div style='font-size: 16px;'>"
    recommendations += "<h3>Personalized Health Recommendations</h3>"
    recommendations += f"<p>Based on your frequent symptoms (<strong>{', '.join(frequent_symptoms)}</strong>) and their frequency (<strong>{symptom_frequency}</strong>), here are some recommendations:</p>"
    
    # Combine symptom-specific and allergen-specific recommendations
    for symptom in frequent_symptoms:
        if symptom.lower() in ["fever", "cough", "headache", "sore throat", "nausea", "rashes", "eye infection", "ear infection", "fatigue", "body pains", "shortness of breath", "stomach ache"]:
            recommendations += f"<h4>{symptom}:</h4>"
            recommendations += "<ul>"
            recommendations += f"<li>{symptom_specific_tips(symptom.lower())}</li>"
            if allergies:
                recommendations += f"<li>{allergen_specific_tips(symptom.lower(), allergies)}</li>"
            recommendations += "</ul>"

    recommendations += "</div>"
    return recommendations

# Function to provide symptom-specific tips
def symptom_specific_tips(symptom):
    tips = {
        "fever": "Drink plenty of fluids and rest.",
        "cough": "Use a humidifier and avoid irritants like smoke.",
        "headache": "Practice relaxation techniques and avoid caffeine.",
        "sore throat": "Gargle with warm salt water and stay hydrated.",
        "nausea": "Eat small, bland meals and avoid strong smells.",
        "rashes": "Apply a cold compress and avoid scratching.",
        "eye infection": "Avoid rubbing your eyes and use prescribed eye drops.",
        "ear infection": "Keep your ear dry and use pain relievers.",
        "fatigue": "Get adequate sleep and eat a balanced diet.",
        "body pains": "Apply a warm compress and practice gentle stretching.",
        "shortness of breath": "Practice deep breathing exercises and avoid triggers.",
        "stomach ache": "Avoid spicy or fatty foods and drink herbal teas."
    }
    return tips.get(symptom, "Consult a doctor for specific advice.")

# Function to provide allergen-specific tips
def allergen_specific_tips(symptom, allergies):
    tips = {
        "fever": f"Avoid allergens like {allergies} to prevent worsening symptoms.",
        "cough": f"Avoid allergens like {allergies} and use an air purifier.",
        "headache": f"Stay away from allergens like {allergies} and strong perfumes.",
        "sore throat": f"Avoid allergens like {allergies} and keep your environment clean.",
        "nausea": f"Avoid allergens like {allergies} in your diet.",
        "rashes": f"Avoid allergens like {allergies} and certain fabrics or chemicals.",
        "eye infection": f"Stay away from allergens like {allergies} and keep your eyes clean.",
        "ear infection": f"Avoid allergens like {allergies} and keep your ear dry.",
        "fatigue": f"Avoid allergens like {allergies} and maintain a clean living space.",
        "body pains": f"Avoid allergens like {allergies} and certain foods.",
        "shortness of breath": f"Avoid allergens like {allergies} and practice deep breathing.",
        "stomach ache": f"Avoid allergens like {allergies} in your diet."
    }
    return tips.get(symptom, f"Avoid allergens like {allergies} to manage symptoms.")

# Personalized Suggestions Page
def personalized_suggestions():
    st.subheader("🌟 Personalized Suggestions")
    st.write("Explore blogs, join the community, and get personalized health recommendations.")

    # Tabs for Blogs and Community Support
    tab1, tab2 = st.tabs(["📚 Blogs", "💬 Community Support"])

    with tab1:
        st.header("📚 Health Blogs")
        st.write("Read informative articles on health, wellness, and symptom management.")

        # Sample Blog Data
        blogs = [
            {"title": "5 Tips to Boost Your Immune System", "content": "Learn how to strengthen your immune system naturally with these simple tips.", "author": "Dr. John Doe"},
            {"title": "Understanding Common Cold vs. Flu", "content": "Discover the differences between a common cold and the flu, and how to treat them.", "author": "Dr. Jane Smith"},
            {"title": "Managing Stress for Better Health", "content": "Explore effective techniques to manage stress and improve your overall well-being.", "author": "Dr. Emily Brown"},
        ]

        # Display Blogs
        for blog in blogs:
            with st.expander(f"{blog['title']}** by {blog['author']}"):
                st.write(blog["content"])
                if st.button("Read More", key=blog["title"]):
                    st.write(f"Full article on '{blog['title']}' is coming soon!")

        # Search and Filter Blogs
        search_query = st.text_input("Search blogs by keyword:")
        if search_query:
            filtered_blogs = [blog for blog in blogs if search_query.lower() in blog["title"].lower() or search_query.lower() in blog["content"].lower()]
            if filtered_blogs:
                st.write(f"Search Results for '{search_query}':")
                for blog in filtered_blogs:
                    st.write(f"- {blog['title']} by {blog['author']}")
            else:
                st.warning("No blogs found matching your search.")

    with tab2:
        st.header("💬 Community Support")
        st.write("Join the community to share your experiences, ask questions, and support others.")

        # Sample Community Posts
        if 'community_posts' not in st.session_state:
            st.session_state.community_posts = [
                {"user": "User123", "post": "Has anyone found relief for chronic migraines?", "replies": ["Try acupuncture, it worked for me! - User456"]},
                {"user": "User789", "post": "Best home remedies for a sore throat?", "replies": ["Gargle with warm salt water! - User101"]},
            ]

        # Display Community Posts
        st.write("Recent Posts:")
        for post in st.session_state.community_posts:
            with st.expander(f"{post['user']}: {post['post']}"):
                st.write("Replies:")
                for reply in post["replies"]:
                    st.write(f"- {reply}")

        # Add a New Post
        st.write("Create a New Post:")
        new_post = st.text_area("Share your experience or ask a question:")
        if st.button("Post"):
            if new_post.strip():
                st.session_state.community_posts.append({"user": "You", "post": new_post, "replies": []})
                st.success("Your post has been shared!")
            else:
                st.warning("Please write something to post.")

        # Reply to a Post
        st.write("Reply to a Post:")
        selected_post = st.selectbox("Select a post to reply to:", [post["post"] for post in st.session_state.community_posts])
        reply_text = st.text_area("Write your reply:")
        if st.button("Reply"):
            if reply_text.strip():
                for post in st.session_state.community_posts:
                    if post["post"] == selected_post:
                        post["replies"].append(f"{reply_text} - You")
                        st.success("Your reply has been posted!")
                        break
            else:
                st.warning("Please write something to reply.")

# Streamlit UI
st.title("⚕ Carewise AI")
st.subheader("Revolutionizing Healthcare with Artificial Intelligence")
st.write("Carewise AI leverages cutting-edge technology to enhance patient care, streamline diagnoses, and improve healthcare outcomes. Join us in building a healthier future powered by intelligent solutions.")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Home", "AI Symptom Checker", "Personalized Recommendations", "Blogs and Community Support", "About Us", "Contact"])

if page == "Home":
    st.subheader("🏡 Welcome to Carewise AI")
    st.write("We provide cutting-edge AI-powered solutions for healthcare.")
    st.write("""
    ### Our Mission
    To revolutionize healthcare by making it more accessible, efficient, and personalized through the power of artificial intelligence.

    ### Our Vision
    A world where everyone has access to high-quality healthcare, powered by intelligent systems that predict, prevent, and treat illnesses effectively.
    """)

elif page == "AI Symptom Checker":
    st.subheader("🤖 AI Symptom Checker")
    st.write("Describe your symptoms, and our AI will provide you with possible causes, home remedies, and suggested medicines.")
    
    # Ask the user for their preferred language
    language = st.selectbox("Select your language", ["en", "hi", "te", "da", "no", "sv", "de", "fr", "es", "it"])
    
    user_input = st.text_area("Describe your symptoms:")
    if st.button("Analyze"):
        symptom_checker = AISymptomChecker()
        result = symptom_checker.suggest_treatment(user_input, language)
        
        # Display the result with typing effect
        type_text(result)

elif page == "Personalized Recommendations":
    personalized_recommendations()

elif page == "Blogs and Community Support":
    personalized_suggestions()

elif page == "About Us":
    st.subheader("📌 About Us")
    st.write("We are a team dedicated to transforming healthcare through AI.")
    st.write("""
    ### Who We Are
    We are a group of passionate engineers, data scientists, and healthcare professionals committed to improving global health outcomes.

    ### What We Do
    - Develop AI-powered diagnostic tools.
    - Provide personalized treatment recommendations.
    - Collaborate with healthcare providers to integrate AI into clinical workflows.
    """)

elif page == "Contact":
    st.subheader("📞 Contact Us")
    st.write("We'd love to hear from you! Reach out to us for any inquiries or collaborations.")
    st.write("""
    ### Contact Information
    📧 Email: support@carewise.ai  
    📞 Phone: +1 (123) 456-7890  
    📍 Location: Hyderabad, India  

    ### Social Media
    - [LinkedIn](#)
    - [Twitter](#)
    - [Facebook](#)
    """)

st.markdown("---")
st.markdown("© 2025 Carewise AI | All Rights Reserved")
