
import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(
    page_title="AI Project Idea Generator",
    page_icon="💡",
    layout="wide"
)
st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.idea-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

PROJECTS = [
    {
        "title": "AI Resume Skill Gap Analyzer",
        "domain": "Career",
        "level": "Beginner",
        "keywords": "resume job skills NLP career machine learning",
        "description": "Analyzes resumes and identifies missing skills required for a target job."
    },
    {
        "title": "Smart Campus Safety Agent",
        "domain": "Education",
        "level": "Intermediate",
        "keywords": "campus safety emergency AI agent security students",
        "description": "An AI agent that helps identify campus emergencies and recommends actions."
    },
    {
        "title": "AI Study Planner",
        "domain": "Education",
        "level": "Beginner",
        "keywords": "students education study timetable exams learning AI",
        "description": "Creates personalized study schedules based on subjects and deadlines."
    },
    {
        "title": "Fake News Detection System",
        "domain": "Social Media",
        "level": "Intermediate",
        "keywords": "fake news NLP classification social media machine learning",
        "description": "Uses NLP and machine learning to classify news as potentially fake or genuine."
    },
    {
        "title": "AI Code Bug Explainer",
        "domain": "Software",
        "level": "Intermediate",
        "keywords": "coding programming debugging errors software AI",
        "description": "Explains programming errors and suggests debugging approaches."
    },
    {
        "title": "Green AI Energy Advisor",
        "domain": "Environment",
        "level": "Advanced",
        "keywords": "green AI environment energy sustainability carbon emissions",
        "description": "Provides recommendations for reducing energy consumption of AI workloads."
    },
    {
        "title": "AI Public Transport Predictor",
        "domain": "Transportation",
        "level": "Advanced",
        "keywords": "bus transportation passengers traffic prediction machine learning",
        "description": "Predicts passenger demand for public transportation."
    },
    {
        "title": "Bharatanatyam Learning Assistant",
        "domain": "Culture",
        "level": "Intermediate",
        "keywords": "Bharatanatyam dance Indian culture learning AI",
        "description": "An AI assistant for learning Bharatanatyam terminology and theory."
    },
    {
        "title": "AI Project Documentation Generator",
        "domain": "Software",
        "level": "Beginner",
        "keywords": "project report documentation abstract objectives methodology AI",
        "description": "Generates project documentation from a project idea."
    },
    {
        "title": "AI Vocabulary Coach",
        "domain": "Education",
        "level": "Beginner",
        "keywords": "English vocabulary NLP words learning education",
        "description": "Creates personalized vocabulary exercises."
    }
]

# -----------------------------
# Gemini Function
# -----------------------------
def generate_with_gemini(prompt):

    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None, "Gemini API key not found."

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text, None

    except Exception as e:
        return None, str(e)


# -----------------------------
# Extract JSON
# -----------------------------
def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    start = text.find("[")

    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Invalid AI response.")

    return json.loads(text[start:end + 1])


# -----------------------------
# Recommendation Engine
# -----------------------------
def recommend_projects(
    query,
    level,
    domain
):

    df = pd.DataFrame(PROJECTS)

    documents = (
        df["keywords"]
        + " "
        + df["description"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        documents.tolist() + [query]
    )

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )[0]

    df["score"] = similarity

    # Skill level boost
    df.loc[
        df["level"] == level,
        "score"
    ] += 0.10

    # Domain boost
    if domain != "Any":

        df.loc[
            df["domain"] == domain,
            "score"
        ] += 0.10

    return df.sort_values(
        "score",
        ascending=False
    )


# -----------------------------
# Generate Prompt
# -----------------------------
def create_prompt(
    interest,
    domain,
    level,
    skills,
    duration,
    team_size,
    novelty,
    constraints,
    count
):

    return f"""
You are an expert AI project mentor.

Generate {count} unique and practical AI/ML project ideas.

Student information:

Interest:
{interest}

Domain:
{domain}

Skill Level:
{level}

Skills:
{skills}

Project Duration:
{duration}

Team Size:
{team_size}

Desired Novelty:
{novelty}/10

Constraints:
{constraints}

The projects should:

1. Be realistic for college students.
2. Use AI or machine learning meaningfully.
3. Be possible to implement using Python.
4. Have a Streamlit interface.
5. Be suitable for a college project.
6. Have some unique feature.
7. Suggest a dataset or data source.

Return ONLY JSON.

Each project must contain:

title
one_line_pitch
problem_statement
solution
ai_component
technologies
dataset
difficulty
estimated_weeks
novelty_score
features
architecture
future_scope

Example format:

[
  {{
    "title": "Example Project",
    "one_line_pitch": "Short description",
    "problem_statement": "Problem",
    "solution": "Solution",
    "ai_component": "AI component",
    "technologies": ["Python", "Streamlit"],
    "dataset": "Dataset",
    "difficulty": "Intermediate",
    "estimated_weeks": 6,
    "novelty_score": 8,
    "features": ["Feature 1", "Feature 2"],
    "architecture": "Architecture description",
    "future_scope": "Future improvements"
  }}
]
"""


# -----------------------------
# Session State
# -----------------------------
if "ideas" not in st.session_state:
    st.session_state.ideas = []


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">💡 AI Project Idea Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate unique AI/ML project ideas based on your skills and interests'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Project Configuration")

interest = st.sidebar.text_input(
    "Your Interest",
    "Artificial Intelligence and Machine Learning"
)

domain = st.sidebar.selectbox(
    "Preferred Domain",
    [
        "Any",
        "Education",
        "Healthcare",
        "Environment",
        "Finance",
        "Transportation",
        "Career",
        "Software",
        "Culture",
        "Cybersecurity",
        "Social Media"
    ]
)

level = st.sidebar.selectbox(
    "Skill Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

skills = st.sidebar.text_input(
    "Your Skills",
    "Python, HTML, CSS, Machine Learning"
)

duration = st.sidebar.selectbox(
    "Project Duration",
    [
        "2-4 weeks",
        "1-2 months",
        "2-4 months",
        "Semester Project"
    ]
)

team_size = st.sidebar.number_input(
    "Team Size",
    min_value=1,
    max_value=10,
    value=3
)

novelty = st.sidebar.slider(
    "Desired Novelty",
    1,
    10,
    8
)

constraints = st.sidebar.text_area(
    "Project Requirements",
    "Should be easy to demonstrate in VS Code and Streamlit."
)

count = st.sidebar.slider(
    "Number of Ideas",
    3,
    10,
    5
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "🚀 Generate Ideas",
        "🧠 Recommendation Engine",
        "📚 Project Guide"
    ]
)


# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.header("🚀 Generate AI Project Ideas")

    st.write(
        "Provide your requirements in the sidebar and generate "
        "personalized project ideas."
    )

    if st.button(
        "✨ Generate Ideas",
        type="primary",
        use_container_width=True
    ):

        prompt = create_prompt(
            interest,
            domain,
            level,
            skills,
            duration,
            team_size,
            novelty,
            constraints,
            count
        )

        with st.spinner(
            "AI is generating project ideas..."
        ):

            response, error = generate_with_gemini(
                prompt
            )

        if error:

            st.error(
                f"Gemini Error: {error}"
            )

        else:

            try:

                ideas = extract_json(
                    response
                )

                st.session_state.ideas = ideas

                st.success(
                    f"{len(ideas)} project ideas generated!"
                )

            except Exception as e:

                st.error(
                    f"Could not process AI response: {e}"
                )

    # Display ideas
    if st.session_state.ideas:

        st.subheader("💡 Generated Project Ideas")

        for index, idea in enumerate(
            st.session_state.ideas,
            start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"### {index}. {idea.get('title', 'Project')}"
                )

                st.write(
                    idea.get(
                        "one_line_pitch",
                        ""
                    )
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Difficulty",
                    idea.get(
                        "difficulty",
                        "N/A"
                    )
                )

                col2.metric(
                    "Duration",
                    str(
                        idea.get(
                            "estimated_weeks",
                            "N/A"
                        )
                    ) + " weeks"
                )

                col3.metric(
                    "Novelty",
                    str(
                        idea.get(
                            "novelty_score",
                            "N/A"
                        )
                    ) + "/10"
                )

                st.markdown(
                    "**Problem Statement**"
                )

                st.write(
                    idea.get(
                        "problem_statement",
                        ""
                    )
                )

                st.markdown(
                    "**AI Component**"
                )

                st.write(
                    idea.get(
                        "ai_component",
                        ""
                    )
                )

                st.markdown(
                    "**Technologies**"
                )

                technologies = idea.get(
                    "technologies",
                    []
                )

                st.write(
                    ", ".join(
                        technologies
                    )
                    if isinstance(
                        technologies,
                        list
                    )
                    else technologies
                )

                with st.expander(
                    "📋 View Complete Project Plan"
                ):

                    st.markdown(
                        "**Solution**"
                    )

                    st.write(
                        idea.get(
                            "solution",
                            ""
                        )
                    )

                    st.markdown(
                        "**Dataset / Data Source**"
                    )

                    st.write(
                        idea.get(
                            "dataset",
                            ""
                        )
                    )

                    st.markdown(
                        "**Features**"
                    )

                    features = idea.get(
                        "features",
                        []
                    )

                    if isinstance(
                        features,
                        list
                    ):

                        for feature in features:

                            st.write(
                                f"• {feature}"
                            )

                    else:

                        st.write(features)

                    st.markdown(
                        "**Architecture**"
                    )

                    st.write(
                        idea.get(
                            "architecture",
                            ""
                        )
                    )

                    st.markdown(
                        "**Future Scope**"
                    )

                    st.write(
                        idea.get(
                            "future_scope",
                            ""
                        )
                    )


# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.header(
        "🧠 ML-Based Project Recommendation"
    )

    query = st.text_area(
        "Describe the project you want",
        f"I want a {level.lower()} AI project "
        f"in {domain.lower()} using {skills}."
    )

    if st.button(
        "🔎 Find Matching Projects",
        use_container_width=True
    ):

        results = recommend_projects(
            query,
            level,
            domain
        )

        st.subheader(
            "Recommended Projects"
        )

        for _, row in results.head(5).iterrows():

            score = min(
                row["score"] * 100,
                100
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {row['title']}"
                )

                st.write(
                    row["description"]
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Match Score",
                    f"{score:.1f}%"
                )

                col2.metric(
                    "Domain",
                    row["domain"]
                )

                col3.metric(
                    "Level",
                    row["level"]
                )

                st.write(
                    "**Keywords:** "
                    + row["keywords"]
                )

    st.info(
        "The recommendation engine uses "
        "TF-IDF and cosine similarity."
    )


# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.header(
        "📚 How to Build Your Selected Project"
    )

    st.markdown("""
### 1. Problem Definition

Identify a real-world problem and define the target users.

### 2. Data Collection

Collect data from public datasets, APIs, databases or custom datasets.

### 3. Data Preprocessing

Clean the data and prepare it for machine learning.

### 4. AI/ML Model

Choose an appropriate technique such as:

- Classification
- Regression
- Clustering
- NLP
- Computer Vision
- Recommendation Systems
- Generative AI

### 5. Model Evaluation

Use appropriate metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- MAE
- RMSE

### 6. Streamlit Application

Create an interactive interface where users can enter data and receive AI-generated results.

### 7. Testing

Test normal inputs, incorrect inputs and edge cases.

### 8. Deployment

Deploy the Streamlit application using Streamlit Community Cloud or another hosting platform.
""")

    if st.session_state.ideas:

        json_data = json.dumps(
            st.session_state.ideas,
            indent=4
        )

        st.download_button(
            "⬇️ Download Ideas as JSON",
            json_data,
            file_name="ai_project_ideas.json",
            mime="application/json"
        )

st.divider()

st.caption(
    "AI Project Idea Generator | Python 3.11 | "
    "Streamlit | Scikit-learn | Gemini"
)