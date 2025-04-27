import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

prompt_template = """
As a highly knowledgeable restaurant assistant, your role is to accurately interpret restaurant-related queries and 
provide responses using our specialized restaurant database. Follow these directives to ensure optimal user interactions:
1. Precision in Answers: Respond solely with information directly relevant to the user's query from our restaurant database. 
    Refrain from making assumptions or adding extraneous details.
2. Topic Relevance: Limit your expertise to specific restaurant-related areas:
    - Restaurant Menus
    - Popular Dishes
    - Restaurant Locations
    - Contact Information
    - Cuisine Types
3. Handling Off-topic Queries: For questions unrelated to restaurants (e.g., general knowledge questions like "Why is the sky blue?"), 
    politely inform the user that the query is outside the chatbot’s scope and suggest redirecting to restaurant-related inquiries.
4. Promoting Restaurant Awareness: Craft responses that emphasize the unique offerings of restaurants, including popular dishes 
    and specialties.
5. Contextual Accuracy: Ensure responses are directly related to the restaurant query, utilizing only pertinent 
    information from our database.
6. Relevance Check: If a query does not align with our restaurant database, guide the user to refine their 
    question or politely decline to provide an answer.
7. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and 
    relevance to each user query.
8. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
    delivering clear, concise, and direct answers.
9. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "RestaurantBot" in responses.
10. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each 
    sentence should be unique and contribute to the overall message without redundancy.

Restaurant Query:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

st.markdown("""
    <h3 style='text-align: left; color: black; padding-top: 35px; border-bottom: 3px solid red;'>
        Discover the Best Restaurants and Dishes 🍽️
    </h3>""", unsafe_allow_html=True)


side_bar_message = """
Hi! 👋 I'm here to help you with your restaurant choices. What would you like to know or explore?
\nHere are some areas you might be interested in:
1. **Restaurant Menus** 📜
2. **Popular Dishes** 🍲
3. **Restaurant Locations** 📍
4. **Cuisine Types** 🍴

Feel free to ask me anything about restaurants!
"""

with st.sidebar:
    st.title('🤖RestaurantBot: Your AI Dining Companion')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I'm your RestaurantBot 🤖 
    Here are some questions you might ask me:\n
     🍽️ What are the popular dishes at Foodbay?\n
     🍽️ Can you suggest a good vegetarian restaurant in Roorkee?\n
     🍽️ What is the contact number for Pizza Hut in Roorkee?\n
     🍽️ What cuisines are available at Desi Tadka?\n
     🍽️ Where is Baap of Rolls located?
"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the latest restaurant information for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)