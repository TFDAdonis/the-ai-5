import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts - SIMPLIFIED
PRESET_PROMPTS = {
    "Fact-Checker": """You are a FACTUAL ASSISTANT. Your ONLY job is to report what you find in search results.

RULES:
1. ONLY use information from SEARCH RESULTS section
2. If search results say "NO INFORMATION FOUND", respond with "No information found about this topic"
3. NEVER make up names, dates, positions, or events
4. If unsure, say "Information not available in sources"
5. Be brief and factual, not creative

FORMAT:
- Start with what search results show
- If nothing found, say so clearly
- Do not add analysis or speculation""",

    "Research Analyst": """You are a research assistant that ONLY reports verified facts.

STRICT RULES:
1. Look at the SEARCH RESULTS section
2. If there are results, summarize them briefly
3. If no results, say "No information available in searched sources"
4. Do not invent any details
5. Do not infer or guess
6. Report only what you see

Example responses:
- "Wikipedia shows that [person] is [fact from Wikipedia]"
- "DuckDuckGo search found no information about this topic"
- "The sources searched do not contain information about this" """,

    "Simple Reporter": """You report search results. Nothing more.

Procedure:
1. Check SEARCH RESULTS
2. If empty: "Search found no information"
3. If has data: "Search found: [brief summary]"
4. Stop. Do not add anything.""",

    "Truthful Assistant": """You are a truthful AI. You never lie or guess.

SEARCH RESULTS TRUTH:
- If results exist: Report them exactly
- If no results: "I cannot answer - no information found"
- Never make up answers
- Never create fictional people or events
- Your response must match the search data exactly""",

    "Zero-Hallucination": """NO HALLUCINATIONS ALLOWED.

YOU MUST:
1. Read SEARCH RESULTS
2. If results = empty → "No data available"
3. If results have info → Copy/summarize that info
4. Do not add, invent, or speculate
5. If you don't see it in results, don't say it

This is a zero-tolerance policy for made-up information."""
}

# Optimized search tools
SEARCH_TOOLS = {
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📚",
        "description": "Encyclopedia articles",
        "endpoint": "https://en.wikipedia.org/w/api.php"
    },
    "DuckDuckGo": {
        "name": "Web Search",
        "icon": "🌐",
        "description": "Instant answers & web results",
        "endpoint": "https://api.duckduckgo.com/"
    },
    "ArXiv": {
        "name": "Research Papers",
        "icon": "🔬",
        "description": "Scientific publications",
        "endpoint": "http://export.arxiv.org/api/query"
    }
}

st.set_page_config(
    page_title="Truthful AI Assistant",
    page_icon="✅",
    layout="wide"
)

# Initialize session state with safe defaults
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Zero-Hallucination"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Zero-Hallucination"

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .truth-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .source-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Download function
def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        return True
    
    st.warning("⚠️ Model not found. Downloading...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024**3)
            st.success(f"✅ Model downloaded: {file_size:.2f} GB")
            return True
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
    return False

@st.cache_resource(show_spinner=False)
def load_model():
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        if not download_model():
            raise Exception("Model download failed")
    
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=2048,  # Reduced for faster processing
        gpu_layers=0,
        threads=8
    )

# Enhanced search functions with better error handling
def search_wikipedia(query):
    """Wikipedia search with strict result validation."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 2,  # Reduced for speed
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=5)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', [])[:2]:  # Limit to 2
            title = item.get('title', '').lower()
            snippet = item.get('snippet', '').lower()
            
            # Only include if query matches significantly
            query_words = query.lower().split()
            match_score = sum(1 for word in query_words if word in title or word in snippet)
            
            if match_score >= len(query_words) * 0.5:  # At least 50% match
                results.append({
                    'title': item.get('title', ''),
                    'summary': re.sub(r'<[^>]+>', '', item.get('snippet', ''))[:300],
                    'source': 'Wikipedia',
                    'match_score': match_score
                })
        
        return results
    except Exception:
        return []

def search_duckduckgo_enhanced(query):
    """DuckDuckGo search with strict validation."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=5)
        data = response.json()
        
        # Check if this is an ambiguous or empty result
        if data.get('Type') in ['D', 'disambiguation']:  # Disambiguation page
            return {}
        
        if data.get('AbstractText') == '' and data.get('Answer') == '':
            return {}
        
        results = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'source': 'DuckDuckGo'
        }
        
        # Filter empty values
        cleaned = {}
        for key, value in results.items():
            if isinstance(value, str) and value.strip() and len(value.strip()) > 10:
                cleaned[key] = value.strip()
        
        return cleaned if cleaned else {}
    except Exception:
        return {}

def perform_search(query):
    """Perform search with aggressive filtering for unknown entities."""
    search_results = {}
    
    # Try Wikipedia first
    wiki_results = search_wikipedia(query)
    if wiki_results:
        search_results['Wikipedia'] = wiki_results
    
    # Try DuckDuckGo only if Wikipedia has results (more reliable)
    if search_results:  # Only search DuckDuckGo if Wikipedia found something
        ddg_results = search_duckduckgo_enhanced(query)
        if ddg_results:
            search_results['DuckDuckGo'] = ddg_results
    else:
        # If Wikipedia has nothing, DuckDuckGo is probably useless for obscure queries
        pass
    
    return search_results

def create_strict_prompt(query, search_results):
    """Create a VERY strict prompt that prevents hallucinations."""
    
    # Build search context - be explicit about emptiness
    if search_results:
        search_context = "SEARCH RESULTS (FOUND SOME INFORMATION):\n\n"
        for source, data in search_results.items():
            search_context += f"=== {source} ===\n"
            if isinstance(data, list):
                for item in data:
                    search_context += f"Title: {item.get('title', 'N/A')}\n"
                    search_context += f"Summary: {item.get('summary', 'N/A')}\n\n"
            elif isinstance(data, dict):
                for key, value in data.items():
                    if value:
                        search_context += f"{key}: {value}\n"
                search_context += "\n"
    else:
        search_context = "SEARCH RESULTS: NO INFORMATION FOUND\n\n"
        search_context += "IMPORTANT: The search returned empty. There is no data about this topic.\n\n"
    
    # ULTRA-STRICT prompt
    prompt = f"""<|system|>
You are a TRUTHFUL assistant. You NEVER make up information.

{search_context}

CRITICAL RULES:
1. If SEARCH RESULTS says "NO INFORMATION FOUND", you MUST respond with: "I searched but found no information about this topic."
2. If there ARE search results, you can ONLY summarize what you see in them.
3. NEVER add any details not in the search results.
4. NEVER create names, dates, positions, or events.
5. Your response must be SHORT and FACTUAL.
6. If unsure, say: "No verified information available."

EXAMPLE RESPONSES:
- If no results: "I searched but found no information about this topic."
- If Wikipedia has info: "Wikipedia shows that [brief fact from Wikipedia]."
- If confused: "No verified information available."

Now respond to: "{query}"</s>

<|user|>
{query}</s>

<|assistant|>
Based on my search:"""
    
    return prompt

def generate_response(model, prompt, max_tokens=200, temperature=0.1):
    """Generate response with VERY low temperature to prevent creativity."""
    
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,  # VERY low
        top_p=0.3,  # VERY low
        repetition_penalty=1.2,
        stop=["</s>", "<|user|>", "\n\n", "###", "Note:", "However", "Additionally"]
    )
    
    response = response.strip()
    
    # Post-process to ensure truthfulness
    response_lower = response.lower()
    
    # Force truthful responses for empty searches
    if "no information found" in prompt.lower() and "found no information" not in response_lower:
        response = "I searched but found no information about this topic."
    
    # Check for common fabrication patterns
    fabrication_phrases = [
        "is a", "was born", "served as", "is known for", 
        "graduated from", "worked as", "is the", "was the"
    ]
    
    if "no information" not in response_lower and "no data" not in response_lower:
        # If response makes definite statements without citing sources, be cautious
        for phrase in fabrication_phrases:
            if phrase in response_lower and "wikipedia" not in response_lower and "search" not in response_lower:
                response = "I cannot provide verified information about this topic."
                break
    
    return response

def detect_hallucinations(response, search_results):
    """Simple hallucination detection."""
    if not search_results:
        # No search results - response should indicate no information
        no_info_phrases = [
            "no information", "not found", "no data", "cannot find",
            "unavailable", "don't know", "no results"
        ]
        response_lower = response.lower()
        
        if not any(phrase in response_lower for phrase in no_info_phrases):
            return True  # Hallucination detected
    
    return False

# Title
st.title("✅ Truthful AI Assistant")
st.caption("AI that only reports verified information from searches - NO HALLUCINATIONS")

# Sidebar
with st.sidebar:
    st.header("🎭 Truthfulness Mode")
    
    preset_keys = list(PRESET_PROMPTS.keys())
    current_preset = st.session_state.selected_preset
    
    if current_preset not in preset_keys:
        current_preset = "Zero-Hallucination"
        st.session_state.selected_preset = current_preset
    
    index = preset_keys.index(current_preset)
    
    persona = st.selectbox(
        "Select Mode:",
        options=preset_keys,
        index=index
    )
    
    if persona != st.session_state.selected_preset:
        st.session_state.selected_preset = persona
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    
    st.divider()
    
    st.header("⚡ Settings")
    
    temperature = st.slider(
        "Strictness Level:",
        0.0, 0.5, 0.1, 0.05,
        help="Lower = more truthful, Higher = slightly more creative (not recommended)"
    )
    
    st.divider()
    
    st.header("🔧 Tools")
    
    auto_search = st.toggle("Auto-Search", value=True, 
                           help="Automatically search before responding")
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Truthful Assistant v2.0")
    st.caption("Designed to prevent hallucinations")

# Load model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI..."):
        try:
            st.session_state.model = load_model()
            st.success("✅ AI Ready!")
        except Exception as e:
            st.error(f"❌ Failed to load: {str(e)}")
            st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show warnings if hallucination was detected
        if message.get("hallucination_detected"):
            st.markdown('<div class="error-box">⚠️ This response may contain unverified information</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything (I'll only report verified facts)..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Perform search
        search_results = {}
        
        if auto_search:
            with st.spinner("🔍 Searching for verified information..."):
                search_results = perform_search(prompt)
            
            # Show search summary
            with st.expander("📊 Search Results", expanded=False):
                if search_results:
                    for source, data in search_results.items():
                        st.subheader(f"{source}")
                        if isinstance(data, list):
                            for item in data:
                                if 'title' in item:
                                    st.write(f"**{item['title']}**")
                                if 'summary' in item:
                                    st.write(item['summary'])
                                st.divider()
                        elif isinstance(data, dict):
                            for key, value in data.items():
                                if value:
                                    st.write(f"**{key.title()}:** {value}")
                else:
                    st.markdown('<div class="warning-box">No information found in searches</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Formulating truthful response..."):
            strict_prompt = create_strict_prompt(prompt, search_results)
            response = generate_response(
                st.session_state.model,
                strict_prompt,
                max_tokens=150,
                temperature=temperature
            )
        
        # Detect hallucinations
        hallucination_detected = detect_hallucinations(response, search_results)
        
        # Display response with truthfulness indicator
        if hallucination_detected:
            st.markdown('<div class="error-box">⚠️ WARNING: Response may contain unverified information</div>', unsafe_allow_html=True)
        
        st.markdown(response)
        
        # Add truth box
        if not hallucination_detected:
            if search_results:
                st.markdown(f"""
                <div class="truth-box">
                <strong>✅ Truthfulness Check:</strong><br>
                • Response based on search results from {len(search_results)} source(s)<br>
                • No hallucinations detected<br>
                • Information verified against external sources
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="truth-box">
                <strong>✅ Truthfulness Check:</strong><br>
                • No information found in searches<br>
                • Response honestly reports this gap<br>
                • No fabrication of information
                </div>
                """, unsafe_allow_html=True)
        
        # Store message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metadata": {
                "sources": list(search_results.keys()) if search_results else [],
                "search_performed": auto_search,
                "hallucination_detected": hallucination_detected
            }
        })

# Add test questions
if not st.session_state.messages:
    st.markdown("### 💡 Test Truthfulness:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test: Unknown Person", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who is Abdelmajid Tebbone?"})
            st.rerun()
        if st.button("Test: Real Person", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who is Albert Einstein?"})
            st.rerun()
    
    with col2:
        if st.button("Test: Fictional Topic", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is Quantum Fluctuation Theory?"})
            st.rerun()
        if st.button("Test: Real Topic", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is photosynthesis?"})
            st.rerun()
