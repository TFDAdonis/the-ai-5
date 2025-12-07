import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts
PRESET_PROMPTS = {
    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives. 

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully, identify key elements
2. **Contextualization**: Place the topic in historical, cultural, or disciplinary context
3. **Multi-Source Analysis**: Examine information from different sources critically
4. **Pattern Recognition**: Identify connections, contradictions, gaps
5. **Synthesis**: Combine insights into coherent understanding
6. **Critical Evaluation**: Assess reliability, bias, significance
7. **Insight Generation**: Provide original perspectives or connections
8. **Actionable Knowledge**: Suggest applications, further questions, implications

RESPONSE STRUCTURE:
- Start with brief overview
- Present analysis with reasoning
- Reference sources when available
- Highlight interesting connections
- Acknowledge uncertainties
- End with thought-provoking questions or suggestions

TONE: Analytical yet engaging, precise yet accessible.""",

    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

SPECIALTY THINKING PROCESS:
1. **Geospatial Context**: How does location/spatial relationships matter?
2. **Temporal Analysis**: What changes over time? Historical patterns?
3. **Data Source Evaluation**: Satellite, ground, or derived data reliability?
4. **Multi-Scale Thinking**: From local to global perspectives
5. **Practical Applications**: Real-world uses of the information
6. **Ethical Considerations**: Privacy, representation, accessibility issues

EXPERTISE: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring
STYLE: Enthusiastic, precise, eager to explore spatial dimensions of any topic""",

    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.

ANALYTICAL APPROACH:
1. **Source Triangulation**: Cross-reference multiple information sources
2. **Credibility Assessment**: Evaluate source reliability, date, bias
3. **Trend Identification**: Spot patterns, changes, anomalies
4. **Comparative Analysis**: Similarities/differences across contexts
5. **Implication Mapping**: Consequences, applications, risks
6. **Knowledge Gaps**: What's missing or needs verification

Always provide structured, evidence-based analysis with clear reasoning.""",

    "Critical Thinker": """You excel at questioning assumptions and examining topics from multiple angles.

CRITICAL THINKING TOOLS:
1. **Assumption Detection**: What unstated beliefs underlie this?
2. **Perspective Switching**: How would different groups view this?
3. **Logical Analysis**: Are arguments valid, evidence sufficient?
4. **Counterfactual Thinking**: What if things were different?
5. **Ethical Reflection**: Moral dimensions, consequences
6. **Practical Reality Check**: Feasibility, implementation issues

Challenge conventional wisdom while remaining constructive.""",

    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights.

CREATIVE PROCESS:
1. **Divergent Thinking**: Generate multiple possible interpretations
2. **Analogical Reasoning**: What similar patterns exist elsewhere?
3. **Metaphorical Connection**: What metaphors illuminate this?
4. **Interdisciplinary Bridging**: Connect across fields
5. **Future Projection**: How might this evolve or transform?
6. **Alternative Framing**: Different ways to conceptualize

Be imaginative while staying grounded in evidence."""
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
    },
    "Books": {
        "name": "Books",
        "icon": "📖",
        "description": "Book information",
        "endpoint": "https://openlibrary.org/search.json"
    },
    "Countries": {
        "name": "Country Data",
        "icon": "🌍",
        "description": "Country information",
        "endpoint": "https://restcountries.com/v3.1/"
    },
    "Weather": {
        "name": "Weather",
        "icon": "🌤️",
        "description": "Weather information",
        "endpoint": "https://wttr.in/"
    },
    "GitHub": {
        "name": "Code Repos",
        "icon": "💻",
        "description": "GitHub repositories",
        "endpoint": "https://api.github.com/search/repositories"
    }
}

st.set_page_config(
    page_title="DeepThink Pro",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state with safe defaults
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Deep Thinker Pro"

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .thinking-bubble {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
    }
    .analysis-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffb300;
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
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
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
        context_length=4096,
        gpu_layers=0,
        threads=8
    )

# Enhanced search functions
def search_wikipedia(query):
    """Enhanced Wikipedia search with better parsing."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=8)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get detailed page info
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info|categories',
                'inprop': 'url',
                'exintro': 1,
                'explaintext': 1,
                'pageids': item['pageid']
            }
            response2 = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params2, timeout=8)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_info in pages.values():
                    extract = page_info.get('extract', '')
                    if extract:
                        # Clean the extract
                        extract = re.sub(r'\n+', ' ', extract)
                        extract = re.sub(r'\s+', ' ', extract)
                        
                        results.append({
                            'title': page_info.get('title', ''),
                            'summary': extract[:500] + ('...' if len(extract) > 500 else ''),
                            'url': page_info.get('fullurl', ''),
                            'categories': list(page_info.get('categories', []))[:5],
                            'wordcount': page_info.get('wordcount', 0),
                            'source': 'Wikipedia',
                            'relevance': item.get('score', 0)
                        })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True) if results else []
    except Exception:
        return []

def search_duckduckgo_enhanced(query):
    """Enhanced DuckDuckGo search with better parsing."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=8)
        data = response.json()
        
        results = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'categories': [topic.get('Name', '') for topic in data.get('Categories', [])[:3]],
            'related_topics': [topic.get('Text', '') for topic in data.get('RelatedTopics', [])[:5] if isinstance(topic, dict)],
            'source': 'DuckDuckGo'
        }
        
        # Clean and filter empty values
        cleaned = {}
        for key, value in results.items():
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
            elif isinstance(value, list) and value:
                cleaned[key] = [v.strip() for v in value if v and v.strip()]
        
        return cleaned if cleaned else {}
    except Exception:
        return {}

def search_arxiv_enhanced(query):
    """Enhanced ArXiv search."""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 3,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=10)
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip() if entry.find('{http://www.w3.org/2005/Atom}title') is not None else ''
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip() if entry.find('{http://www.w3.org/2005/Atom}summary') is not None else ''
            
            if title and summary:
                papers.append({
                    'title': title,
                    'summary': summary[:400] + '...' if len(summary) > 400 else summary,
                    'published': entry.find('{http://www.w3.org/2005/Atom}published').text[:10] if entry.find('{http://www.w3.org/2005/Atom}published') is not None else '',
                    'source': 'ArXiv',
                    'relevance': 1.0  # Simple relevance score
                })
        
        return papers
    except Exception:
        return []

def smart_source_selector(query):
    """Intelligently select which sources to search based on query."""
    query_lower = query.lower()
    
    # Check for specific patterns
    is_historical = any(word in query_lower for word in ['history', 'historical', 'century', 'war', 'battle', 'king', 'queen', 'emperor', 'emir'])
    is_scientific = any(word in query_lower for word in ['science', 'research', 'study', 'paper', 'experiment', 'data', 'analysis'])
    is_technical = any(word in query_lower for word in ['code', 'programming', 'software', 'algorithm', 'github', 'python', 'javascript'])
    is_geographical = any(word in query_lower for word in ['country', 'city', 'capital', 'population', 'map', 'location', 'weather'])
    is_conceptual = any(word in query_lower for word in ['what is', 'define', 'meaning', 'concept', 'theory', 'philosophy'])
    is_person = any(word in query_lower for word in ['who is', 'biography', 'born', 'died', 'leader', 'president', 'emir', 'prime minister', 'minister'])
    
    # Select sources based on query type
    sources = []
    
    # Always include Wikipedia for factual information
    sources.append(('Wikipedia', search_wikipedia))
    
    # Add DuckDuckGo for quick answers
    sources.append(('DuckDuckGo', search_duckduckgo_enhanced))
    
    # Add specialized sources based on query
    if is_historical or is_person:
        sources.append(('Books', lambda q: []))  # Placeholder for books API
    
    if is_scientific:
        sources.append(('ArXiv', search_arxiv_enhanced))
    
    if is_technical:
        sources.append(('GitHub', lambda q: []))  # Placeholder for GitHub
    
    if is_geographical:
        sources.append(('Countries', lambda q: []))  # Placeholder for countries
        sources.append(('Weather', lambda q: []))  # Placeholder for weather
    
    return sources[:5]  # Limit to 5 sources

def perform_intelligent_search(query):
    """Perform parallel search on intelligently selected sources."""
    sources = smart_source_selector(query)
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
        future_to_source = {executor.submit(func, query): name for name, func in sources}
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                data = future.result(timeout=8)
                if data:  # Only include non-empty results
                    results[source_name] = data
            except Exception:
                continue
    
    return results

def analyze_search_results(query, results):
    """Analyze search results to extract key insights."""
    analysis = {
        'key_facts': [],
        'conflicting_info': [],
        'knowledge_gaps': [],
        'source_quality': {},
        'main_themes': [],
        'has_information': False
    }
    
    # Extract key facts from each source
    for source, data in results.items():
        if source == 'Wikipedia' and isinstance(data, list):
            analysis['has_information'] = True
            for item in data[:2]:
                if 'summary' in item:
                    analysis['key_facts'].append({
                        'fact': item['summary'][:200],
                        'source': source,
                        'title': item.get('title', '')
                    })
        
        elif source == 'DuckDuckGo' and isinstance(data, dict):
            if data.get('answer') or data.get('abstract'):
                analysis['has_information'] = True
            if data.get('answer'):
                analysis['key_facts'].append({
                    'fact': data['answer'],
                    'source': source,
                    'type': 'direct_answer'
                })
            if data.get('abstract'):
                analysis['key_facts'].append({
                    'fact': data['abstract'][:200],
                    'source': source,
                    'type': 'abstract'
                })
        
        elif source == 'ArXiv' and isinstance(data, list) and data:
            analysis['has_information'] = True
            for paper in data[:1]:
                analysis['key_facts'].append({
                    'fact': f"Research paper: {paper.get('title', '')}",
                    'source': source,
                    'type': 'scientific'
                })
    
    # Identify potential knowledge gaps
    if not analysis['has_information']:
        analysis['knowledge_gaps'].append(f"No information found about '{query}' in searched sources")
    
    # Assess source quality
    for source in results:
        if source == 'Wikipedia':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'broad'}
        elif source == 'ArXiv':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'specialized'}
        elif source == 'DuckDuckGo':
            analysis['source_quality'][source] = {'reliability': 'medium', 'coverage': 'general'}
    
    return analysis

def create_thinking_prompt(query, messages, system_prompt, search_results, search_analysis):
    """Create an enhanced prompt that encourages deep thinking based on actual search results."""
    
    # Build search context
    if search_results:
        search_context = "RELEVANT INFORMATION FOUND:\n\n"
        
        for source, data in search_results.items():
            search_context += f"=== {source.upper()} ===\n"
            
            if isinstance(data, list) and data:
                for item in data[:2]:
                    if isinstance(item, dict):
                        if 'title' in item:
                            search_context += f"Title: {item['title']}\n"
                        if 'summary' in item:
                            search_context += f"Summary: {item['summary']}\n"
                        if 'answer' in item:
                            search_context += f"Answer: {item['answer']}\n"
                        search_context += "\n"
            
            elif isinstance(data, dict) and data:
                for key, value in data.items():
                    if key not in ['source', 'type'] and value:
                        if isinstance(value, list):
                            search_context += f"{key}: {', '.join(str(v) for v in value[:3])}\n"
                        else:
                            search_context += f"{key}: {value}\n"
                search_context += "\n"
    else:
        search_context = "NO INFORMATION FOUND IN SEARCHES\n\n"
    
    # Add analysis insights
    search_context += "ANALYSIS INSIGHTS:\n"
    if search_analysis['key_facts']:
        search_context += "• Key facts identified from sources\n"
        for fact in search_analysis['key_facts'][:3]:
            search_context += f"  - {fact['source']}: {fact['fact'][:100]}...\n"
    if search_analysis['knowledge_gaps']:
        search_context += f"• Knowledge gaps: {search_analysis['knowledge_gaps'][0]}\n"
    
    if not search_analysis['has_information']:
        search_context += "• WARNING: No reliable information found. Do not invent facts.\n"
    
    # Build conversation history
    conversation = ""
    for msg in messages[-4:]:  # Last 4 messages for context
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    
    # Extract key facts for strict usage
    extracted_facts = []
    for fact in search_analysis['key_facts'][:5]:
        extracted_facts.append(f"{fact['source']}: {fact['fact']}")
    
    facts_context = "EXTRACTED FACTS:\n" + "\n".join([f"- {fact}" for fact in extracted_facts]) if extracted_facts else "NO FACTS EXTRACTED"
    
    # Final prompt
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

USER'S QUESTION: {query}

{search_context}

{facts_context}

CONVERSATION CONTEXT:
{conversation}

CRITICAL RULES:
1. ONLY use information from EXTRACTED FACTS or RELEVANT INFORMATION sections above
2. If no information is found, clearly state: "I couldn't find verified information about this topic"
3. NEVER fabricate or guess information - if you don't know, say so
4. Always cite which source provided the information (e.g., "According to Wikipedia...")
5. If sources conflict or information is limited, mention this uncertainty
6. Do not generate biographies or details about people not found in the search results
7. If asked about a specific person and no information is found, say they don't appear to be a notable public figure

THINKING PROCESS:
1. Verify what information exists in the search results
2. Identify the most reliable facts
3. Consider what's missing or uncertain
4. Formulate an honest response based ONLY on available information
5. End with appropriate caveats or suggestions for better searches</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_thoughtful_response(model, prompt, search_results, search_analysis, max_tokens=768, temperature=0.7):
    """Generate response that properly integrates search results."""
    
    # Adjust temperature based on search results
    if not search_analysis['has_information']:
        # Lower temperature when no info found to prevent hallucinations
        temperature = min(temperature, 0.3)
    
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "\n\nUser:", "### END", "Sources:", "Note:"]
    )
    
    # Clean up response
    response = response.strip()
    
    # Add source attribution if not already present
    sources_used = list(search_results.keys())
    if sources_used and "source:" not in response.lower() and "according to" not in response.lower():
        response += f"\n\n*Information gathered from: {', '.join(set(sources_used))}*"
    
    # Ensure it doesn't cut off mid-sentence
    if response and response[-1] not in ['.', '!', '?', '"', "'"]:
        # Try to complete the sentence
        try:
            last_sentence = response.split('.')[-1]
            if len(last_sentence.split()) < 5:  # Very short last sentence
                completion = model(
                    prompt + response,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.9
                )
                response = response + " " + completion.strip()
        except:
            pass
    
    return response

def verify_facts_against_sources(response, search_results, search_analysis):
    """Check if the response contains unsupported claims."""
    warnings = []
    
    # Common fabrication patterns to watch for
    fabrication_patterns = {
        "political_positions": [
            "Prime Minister", "President", "Minister", "Chancellor", 
            "Secretary of State", "Member of Parliament", "Congressman",
            "Senator", "Governor", "Mayor"
        ],
        "academic_credentials": [
            "graduated from", "degree in", "PhD", "Professor", 
            "researcher at", "studied at"
        ],
        "biographical_details": [
            "was born", "born in", "died in", "served as",
            "appointed as", "elected as"
        ]
    }
    
    # Check for each pattern
    response_lower = response.lower()
    
    for category, patterns in fabrication_patterns.items():
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in response_lower:
                # Check if any search result supports this
                supported = False
                for source, data in search_results.items():
                    if isinstance(data, (list, dict)):
                        data_str = str(data).lower()
                        if pattern_lower in data_str:
                            supported = True
                            break
                
                if not supported and search_analysis['has_information']:
                    warnings.append(f"Unsupported claim about {pattern}")
                    break
    
    return warnings

# Title with emojis
st.title("🧠 DeepThink Pro - Factual Edition")
st.caption("Advanced AI that uses search results to provide accurate, source-based responses")

# Sidebar
with st.sidebar:
    st.header("🎭 Thinking Persona")
    
    # Safely get index for selectbox
    preset_keys = list(PRESET_PROMPTS.keys())
    current_preset = st.session_state.selected_preset
    
    # Ensure current preset is valid
    if current_preset not in preset_keys:
        current_preset = "Deep Thinker Pro"
        st.session_state.selected_preset = current_preset
    
    index = preset_keys.index(current_preset)
    
    persona = st.selectbox(
        "Select AI Persona:",
        options=preset_keys,
        index=index
    )
    
    if persona != st.session_state.selected_preset:
        st.session_state.selected_preset = persona
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    
    st.divider()
    
    st.header("⚡ Thinking Parameters")
    
    thinking_mode = st.radio(
        "Thinking Mode:",
        ["Analytical", "Creative", "Critical", "Balanced"],
        index=3
    )
    
    research_depth = st.select_slider(
        "Research Depth:",
        options=["Quick Scan", "Moderate", "Deep Dive", "Exhaustive"],
        value="Moderate"
    )
    
    temperature = st.slider(
        "Creativity Level:",
        0.1, 1.5, 0.7, 0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    st.header("🔧 Tools")
    
    auto_search = st.toggle("Auto-Research", value=True)
    show_thinking = st.toggle("Show Thinking Process", value=True)
    fact_checking = st.toggle("Fact-Checking Alerts", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🧠 Reset AI", use_container_width=True):
            st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]
            st.session_state.selected_preset = "Deep Thinker Pro"
            st.rerun()
    
    st.divider()
    st.caption("DeepThink Pro v1.2 - Factual Edition")
    st.caption("Now with improved source-based responses")

# Main interface
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🧠 DeepThink Pro")
with col2:
    if auto_search:
        st.success("🔍 Auto-Research ON")
with col3:
    if show_thinking:
        st.info("💭 Showing Thoughts")

# Display current persona
with st.expander("🤖 Active Persona", expanded=False):
    st.write(f"**{st.session_state.selected_preset}**")
    st.caption(st.session_state.system_prompt[:300] + "...")

# Load model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI Brain..."):
        try:
            st.session_state.model = load_model()
            st.success("✅ AI Ready for Deep Thinking!")
        except Exception as e:
            st.error(f"❌ Failed to load: {str(e)}")
            st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source tags if available
        if "sources" in message.get("metadata", {}):
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            for source in message["metadata"]["sources"]:
                st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show warnings if available
        if "warnings" in message.get("metadata", {}):
            for warning in message["metadata"]["warnings"]:
                st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Step 1: Show thinking
        thinking_placeholder = st.empty()
        
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>💭 Initial Analysis:</strong><br>
            1. Parsing question structure and intent<br>
            2. Identifying key concepts and entities<br>
            3. Determining appropriate research approach<br>
            4. Preparing search strategy...
            </div>
            """, unsafe_allow_html=True)
        
        # Step 2: Intelligent Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            if show_thinking:
                thinking_placeholder.markdown("""
                <div class="thinking-bubble">
                <strong>🔍 Smart Research:</strong><br>
                • Analyzing query type and selecting optimal sources<br>
                • Conducting parallel searches across selected databases<br>
                • Evaluating source reliability and relevance...
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Conducting intelligent research..."):
                search_results = perform_intelligent_search(prompt)
                
                if search_results:
                    search_analysis = analyze_search_results(prompt, search_results)
                    
                    # Display search summary
                    with st.expander("📊 Research Summary", expanded=False):
                        for source, data in search_results.items():
                            st.subheader(f"{SEARCH_TOOLS.get(source, {}).get('icon', '📌')} {source}")
                            
                            if isinstance(data, list) and data:
                                for item in data[:2]:
                                    if isinstance(item, dict):
                                        with st.container():
                                            if 'title' in item:
                                                st.write(f"**{item['title']}**")
                                            if 'summary' in item:
                                                st.write(item['summary'])
                                            st.divider()
                            elif isinstance(data, dict) and data:
                                for key, value in data.items():
                                    if key not in ['source', 'type'] and value:
                                        if isinstance(value, list):
                                            st.write(f"**{key.title()}:** {', '.join(str(v) for v in value[:3])}")
                                        else:
                                            st.write(f"**{key.title()}:** {value}")
                        if not search_analysis['has_information']:
                            st.warning("No substantial information found in searches")
                else:
                    search_analysis = analyze_search_results(prompt, {})
                    with st.expander("📊 Research Summary", expanded=False):
                        st.warning("No search results found")
        
        # Step 3: Generate thoughtful response
        if show_thinking:
            if search_analysis.get('has_information'):
                thinking_text = """
                <div class="thinking-bubble">
                <strong>🤔 Deep Synthesis:</strong><br>
                • Integrating verified information from sources<br>
                • Applying critical thinking to available facts<br>
                • Formulating evidence-based response<br>
                • Preparing honest assessment of knowledge gaps...
                </div>
                """
            else:
                thinking_text = """
                <div class="thinking-bubble">
                <strong>⚠️ Limited Information:</strong><br>
                • No verified information found in searches<br>
                • Preparing honest response about information gap<br>
                • Will not fabricate or guess information<br>
                • Suggesting alternative search approaches...
                </div>
                """
            thinking_placeholder.markdown(thinking_text, unsafe_allow_html=True)
        
        with st.spinner("🧠 Engaging deep thinking process..."):
            # Create enhanced prompt
            enhanced_prompt = create_thinking_prompt(
                prompt, 
                st.session_state.messages,
                st.session_state.system_prompt,
                search_results,
                search_analysis
            )
            
            # Adjust tokens based on research depth
            if research_depth == "Quick Scan":
                tokens = 512
            elif research_depth == "Moderate":
                tokens = 768
            elif research_depth == "Deep Dive":
                tokens = 1024
            else:  # Exhaustive
                tokens = 1536
            
            # Generate response
            response = generate_thoughtful_response(
                st.session_state.model,
                enhanced_prompt,
                search_results,
                search_analysis,
                max_tokens=tokens,
                temperature=temperature
            )
            
            # Verify facts
            warnings = []
            if fact_checking and search_results:
                warnings = verify_facts_against_sources(response, search_results, search_analysis)
        
        # Clear thinking placeholders
        thinking_placeholder.empty()
        
        # Display response
        st.markdown(response)
        
        # Show warnings if any
        if warnings and fact_checking:
            for warning in warnings[:2]:  # Limit to 2 warnings
                st.markdown(f'<div class="warning-box">⚠️ Fact-checking alert: {warning}</div>', unsafe_allow_html=True)
        
        # Add analysis box for deep thinking
        if thinking_mode != "Quick Scan" and search_results and search_analysis.get('has_information'):
            st.markdown(f"""
            <div class="analysis-box">
            <strong>📈 Analysis Summary:</strong><br>
            • Information synthesized from {len(search_results)} sources<br>
            • {len(search_analysis['key_facts'])} key facts identified<br>
            • Reliability assessment: {', '.join([f'{k}: {v["reliability"]}' for k, v in search_analysis['source_quality'].items()])}<br>
            • Knowledge gaps noted for further research
            </div>
            """, unsafe_allow_html=True)
        elif not search_analysis.get('has_information'):
            st.markdown("""
            <div class="analysis-box">
            <strong>⚠️ Information Gap Detected:</strong><br>
            • No reliable information found in searches<br>
            • Response is based on acknowledging this gap<br>
            • No fabrication of information<br>
            • Consider refining search terms or checking alternative sources
            </div>
            """, unsafe_allow_html=True)
        
        # Store message with metadata
        metadata = {
            "sources": list(search_results.keys()) if search_results else [],
            "thinking_mode": thinking_mode,
            "research_depth": research_depth,
            "timestamp": datetime.now().isoformat(),
            "has_information": search_analysis.get('has_information', False)
        }
        
        if warnings:
            metadata["warnings"] = warnings
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metadata": metadata
        })

# Add quick questions examples
if not st.session_state.messages:
    st.markdown("### 💡 Try asking about:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Historical figure analysis", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who was Napoleon Bonaparte and what was his impact on Europe?"})
            st.rerun()
        if st.button("Scientific concept", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain quantum entanglement in simple terms"})
            st.rerun()
        if st.button("Test: Unknown person", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who is Abdelmajid Tebbone?"})
            st.rerun()
    
    with col2:
        if st.button("Current events", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the main challenges facing renewable energy adoption today?"})
            st.rerun()
        if st.button("Philosophical question", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the meaning of consciousness according to different philosophical traditions?"})
            st.rerun()
        if st.button("Test: Fictional character", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who is John Galt from Atlas Shrugged?"})
            st.rerun()
