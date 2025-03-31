import streamlit as st
import requests
from typing import Optional
import json

# Config for page
st.set_page_config(
    page_title="LLM Query Interface",
    page_icon="🤖",
    layout="wide"
)

# Store your ngrok URL in session state
if 'API_URL' not in st.session_state:
    # Replace with your ngrok URL
    st.session_state.API_URL = "https://your-ngrok-url.ngrok-free.app/"

def verify_api_connection():
    """Verify that the API is accessible"""
    try:
        base_url = st.session_state.API_URL.rsplit('/', 1)[0]
        response = requests.get(base_url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def process_query(
    query: str,
    max_length: Optional[int] = 512,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9
):
    """Send query to API and get response"""
    try:
        payload = {
            "query": query,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Log the request for debugging
        st.write("Sending request to:", st.session_state.API_URL+'process')
        st.write("Payload:", json.dumps(payload, indent=2))
        
        response = requests.post(st.session_state.API_URL+'process', json=payload)
        
        # Log the response status for debugging
        st.write(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            st.error(f"API returned error {response.status_code}: {response.text}")
            return None
            
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None

def main():
    st.title("🤖 LLM Query Interface")
    
    # API URL Configuration
    with st.expander("API Configuration"):
        new_url = st.text_input("API URL", st.session_state.API_URL)
        if new_url != st.session_state.API_URL:
            st.session_state.API_URL = new_url
            st.success("API URL updated!")
    
    # API Connection Status
    if verify_api_connection():
        st.success("✅ API is connected and running")
    else:
        st.error("❌ Cannot connect to API. Please check the URL and make sure the server is running.")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Enter your query:",
            height=150,
            placeholder="Type your question here..."
        )
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            max_length = st.slider("Maximum Length", 64, 1024, 512)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        
        # Process button
        if st.button("Submit Query", type="primary"):
            if not query:
                st.warning("Please enter a query first.")
                return
                
            with st.spinner("Processing your query..."):
                result = process_query(
                    query=query,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
                
                if result:
                    st.success(f"Processed in {result['processing_time']:.2f} seconds")
    
    with col2:
        # Display results
        if 'result' in locals() and result:
            st.subheader("Response")
            st.write(result["response"])
            
            # Processing time in small text
            st.text(f"Processing time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    main()