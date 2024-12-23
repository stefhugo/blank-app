st.write("Debug: Checking GOOGLE_API_KEY...")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set. Please set the environment variable.")
else:
    st.write(f"Debug: API Key Loaded: {api_key[:4]}***")  # Mask the key for security
    genai.configure(api_key=api_key)
