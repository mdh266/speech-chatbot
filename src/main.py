import os 
import streamlit as st
from utils import (text_to_speech,
                   speech_to_text,
                   ask_question,
                   create_audio,
                   tuplify,
                   clear_session)


def main(debug: bool = False):
    if debug:
        from dotenv import load_dotenv
        load_dotenv()

    groq_api = os.getenv("GROQ_API_KEY")
    google_api = os.getenv("GOOGLE_API_KEY")

    if "messages" not in st.session_state:
        clear_session()
    

    st.markdown("# Serverless Multimodal AI Chatbot")
    st.markdown("Have a conversation with an Artifical Intelligence Bot in English, Hebrew or French")

    col1, col2 = st.columns(2, gap="large")

    response = None
    transcribe = None

    with col1:
        
        with st.form("conversation"):
            st.markdown("### Submit A Question")
            
            human_language = st.selectbox("Human Language:", ("English"))#, "Hebrew", "French"))
            ai_language = st.selectbox("Bot Language:", ("English"))#, "Hebrew", "French"))
            transcribe = st.checkbox("Transcribe Conversation")
    
            
            question = create_audio()
            submitted = st.form_submit_button("Submit")

        if submitted and question:

            text_question, english_question = speech_to_text(
                                                human_language=human_language,
                                                google_api=google_api,
                                                question=question)
            
            st.session_state.messages.append({
                "role": "human", 
                "content": f"Human: {text_question}"
            })

            st.session_state.english_messages.append({
                "role": "human", 
                "content": f"Human: {english_question}"
            })
            
            history = tuplify(st.session_state.english_messages)
            
            answer, translated_answer = ask_question(
                                            history=history, 
                                            question=english_question,
                                            ai_language=ai_language,
                                            google_api=google_api)

            st.session_state.english_messages.append({
                    "role": "ai", 
                    "content": f"Bot: {answer}"
            })

            st.session_state.messages.append({
                "role": "ai", 
                "content": f"Bot: {translated_answer}"
            })

            response = text_to_speech(
                            ai_language=ai_language,
                            google_api=google_api,
                            translated_answer=translated_answer)

        st.button("Clear Conversation", on_click=clear_session) 

        with col2:

            st.markdown("**Bot's Last Response**")    
            with st.container(border=True, height=90):   
                # st.markdown("**Bot's Last Response**")                    
                if response:
                        st.audio(response.audio_content, "audio/mp3")

            st.markdown("**Transcribed Conversation**")
            # placeholder = st.empty()
            with st.container(border=True, height=340):
                if transcribe:  
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])


if __name__ == "__main__":
    main(debug=False)
