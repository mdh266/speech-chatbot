import os 
import streamlit as st
from utils import (text_to_speech,
                   speech_to_text,
                   ask_question,
                   create_audio,
                   tuplify,
                   clear_session,
                   lang_code_map)


def main(debug: bool = False):
    if debug:
        from dotenv import load_dotenv
        load_dotenv()

    groq_api = os.getenv("GROQ_API_KEY")
    google_api = os.getenv("GOOGLE_API_KEY")

    if "messages" not in st.session_state:
        clear_session()
    

    st.markdown("# A Serverless Multimodal Chatbot")
    st.markdown("Have a conversation with an Artifical Intelligence Bot in English, Hebrew or French")

    col1, col2 = st.columns(2, gap="large")

    response = None
    transcribe = None

    with col1:
        
        with st.form("conversation"):
            st.markdown("### Submit A Question")
            
            languages = tuple(lang_code_map.keys())
            human_language = st.selectbox("Human Language:", languages)
            ai_language = st.selectbox("Bot Language:", languages)
            transcribe = st.checkbox("Transcribe Conversation")
    
            question = create_audio()
            submitted = st.form_submit_button("Submit")

        if submitted and question:

            text_question = speech_to_text(human_language=human_language,
                                           google_api=google_api,
                                           question=question)
            
            st.session_state.messages.append({
                "role": "human", 
                "content": f"Human: {text_question}"
            })

            history = tuplify(st.session_state.messages)
            
            answer = ask_question(history=history, 
                                  question=text_question,
                                  human_language=human_language,
                                  ai_language=ai_language)

            st.session_state.messages.append({
                "role": "ai", 
                "content": f"Bot: {answer}"
            })

            response = text_to_speech(
                                ai_language=ai_language,
                                google_api=google_api,
                                translated_answer=answer)

        st.button("Clear Conversation", on_click=clear_session) 

        with col2:

            st.markdown("**Bot's Last Response**")    
            with st.container(border=True, height=90):                    
                if response:
                        st.audio(response.audio_content, "audio/mp3")

            st.markdown("**Transcribed Conversation**")
            with st.container(border=True, height=340):
                if transcribe:  
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])


if __name__ == "__main__":
    main(debug=True)
