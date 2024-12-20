import streamlit as st
from google.cloud import texttospeech, speech
from google.cloud import translate_v2 as translate
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from typing import Iterator, List, Dict, Tuple


groq_api = os.getenv("GROQ_API_KEY")
google_api = os.getenv("GOOGLE_API_KEY")


def tuplify(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(d['role'], d['content']) for d in history]


def ask_question(llm: ChatGroq, history: List[Tuple[str, str]], question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ]
    )

    new_prompt = prompt.invoke(
           {
           "history": history,
           "question": question
       }
    )

    response = llm.invoke(new_prompt)
    
    return response.content

def translate_text(target: str, text: str) -> dict:

    translate_client = translate.Client(client_options={"api_key": google_api})

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)


    return result


def main():
    def clear_session():
        st.session_state.messages = []
        st.session_state.english_messages = []
        st.session_state.transcribe = False
        question = None
        text_question = None
        english_question = None
        answer = None
        translated_answer = None

    
    if "messages" not in st.session_state:
        clear_session()


    llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2)
    
    stt = speech.SpeechClient(client_options={"api_key": google_api})
    tts = texttospeech.TextToSpeechClient(client_options={"api_key": google_api}) 
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.session_state.human_language = st.selectbox("Human Language:", ("English", "Hebrew"))
        st.session_state.ai_language = st.selectbox("AI Language:", ("English", "Hebrew"))
        st.session_state.transcribe = st.checkbox("Transcribe Conversation", key="disabled")
    
        st.markdown("Ask A Question:")
        question = st.audio_input("Click the microphone to start and stop recording")

        if question:
            if st.session_state.human_language == "English":
                stt_language_code="en-US" 
            else:
                stt_language_code="he-IL"

            stt_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code=stt_language_code)

            audio = speech.RecognitionAudio(content=question.getvalue())

            response = stt.recognize(config=stt_config, audio=audio)

            text_question = response.results.pop().alternatives[0].transcript

            st.session_state.messages.append({"role": "human", 
                                              "content": f"Human: {text_question}"})
            
            if st.session_state.human_language == "Hebrew":
                english_question = translate_text("en", text_question ).get("translatedText")
            else:
                english_question = text_question

            if english_question:
                history = tuplify(st.session_state.english_messages)

                st.session_state.english_messages.append({"role": "human", 
                                                          "content": f"Human: {english_question}"})
                
                answer = ask_question(llm=llm, history=history, question=english_question)
                
                st.session_state.english_messages.append({"role": "ai", "content": f"Bot: {answer}"})

                if st.session_state.ai_language == "Hebrew":
                    translated_answer = translate_text("he", answer ).get("translatedText")
                    tts_language_code = "en-GB"
                    tts_name = "en-GB-Journey-F"
                else:
                    translated_answer = answer
                    tts_language_code = "he-IL"
                    tts_name = "he-IL-Standard-A"

                st.session_state.messages.append({"role": "ai", 
                                                  "content": f"Bot: {translated_answer}"})
                    

                synthesis_input = texttospeech.SynthesisInput(text=translated_answer)

                voice = texttospeech.VoiceSelectionParams(
                    language_code=tts_language_code, name=tts_name)

                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )

                response = tts.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                if response:
                    st.markdown("Play AI Response:")
                    st.audio(response.audio_content, "audio/mp3")

        st.button("Clear Conversation", on_click=clear_session) 

                
        with col2:
            if st.session_state.transcribe:
                    # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                         st.markdown(message["content"])

            # css='''
            # <style>
            #     section.main>div {
            #         padding-bottom: 1rem;
            #     }
            #     [data-testid="column"]>div>div>div>div>div {
            #         overflow: auto;
            #         height: 70vh;
            #     }
            # </style>
            # '''

            # st.markdown(css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
