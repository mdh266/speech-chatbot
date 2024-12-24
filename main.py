import streamlit as st
from google.cloud import texttospeech, texttospeech_v1, speech
from google.cloud import translate_v2 as translate
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from typing import Iterator, List, Dict, Tuple
from dataclasses import dataclass


def tuplify(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(d['role'], d['content']) for d in history]


@st.cache_data
def ask_question(history: List[Tuple[str, str]], question: str) -> str:
    llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ]
    )

    chain = prompt | llm 
    
    response = chain.invoke({
           "history": history,
           "question": question
       })
    
    return response.content


@st.cache_data
def translate_text(target: str, text: str, google_api: str) -> Dict[str, str]:

    translate_client = translate.Client(client_options={"api_key": google_api})

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result


@st.cache_data
def text_to_speech(
    language_code: str,
    name: str,
    google_api: str,
    translated_answer: str
) -> texttospeech_v1.types.cloud_tts.SynthesizeSpeechResponse:
    
    tts = texttospeech.TextToSpeechClient(client_options={"api_key": google_api})

    synthesis_input = texttospeech.SynthesisInput(text=translated_answer)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=name)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response


@st.cache_data
def speech_to_text(
    language_code: str,
    google_api: str,
    question: st.runtime.uploaded_file_manager.UploadedFile
) -> str:

    stt = speech.SpeechClient(client_options={"api_key": google_api})

    stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code)

    audio = speech.RecognitionAudio(content=question.getvalue())

    response = stt.recognize(config=stt_config, audio=audio)

    text_question = response.results.pop().alternatives[0].transcript

    return text_question


def create_audio():
    return st.audio_input("Click the microphone to start and stop recording")


def clear_session():
    st.session_state.messages = []
    st.session_state.english_messages = []
    st.session_state.transcribe = None
    st.session_state.text_question = None
    st.session_state.question = None


def main(debug: bool = False):
    if debug:
        from dotenv import load_dotenv
        load_dotenv()

    groq_api = os.getenv("GROQ_API_KEY")
    google_api = os.getenv("GOOGLE_API_KEY")

    if "messages" not in st.session_state:
        clear_session()
    
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.session_state.human_language = st.selectbox("Human Language:", ("English", "Hebrew"))
        st.session_state.ai_language = st.selectbox("AI Language:", ("English", "Hebrew"))
        st.session_state.transcribe = st.checkbox("Transcribe Conversation", key="disabled")
    
        st.markdown("Ask A Question:")
        st.session_state.question = create_audio()

        if st.session_state.question:
            if st.session_state.human_language == "English":
                stt_language_code="en-US" 
            else:
                stt_language_code="he-IL"

            st.session_state.text_question = speech_to_text(
                language_code=stt_language_code,
                google_api=google_api,
                question=st.session_state.question)
            
            st.session_state.messages.append({
                "role": "human", 
                "content": f"Human: {st.session_state.text_question}"
            })
            
            if st.session_state.human_language == "Hebrew":
                english_question = translate_text(target="en", 
                                                text=st.session_state.text_question, 
                                                google_api=google_api).get("translatedText")
            else:
                english_question = st.session_state.text_question

            if english_question:
                history = tuplify(st.session_state.english_messages)

                st.session_state.english_messages.append({
                    "role": "human", 
                    "content": f"Human: {english_question}"
                })
                
                answer = ask_question(history=history, question=english_question)
                
                st.session_state.english_messages.append({"role": "ai", 
                                                        "content": f"Bot: {answer}"})

                if st.session_state.ai_language == "Hebrew":
                    translated_answer = translate_text(target="he", 
                                                    text=answer, 
                                                    google_api=google_api).get("translatedText")
                    
                    tts_language_code = "en-US"
                    tts_name = "en-US-Journey-A"
                else:
                    translated_answer = answer
                    tts_language_code = "he-IL"
                    tts_name = "he-IL-Standard-A"

                st.session_state.messages.append({
                    "role": "ai", 
                    "content": f"Bot: {translated_answer}"
                })


                response = text_to_speech(language_code=tts_language_code,
                                        name=tts_name,
                                        google_api=google_api,
                                        translated_answer=translated_answer)

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
    main(debug=True)
