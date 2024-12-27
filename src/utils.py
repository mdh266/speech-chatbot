import streamlit as st
from google.cloud import texttospeech, texttospeech_v1, speech
from google.cloud import translate_v2 as translate
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from typing import Iterator, List, Dict, Tuple


lang_code_map = {
    "English": "en-US",
    "Hebrew": "he-IL",
    "French": "fr-FR"
}


tts_name_map = {
    "English": "en-US-Journey-F",
    "Hebrew": "he-IL-Standard-A",
    "French": "fr-FR-Standard-A"
}


def tuplify(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(d['role'], d['content']) for d in history]


@st.cache_data
def ask_question(
    history: List[Tuple[str, str]], 
    question: str,
    ai_language: str,
    google_api: str
) -> str:
    llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful teacher having a conversation with a student."),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ]
    )

    chain = prompt | llm 
    
    response = chain.invoke({
           "history": history,
           "question": question
       })
    
    answer = response.content
    translated_response = answer

    if ai_language != "English":
        translated_response = translate_text(
                                    target="he", 
                                    text=answer, 
                                    google_api=google_api)

    return answer, translated_response


@st.cache_data
def translate_text(target: str, text: str, google_api: str) -> Dict[str, str]:

    translate_client = translate.Client(client_options={"api_key": google_api})

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result.get("translatedText")


@st.cache_data
def text_to_speech(
    ai_language: str,
    google_api: str,
    translated_answer: str
) -> texttospeech_v1.types.cloud_tts.SynthesizeSpeechResponse:
    
    tts = texttospeech.TextToSpeechClient(client_options={"api_key": google_api})

    synthesis_input = texttospeech.SynthesisInput(text=translated_answer)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code_map[ai_language], name=tts_name_map[ai_language])

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response


@st.cache_data
def speech_to_text(
    human_language: str,
    google_api: str,
    question: st.runtime.uploaded_file_manager.UploadedFile
) -> str:

    stt = speech.SpeechClient(client_options={"api_key": google_api})

    stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=lang_code_map[human_language])

    audio = speech.RecognitionAudio(content=question.getvalue())

    response = stt.recognize(config=stt_config, audio=audio)

    text_question = response.results.pop().alternatives[0].transcript

    english_question = text_question

    if human_language != "English":
        english_question = translate_text(target="en",
                                          text=text_question, 
                                          google_api=google_api)
        
    return text_question, english_question


def create_audio():
    return st.audio_input("Click the microphone to start and stop recording")


def clear_session():
    st.session_state.messages = []
    st.session_state.english_messages = []
    st.session_state.transcribe = None
