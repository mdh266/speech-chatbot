import streamlit as st
from google.cloud import texttospeech
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


def main():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("Ask A Question:")
    question = st.audio_input("Click the microphone to start and stop recording")

    client = Groq()

    llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2)
    
    if question:
        transcription = client.audio.transcriptions.create(
                        file=("question.wav", question),
                        model="whisper-large-v3-turbo",
                        response_format="json")
        
        text_question = transcription.text 

        st.session_state.messages.append({"role": "human", "content": f"Human: {text_question}"})

        if text_question:
            
            history = tuplify(st.session_state.messages)
            answer = ask_question(llm=llm, history=history, question=text_question)

            st.session_state.messages.append({"role": "ai", "content": answer})
        
            client = texttospeech.TextToSpeechClient(client_options={"api_key": google_api})

            synthesis_input = texttospeech.SynthesisInput(text=answer)

            voice = texttospeech.VoiceSelectionParams(
                language_code="en-GB", name="en-GB-Journey-F"
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            if response:
                st.markdown("Play AI Response:")
                st.audio(response.audio_content, "audio/mp3")

if __name__ == "__main__":
    main()
