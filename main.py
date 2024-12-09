import streamlit as st
import requests
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os


groq_api = os.getenv("GROQ_API_KEY")
play_ht_api = os.getenv("PLAY_HT_API")
play_ht_user_id = os.getenv("PLAY_HT_USER_ID")


def main():

    st.markdown("Ask A Question:")
    question = st.audio_input("Record a voice message")

    client = Groq()

    if question:

        transcription = client.audio.transcriptions.create(
                        file=("question.wav", question),
                        model="whisper-large-v3-turbo",
                        response_format="json")
        
        text_question = transcription.text 

        # st.markdown(f"Question: {text_question}")

        if text_question:
            llm = ChatGroq(
                        model="llama-3.1-70b-versatile",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2
                        )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{question}")
                ]
            )

            ai_response = llm.invoke(prompt.invoke({"question": text_question}))
            text_response = ai_response.content

            # if text_response:
            #     st.write(f"Answer: {text_response}")

            url = "https://api.play.ht/api/v2/tts/stream"

            payload = {
                "voice": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "output_format": "mp3",
                "text": text_response,
                "voice_2": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "voice_engine": "PlayDialog",
                "turn_prefix_2": "Town Mouse:",
                "turn_prefix": "Country Mouse:"
            }
            headers = {
                "accept": "audio/mpeg",
                "content-type": "application/json",
                "AUTHORIZATION": play_ht_api,
                "X-USER-ID": play_ht_user_id
            }

            response = requests.post(url, json=payload, headers=headers)

            if response:
                st.markdown("Play AI Response:")
                st.audio(response.content, "audio/wav")

if __name__ == "__main__":
    main()
