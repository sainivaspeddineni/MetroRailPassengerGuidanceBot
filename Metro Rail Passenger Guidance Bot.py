# pip install streamlit google-genai

import streamlit as st
from google import genai
from google.genai import types

# -------------------------------
# API KEY (LOCAL TESTING ONLY)
# ⚠️ Regenerate after testing
# -------------------------------
GEMINI_API_KEY = "AIzaSyDDPNtgxNtYIRxw4OeyM3xjORajv6EQRsQ"

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Metro Guidance Bot", page_icon="🚇")
st.title("🚇 Metro Rail Passenger Guidance Bot")
st.caption("Informational assistance only • No ticketing or live operations")

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.text_area(
    "Ask a metro-related question:",
    placeholder="Explain metro ticket types"
)

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("Ask Bot"):
    if not user_input.strip():
        st.error("Please enter a question.")
        st.stop()

    # -------------------------------
    # DOMAIN FILTER (STRICT)
    # -------------------------------
    metro_keywords = [
        "metro", "train", "ticket", "platform", "station",
        "security", "entry", "exit", "gate", "coach",
        "passenger", "travel", "check", "rules"
    ]

    if not any(word in user_input.lower() for word in metro_keywords):
        st.warning(
            "I can only assist with metro rail–related information. "
            "Please ask a metro-related question."
        )
        st.stop()

    try:
        # -------------------------------
        # GEMINI CLIENT
        # -------------------------------
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-3-flash-preview"

        # -------------------------------
        # SYSTEM + USER PROMPTS
        # -------------------------------
        contents = [
            # SYSTEM PROMPT
            types.Content(
                role="system",
                parts=[
                    types.Part.from_text(
                        text=(
                            "You are a Metro Rail Passenger Guidance Bot.\n\n"

                            "You must answer ONLY questions directly related to metro rail travel, "
                            "including ticket types, security checks, entry and exit gates, "
                            "platform rules, passenger etiquette, and safety instructions.\n\n"

                            "If a user asks ANY non-metro-related question "
                            "(sports, movies, politics, general topics), "
                            "respond ONLY with:\n"
                            "\"I can only assist with metro rail–related information. "
                            "Please ask a metro-related question.\"\n\n"

                            "You are strictly limited to informational responses only. "
                            "You must NOT sell tickets, handle bookings, show fares, "
                            "give real-time schedules, or perform operational actions.\n\n"

                            "Use simple, commuter-friendly language. "
                            "Never assume city-specific rules. "
                            "Never hallucinate operational details. "
                            "Always prioritize passenger safety and public transit compliance."
                        )
                    )
                ],
            ),

            # USER PROMPT
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            ),
        ]

        # -------------------------------
        # GENERATION CONFIG
        # -------------------------------
        generate_content_config = types.GenerateContentConfig(
            temperature=0.25,
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH"
            )
        )

        # -------------------------------
        # RESPONSE STREAM
        # -------------------------------
        with st.spinner("Generating response..."):
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text

        st.success("Response:")
        st.write(response_text)

    except Exception as e:
        st.error(f"Error: {e}")
