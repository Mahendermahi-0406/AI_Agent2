import os
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.pop("OPENAI_API_KEY", None)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from travel_langchain import TripAgents, Triptasks, TripCrew


def main():
    # ---------- Title at top ----------
    st.markdown(
        "<h1 style='text-align: center;'>🤖 AI Travel Planning Assistant</h1>",
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # ---------- Centered layout ----------
    left, center, right = st.columns([1, 10, 1])

    with center:
        st.subheader("Trip Preferences")

        travel_type = st.selectbox(
            "Travel Type",
            ["Leisure", "Business", "Adventure", "Cultural", "Wellness"]
        )

        interests = st.multiselect(
            "Interests",
            ["History", "Food", "Nature", "Art", "Shopping", "Nightlife", "Wildlifes", "Museums"]
        )

        season = st.selectbox(
            "Season",
            ["Summer", "Winter", "Spring", "Fall"]
        )

        duration = st.slider("Trip Duration (days)", 1, 15, 7)

        companions = st.selectbox(
            "Travel Companions",
            ["Solo", "Couple", "Family", "Friends", "Colleagues"]
        )

        accommodation = st.selectbox(
            "Accommodation Type",
            ["Budget Hotel", "Mid-range Hotel", "Luxury Hotel", "Resort", "Hostel", "Homestay"]
        )

        transport = st.selectbox(
            "Preferred Transportation",
            ["Public Transport", "Car Rental", "Flights + Local Transport", "Any"]
        )

        food_preference = st.selectbox(
            "Food Preference",
            ["No Preference", "Vegetarian", "Vegan", "Non-Vegetarian", "Local Cuisine Focus"]
        )

        visa_required = st.radio(
            "Visa Requirement",
            ["Yes", "No", "Not Sure"]
        )

        budget = st.selectbox(
            "Budget Range",
            ["2000-5000", "5000-10000", "10000-15000", "15000-20000","20000-30000","Luxury"]
        )

        st.write("")  # spacing

        if st.button("Generate Travel Plan", use_container_width=True):
            inputs = {
                "travel_type": travel_type,
                "interests": interests,
                "season": season,
                "duration": duration,
                "budget": budget,
                "companions": companions,
                "accommodation": accommodation,
                "transport": transport,
                "food_preference": food_preference,
                "visa_required": visa_required
            }

            st.write("✅ Button clicked")
            agents = TripAgents()
            st.write("✅ TripAgents created")
            tasks = Triptasks()

            with st.spinner("🤖 AI Agents are working on your perfect trip..."):
                try:
                    crew_output = TripCrew(
                        tasks=tasks,
                        agents=agents,
                        inputs=inputs
                    ).run()

                    city_selection = crew_output.get("city_selection", "❌ No city selection found.")
                    city_research = crew_output.get("city_research", "❌ No city research found.")
                    itinerary = crew_output.get("itinerary", "❌ No itinerary generated.")
                    budget_breakdown = crew_output.get("budget", "❌ No budget breakdown available.")

                    st.subheader("Your AI-Generated Travel Plan")

                    with st.expander("Recommended Cities"):
                        st.markdown(city_selection)

                    with st.expander("Destination Insights"):
                        st.markdown(city_research)

                    with st.expander("Detailed Itinerary"):
                        st.markdown(itinerary)

                    with st.expander("Budget Breakdown"):
                        st.markdown(budget_breakdown)

                    st.success("✅ Trip planning completed! Enjoy your journey!")

                except Exception as e:
                    st.error(f"An error occurred while processing the results: {e}")


if __name__ == "__main__":
    main()
