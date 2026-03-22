import os
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.pop("OPENAI_API_KEY", None)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from travel import TripAgents, Triptasks, TripCrew


def main():
    st.title("🤖 AI Travel Planning Assistant")

    with st.sidebar:
        st.header("Trip Preferences")

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

        budget = st.selectbox(
            "Budget Range",
            ["1000-5000", "5000-15000", "15000-30000", "Luxury"]
        )

        if st.button("Generate Travel Plan"):
            inputs = {
                "travel_type": travel_type,
                "interests": interests,
                "season": season,
                "duration": duration,
                "budget": budget
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
