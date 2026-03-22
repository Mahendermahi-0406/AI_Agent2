print("🚀 travel.py freshly loaded")

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq


# -------------------- AGENTS (LLM HOLDER) --------------------
class TripAgents:
    def __init__(self):
        print("🔥 TripAgents __init__ CALLED")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7
        )
        print("USING LLM:", type(self.llm))

    # These methods remain ONLY for compatibility
    def city_selector_agent(self):
        return self.llm

    def local_expert_agent(self):
        return self.llm

    def travel_planner_agent(self):
        return self.llm

    def budget_manager_agent(self):
        return self.llm


# -------------------- TASKS (PROMPT FACTORY) --------------------
class Triptasks:
    def __init__(self):
        pass

    def city_selection_task(self):
        return PromptTemplate(
            input_variables=["travel_type", "interests", "season", "visa_required"],
            template="""
You are a City Selection Expert.

Travel Type: {travel_type}
Interests: {interests}
Season: {season}
Visa Requirement: {visa_required}

RULES:
- If Visa Requirement is "No", recommend ONLY cities within India.
- If Visa Requirement is "Yes", recommend ONLY international cities.
- If Visa Requirement is "Not Sure", recommend both Indian and international cities.

Recommend 3 cities with short explanations.
Return in markdown bullet points.
"""
        )

    def city_research_task(self):
        return PromptTemplate(
            input_variables=["city"],
            template="""
You are a Local Destination Expert.

Provide detailed insights for {city}:
- Top attractions
- Local cuisine
- Culture & etiquette
- Transport tips

Return in markdown format.
"""
        )

    def itinerary_creation_task(self):
        return PromptTemplate(
            input_variables=["city", "duration", "interests"],
            template="""
You are a Professional Travel Planner.

Create a {duration}-day itinerary for {city}.
Interests: {interests}

Return a day-by-day markdown itinerary.
"""
        )

    def budget_planning_task(self):
        return PromptTemplate(
            input_variables=["city", "budget"],
            template="""
You are a Travel Budget Specialist.

Create a budget plan for {city}.
Budget Range: {budget}

Include accommodation, food, transport, activities.
Return in markdown.
"""
        )


# -------------------- CREW (RUNNABLE ORCHESTRATOR) --------------------
class TripCrew:
    def __init__(self, tasks, agents, inputs=None):
        self.inputs = inputs or {}
        self.llm = agents.city_selector_agent()
        self.tasks = tasks

    def run(self):
        # ---- City Selection ----
        city_chain = RunnableSequence(
            self.tasks.city_selection_task(),
            self.llm
        )
        city_selection = city_chain.invoke({
            "travel_type": self.inputs["travel_type"],
            "interests": self.inputs["interests"],
            "season": self.inputs["season"],
            "visa_required": self.inputs["visa_required"]
        }).content

        # Select first AI-recommended city
        city = city_selection.split("\n")[0]

        # ---- City Research ----
        research_chain = RunnableSequence(
            self.tasks.city_research_task(),
            self.llm
        )
        city_research = research_chain.invoke({
            "city": city
        }).content

        # ---- Itinerary ----
        itinerary_chain = RunnableSequence(
            self.tasks.itinerary_creation_task(),
            self.llm
        )
        itinerary = itinerary_chain.invoke({
            "city": city,
            "duration": self.inputs["duration"],
            "interests": self.inputs["interests"]
        }).content

        # ---- Budget ----
        budget_chain = RunnableSequence(
            self.tasks.budget_planning_task(),
            self.llm
        )
        budget = budget_chain.invoke({
            "city": city,
            "budget": self.inputs["budget"]
        }).content

        return {
            "city_selection": city_selection,
            "city_research": city_research,
            "itinerary": itinerary,
            "budget": budget
        }
