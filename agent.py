from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from rag import search_cars


# ---------------------------------
# TOOL 1 - CAR SEARCH
# ---------------------------------
@tool
def car_search(query: str) -> str:
    """
    Search used cars based on user preferences such as
    budget, SUV, automatic, diesel, city, family car, etc.
    """

    print("\n====================")
    print("TOOL CALLED -> CAR SEARCH")
    print("QUERY:", query)
    print("====================\n")

    results = search_cars(query)

    return str(results)


# ---------------------------------
# TOOL 2 - PRICE CHECK
# ---------------------------------
@tool
def price_checker(car_details: str) -> str:
    """
    Check whether a car is overpriced,
    underpriced, or fairly priced.
    """

    print("\n====================")
    print("TOOL CALLED -> PRICE CHECKER")
    print("INPUT:", car_details)
    print("====================\n")

    return (
        "Price checking tool available. "
        "Can evaluate car pricing."
    )


# ---------------------------------
# LOCAL LLM
# ---------------------------------
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ---------------------------------
# AGENT
# ---------------------------------
agent = create_react_agent(
    llm,
    tools=[car_search, price_checker]
)


# ---------------------------------
# MAIN FUNCTION
# ---------------------------------
def ask_agent(query):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return response["messages"][-1].content


# ---------------------------------
# TEST LOOP
# ---------------------------------
if __name__ == "__main__":

    print("\nMCP Agent Started")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Enter query: ")

        if query.lower() == "exit":
            break

        try:

            answer = ask_agent(query)

            print("\nAgent Response:")
            print(answer)

        except Exception as e:

            print("\nERROR:")
            print(str(e))