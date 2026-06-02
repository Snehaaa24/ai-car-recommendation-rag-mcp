from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from chat_memory import chat_history

from rag import search_cars
from model import check_price


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
 
    if (
        len(results) > 0
        and "message" in results[0]
    ):
        return results[0]["message"]

    if not results:
        return "No matching cars found."

    formatted_results = []

    for car in results:

        formatted_results.append(
            f"""
Car: {car['make']} {car['model']}
Year: {car['year']}
Price: ₹{car['price_lakhs']}L
City: {car['city']}
Fuel: {car['fuel']}
Transmission: {car['transmission']}
Kms Driven: {car['kms_driven']}
"""
        )

    best_car = results[0]

    valuation = check_price(
        best_car["make"],
        best_car["model"],
        best_car["year"],
        best_car["fuel"],
        best_car["transmission"],
        best_car["kms_driven"],
        best_car["price_lakhs"]
    )

    valuation_text = f"""
Best Match Valuation

Estimated Price: ₹{valuation['estimated_price']:.2f}L
Listed Price: ₹{best_car['price_lakhs']}L
Verdict: {valuation['verdict']}
"""

    return "\n".join(formatted_results) + "\n" + valuation_text


# ---------------------------------
# TOOL 2 - PRICE CHECK
# ---------------------------------
@tool
def price_checker(car_details: str) -> str:
    """
Use this tool when user asks:

- Is this car fairly priced?
- Is this overpriced?
- Valuation
- Worth buying?

Input:
Make,Model,Year,Fuel,Transmission,KmsDriven,ListedPrice
"""

    print("\n====================")
    print("TOOL CALLED -> PRICE CHECK")
    print("INPUT:", car_details)
    print("====================\n")

    try:

        make, model, year, fuel, transmission, kms, listed_price = (
            car_details.split(",")
        )

        result = check_price(
            make,
            model,
            int(year),
            fuel,
            transmission,
            int(kms),
            float(listed_price)
        )

        return f"""
Estimated Price: ₹{result['estimated_price']:.2f}L
Verdict: {result['verdict']}
Listed Price: ₹{listed_price}L
"""

    except Exception as e:

        return f"Error: {str(e)}"


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

    previous_context = "\n".join(chat_history)

    full_query = f"""
    Previous Conversation:
    {previous_context}

    Current User Query:
    {query}
    """

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": full_query
                }
            ]
        }
    )

    answer = response["messages"][-1].content

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")

    return answer


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