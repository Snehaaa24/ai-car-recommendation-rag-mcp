from chat_memory import memory

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from rag import search_cars
from model import check_price

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# ---------------------------------
# HOME PAGE
# ---------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ---------------------------------
# SEARCH ENDPOINT
# ---------------------------------
@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    user_query: str = Form(...)
):

    try:

        # -------------------------
        # Load Previous Chat Memory
        # -------------------------
        history = memory.load_memory_variables({})

        previous_context = history.get(
            "history",
            ""
        )

        # -------------------------
        # Combine Old + New Query
        # -------------------------
        full_query = f"{previous_context} {user_query}"
        print("\n===================")
        print("PREVIOUS CONTEXT:", previous_context)
        print("CURRENT QUERY:", user_query)
        print("FULL QUERY:", full_query)
        print("===================\n")

        # -------------------------
        # Search Cars
        # -------------------------
        results = search_cars(full_query)

        # -------------------------
        # Add AI Price Estimation
        # -------------------------
        for car in results:

            price_result = check_price(
                car['make'],
                car['model'],
                int(car['year']),
                car['fuel'],
                car['transmission'],
                int(car['kms_driven']),
                float(car['price_lakhs'])
            )

            car['predicted_price'] = round(
                price_result['estimated_price'],
                2
            )

            car['valuation'] = price_result['verdict']

        # -------------------------
        # Save Conversation Memory
        # -------------------------
        memory.save_context(
            {"input": user_query},
            {"output": user_query}
        )

        # -------------------------
        # Render Frontend
        # -------------------------
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "cars": results,
                "query": user_query
            }
        )

    except Exception as e:

        return HTMLResponse(
            content=f"Error: {str(e)}",
            status_code=500
        )


# ---------------------------------
# PRICE CHECK API
# ---------------------------------
@app.post("/price-check")
async def price_check_api(data: dict):

    try:

        result = check_price(
            data['make'],
            data['model'],
            int(data['year']),
            data['fuel'],
            data['transmission'],
            int(data['kms_driven']),
            float(data['listed_price'])
        )

        return JSONResponse(content=result)

    except Exception as e:

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )