from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from rag import search_cars
from model import check_price

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# HOME PAGE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# SEARCH ENDPOINT
@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    user_query: str = Form(...)
):

    try:

        results = search_cars(user_query)

        # Add AI price analysis
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

            car['predicted_price'] = price_result['estimated_price']
            car['valuation'] = price_result['verdict']

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "cars":results,
                "query":user_query
            }
        )

    except Exception as e:

        return HTMLResponse(
            content=f"Error: {str(e)}",
            status_code=500
        )


# PRICE CHECK API
@app.post("/price-check")
async def price_check(data: dict):

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