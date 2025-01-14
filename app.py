import openai  # Install with `pip install openai`
from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import os
import matplotlib
import json

matplotlib.use('Agg')  # Use non-GUI backend for macOS compatibility

# Set up Flask app
app = Flask(__name__)

# OpenAI API key
openai.api_key = "had to remove secret key"


def plot_stock_prices(hist, ticker):
    """
    Plots stock prices for the last 6 months with enhanced formatting.

    :param hist: Pandas DataFrame containing stock history
    :param ticker: Stock ticker symbol
    """
    plt.figure(figsize=(10, 6))
    plt.plot(hist.index, hist['Close'], marker='o', linestyle='-', color='blue', label=f'{ticker.upper()} Closing Price')

    # Enhancing the plot
    plt.title(f"{ticker.upper()} Stock Prices (Last 6 Months)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="best", fontsize=10)

    # Save the plot
    plot_path = os.path.join('static', 'stock_plot.png')
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path


def preprocess_sentiment(sentiment):
    """
    Preprocesses the sentiment data to make it suitable for rendering in the template.

    :param sentiment: Dictionary containing AI sentiment response.
    :return: Preprocessed dictionary with formatted values.
    """
    for key, value in sentiment.items():
        # If the value is a string and looks like a JSON or list, try parsing it
        if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
            try:
                sentiment[key] = json.loads(value)  # Parse JSON-like string
            except json.JSONDecodeError:
                pass  # Leave as-is if it's not valid JSON
    return sentiment


def get_stock_sentiment(ticker, hist):
    """
    Generates a sentiment analysis prompt based on the latest stock data and general information about the company.

    :param ticker: Stock ticker
    :param hist: Pandas DataFrame containing stock history
    :return: Dictionary containing structured sentiment analysis
    """
    # Get the latest closing price
    latest_price = hist['Close'].iloc[-1]

    # Create a detailed conversation for GPT
    messages = [
        {
            "role": "system",
            "content": "You are a financial assistant who provides structured, easy-to-understand sentiment analyses for stocks."
        },
        {
            "role": "user",
            "content": f"""
            Provide a structured sentiment analysis for {ticker} stock. Include:
            - A brief description of the company.
            - Current stock trends, including the latest closing price (${latest_price:.2f}).
            - Key factors that may influence the stock price, such as news, industry trends, or historical performance.
            Return the response in JSON format with keys 'Company Description', 'Current Stock Trends', and 'Key Factors Influencing Stock Price'.
            """
        }
    ]

    # Fetch sentiment from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you prefer and have access
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    # Parse the JSON-like response from the AI
    sentiment = response.choices[0].message['content'].strip()

    try:
        # Convert the AI's response into a dictionary
        sentiment_dict = json.loads(sentiment)
    except json.JSONDecodeError:
        # Fallback: if the response isn't valid JSON, return it as a single string
        sentiment_dict = {"AI Response": sentiment}

    return sentiment_dict


@app.route('/', methods=['POST', 'GET'])
def index():
    stock_data = None
    sentiment = None
    if request.method == 'POST':
        ticker = request.form.get('stock', '').strip()
        if not ticker:
            return "No ticker provided.", 400
        try:
            # Fetch stock data for the last 6 months
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")  # Fetch 6 months of data
            if hist.empty:
                return "No data found for the provided ticker.", 400
            
            # Plot stock graph
            stock_data = plot_stock_prices(hist, ticker)

            # Get sentiment analysis based on stock trends and company overview
            sentiment = get_stock_sentiment(ticker, hist)
            sentiment = preprocess_sentiment(sentiment)  # Preprocess the data

        except Exception as e:
            return f"Error: {e}", 500

    return render_template('index.html', stock_data=stock_data, sentiment=sentiment)


if __name__ == "__main__":
    app.run(debug=True, port=5001)

#to fixe ticker name gpt problem coud find the name of the company with ticker, save it in another variable, than pass that to the main question area
