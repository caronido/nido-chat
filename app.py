import os
import logging
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

logging.debug(f"SLACK_BOT_TOKEN: {SLACK_BOT_TOKEN}")
logging.debug(f"SLACK_SIGNING_SECRET: {SLACK_SIGNING_SECRET}")
if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
    raise ValueError("Missing environment variables")
# Initialize Slack app
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

from slack_bot import app as bolt_app

# Initialize Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(bolt_app)

# Endpoint to receive Slack events
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    logging.debug(f"Received event: {data}")

    # Handle URL verification
    if "type" in data and data["type"] == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    # Handle other Slack events
    response = handler.handle(request)
    logging.debug(f"Response: {response}")
    return response

# Health check endpoint
@flask_app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

# Add a simple route for testing
@flask_app.route("/")
def home():
    return "Hello, this is the home page."

if __name__ == "__main__":
    flask_app.run(port=8000)
