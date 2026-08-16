#!/usr/bin/env python3
"""
Vibe Matcher web UI.

This file is presentation only -- a thin Flask wrapper. All parsing,
validation, and the single AI call live in vibe_matcher.py and are reused
as-is via get_recommendations(), so the web UI and the CLI share exactly
one code path for turning a request into a validated result.

Run with:
    python app.py
then open http://127.0.0.1:5000
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic
from flask import Flask, render_template, request

import vibe_matcher as vm

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    result = None
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            error = "Please describe what you're looking for."
        else:
            try:
                result = vm.get_recommendations(query)
                result["total_dropped"] = sum(result["dropped"].values())
                result["dropped_detail"] = vm.dropped_detail_string(result["dropped"])
            except vm.MissingAPIKeyError:
                error = ("ANTHROPIC_API_KEY is not set on the server. "
                          "Copy .env.example to .env and add your key, then restart the app.")
            except anthropic.APIError as e:
                error = f"Error calling the Claude API: {e}"

    return render_template("index.html", query=query, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
