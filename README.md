# About

NLP query engine for big data.\
Using Groq with Llama to generate a PySpark job from natural language queries. This uses NYC taxi data from Kaggle and loads into a PySpark DataFrame, takes a user query in plain English, converts it into PySpark code using a structured prompt, executes it on a Spark cluster, and returns an optimized result.

## Prerequisites

1. Ensure that Spark libraries are installed.
2. Generate an API key for Groq.
