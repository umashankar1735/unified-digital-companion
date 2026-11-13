def load_env_variables():
    import os
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file

def validate_query(query):
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string.")
    return query.strip()

def format_response(data):
    return {
        "status": "success",
        "data": data
    }

def handle_error(error_message):
    return {
        "status": "error",
        "message": error_message
    }