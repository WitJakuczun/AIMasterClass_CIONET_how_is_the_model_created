def create_gemma_prompt(row):
    """Creates a prompt for the Gemma model from a data row.
    Expected to have 'Sentence' and 'Sentiment' columns.
    """
    # row is a dict when using map_elements(..., return_dtype=...) or similar in Polars
    # or a named tuple/row object depending on how it's called.

    # Assuming row is a dictionary-like object (e.g. from iter_rows(named=True))
    sentence = row.get("Sentence", "")
    sentiment = row.get("Sentiment", "")

    prompt = f"<start_of_turn>user\nOceń sentyment poniższego zdania jako negatywny, neutralny lub pozytywny. Odpowiedz tylko jednym słowem.\nZdanie: {sentence}<end_of_turn>\n<start_of_turn>model\n{sentiment}<end_of_turn>"
    return prompt
