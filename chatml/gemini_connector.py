# chatml/gemini_connector.py
import os
import google.generativeai as genai
from rich.console import Console

from chatml.function_schemas import tools


class GeminiConnector:
    """
    Klasa odpowiedzialna za komunikację z API Google Gemini.
    Obsługuje stanowe sesje czatu i function calling.
    """

    def __init__(self):
        self.model = None
        self.chat_session = None
        self.console = Console()

    def configure(self):
        """
        Konfiguruje API Gemini, próbując najpierw domyślnych poświadczeń (ADC),
        a następnie wracając do klucza API ze zmiennej środowiskowej.
        """
        try:
            # 1. Spróbuj użyć domyślnych poświadczeń (ADC)
            next(genai.list_models())
            self.console.print(
                "[green]Gemini API configuration (via Application Default Credentials) successful.[/green]"
            )
            return True
        except Exception:
            self.console.print(
                "[yellow]Default credentials (ADC) not found. Attempting to use GEMINI_API_KEY...[/yellow]"
            )
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.console.print(
                    "[bold red]Error: Neither default credentials nor GEMINI_API_KEY environment variable found.[/bold red]"
                )
                return False

            try:
                genai.configure(api_key=api_key)
                next(genai.list_models())
                self.console.print(
                    "[green]Gemini API configuration (via API Key) successful.[/green]"
                )
                return True
            except Exception as e:
                self.console.print(
                    f"[bold red]Error configuring Gemini API using key: {e}[/bold red]"
                )
                return False

    def get_model(self):
        """
        Inicjalizuje i zwraca model generatywny, dynamicznie wybierając dostępny model.
        Tworzy również sesję czatu, jeśli jeszcze nie istnieje.
        """
        if self.model:
            return self.model

        try:
            available_models = list(genai.list_models())
            chat_models = [
                m
                for m in available_models
                if "generateContent" in m.supported_generation_methods
            ]

            if not chat_models:
                self.console.print(
                    "[bold red]Error: No model supporting 'generateContent' found.[/bold red]"
                )
                return None

            selected_model = next(
                (m for m in chat_models if "flash" in m.name), chat_models[0]
            )
            model_name = selected_model.name
            self.console.print(f"[blue]Using model: {model_name}[/blue]")

            self.model = genai.GenerativeModel(model_name, tools=list(tools.values()))

            # Inicjalizacja sesji czatu
            self.chat_session = self.model.start_chat(history=[])

            return self.model

        except Exception as e:
            self.console.print(f"[bold red]Error selecting model: {e}[/bold red]")
            return None

    async def send_message(self, message: str):
        """
        Wysyła wiadomość do modelu w ramach aktywnej sesji czatu.
        """
        if not self.model:
            self.get_model()

        if not self.chat_session:
            return None

        try:
            response = await self.chat_session.send_message_async(message)
            return response
        except Exception as e:
            self.console.print(
                f"[bold red]Error communicating with Gemini: {e}[/bold red]"
            )
            return None

    async def send_tool_response(self, function_name: str, result: dict):
        """
        Wysyła wynik wykonania funkcji (tool output) z powrotem do modelu.
        Model powinien na tej podstawie wygenerować końcową odpowiedź tekstową.
        """
        if not self.chat_session:
            return None

        try:
            # Konstrukcja odpowiedzi dla narzędzia (zgodnie z API Gemini)
            tool_response = {
                "function_response": {"name": function_name, "response": result}
            }

            # Wysyłamy wynik funkcji jako kolejną wiadomość w sesji
            response = await self.chat_session.send_message_async(
                genai.protos.Content(parts=[genai.protos.Part(tool_response)])
            )
            return response
        except Exception as e:
            self.console.print(
                f"[bold red]Error sending function result to Gemini: {e}[/bold red]"
            )
            return None
