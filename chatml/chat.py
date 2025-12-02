import asyncio
import os
import readline
import atexit
from google.protobuf.struct_pb2 import Struct
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from chatml.gemini_connector import GeminiConnector
from chatml.dispatcher import FunctionDispatcher

HISTORY_FILE = os.path.expanduser("~/.chatmlops_history")

def to_dict(struct: Struct) -> dict:
    """Konwertuje obiekt Struct z protobuf na słownik Pythona."""
    result = {}
    for key, value in struct.items():
        if isinstance(value, Struct):
            result[key] = to_dict(value)
        elif isinstance(value, list):
            result[key] = [to_dict(item) if isinstance(item, Struct) else item for item in value]
        else:
            result[key] = value
    return result

async def process_model_response(response, connector, dispatcher, console):
    """
    Przetwarza odpowiedź modelu, obsługując rekurencyjnie wywołania funkcji.
    """
    if not response:
        console.print(Panel("[bold red]No response received from the model (response is None).[/bold red]", title="[bold red]Error[/bold red]"))
        return

    if not response.candidates:
        console.print(Panel("[bold red]The model returned no response candidates. The query might have been blocked.[/bold red]", title="[bold red]Error[/bold red]"))
        if hasattr(response, 'prompt_feedback'):
                console.print(f"Feedback: {response.prompt_feedback}")
        return

    candidate = response.candidates[0]
    
    # Sprawdź czy jest function_call
    function_call = None
    if candidate.content and candidate.content.parts:
        part = candidate.content.parts[0]
        if part.function_call:
            function_call = part.function_call
    
    if function_call:
        function_name = function_call.name
        args = to_dict(function_call.args)

        console.print(Panel(f"Executing function: [bold]{function_name}[/bold] with arguments: {args}", title="[bold blue]Action[/bold blue]"))
        
        # Wykonanie funkcji w oddzielnym wątku, aby nie blokować UI.
        # Nie używamy tutaj spinnera (console.status), aby logi z funkcji mogły swobodnie trafiać na stdout.
        console.print(f"[dim]Starting execution of function {function_name}...[/dim]")
        result = await asyncio.to_thread(dispatcher.dispatch, function_name, args)
        
        # Wyświetlenie surowego wyniku funkcji (opcjonalne, dla debugowania/info)
        # console.print(Panel(str(result), title=f"[bold green]Operation Result[/bold green]"))

        # Przesłanie wyniku z powrotem do modelu
        with console.status("[bold green]Generating response...[/bold green]", spinner="dots"):
            # Pakujemy wynik w słownik zgodnie z oczekiwaniem send_tool_response
            tool_output = {"result": result} 
            final_response = await connector.send_tool_response(function_name, tool_output)
        
        # Rekurencyjne przetwarzanie (model może chcieć wywołać kolejną funkcję lub odpowiedzieć tekstem)
        await process_model_response(final_response, connector, dispatcher, console)

    else:
        # Jeśli nie ma wywołania funkcji, wyświetl zwykłą odpowiedź tekstową
        try:
            text_response = response.text
            console.print(Panel(text_response, title="[bold cyan]Gemini[/bold cyan]", border_style="cyan"))
        except ValueError:
            console.print(Panel("[bold yellow]The model returned a response containing neither text nor a function call.[/bold yellow]", title="[bold yellow]Info[/bold yellow]"))


async def main():
    """
    Główna funkcja uruchamiająca interfejs czatu MLOps.
    """
    console = Console()
    console.print(Panel("Welcome to [bold cyan]ChatMLOps[/bold cyan]!", title="[bold green]Gemini AI Assistant[/bold green]", border_style="green"))

    # Konfiguracja historii readline
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, HISTORY_FILE)

    connector = GeminiConnector()
    if not connector.configure():
        return

    # Inicjalizacja modelu i sesji
    if not connector.get_model():
        return

    dispatcher = FunctionDispatcher()

    while True:
        try:
            # Używamy console.input zamiast Prompt.ask dla obsługi readline
            user_input = console.input("[bold]You[/bold]: ")

            if user_input.lower() == "exit":
                console.print(Panel("Goodbye!", border_style="red"))
                break
            
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = await connector.send_message(user_input)
            
            await process_model_response(response, connector, dispatcher, console)

        except KeyboardInterrupt:
            console.print(Panel("\nGoodbye!", border_style="red"))
            break
        except Exception as e:
            console.print(Panel(f"[bold red]An unexpected error occurred: {e}[/bold red]", title="[bold red]Critical Error[/bold red]"))

if __name__ == "__main__":
    asyncio.run(main())
