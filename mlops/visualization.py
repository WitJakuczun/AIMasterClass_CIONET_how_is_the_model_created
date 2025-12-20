import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List

def plot_model_comparison(
    results: Union[pl.DataFrame, pd.DataFrame, List[Dict]], 
    metric_name: str, 
    output_path: Path, 
    title: str = "Porównanie modeli"
) -> str:
    """
    Generuje wykres słupkowy porównujący modele i zapisuje go do pliku.
    
    Args:
        results: DataFrame (polars/pandas) lub lista słowników z wynikami. 
                 Musi zawierać kolumny/klucze 'model_name' i metric_name.
        metric_name: Nazwa metryki do wizualizacji (oś Y).
        output_path: Ścieżka zapisu pliku (.png).
        title: Tytuł wykresu.
        
    Returns:
        Ścieżka do zapisanego pliku jako string.
    """
    # Konwersja do pandas na potrzeby seaborn/matplotlib
    if isinstance(results, list):
        df = pd.DataFrame(results)
    elif isinstance(results, pl.DataFrame):
        df = results.to_pandas()
    else:
        df = results.copy()

    # Sprawdzenie czy wymagane kolumny istnieją
    if "model_name" not in df.columns:
        raise ValueError("Dane muszą zawierać kolumnę 'model_name'")
    if metric_name not in df.columns:
        raise ValueError(f"Dane muszą zawierać kolumnę metryki '{metric_name}'")

    # Ustawienie stylu
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Rysowanie wykresu
    ax = sns.barplot(
        data=df,
        x="model_name",
        y=metric_name,
        hue="model_name",
        palette="viridis",
        legend=False
    )

    # Dodanie wartości nad słupkami
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)

    plt.title(title)
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Utworzenie katalogu jeśli nie istnieje
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Zapis
    plt.savefig(output_path, dpi=300)
    plt.close()

    return str(output_path)
