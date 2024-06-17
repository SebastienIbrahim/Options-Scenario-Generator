import plotly.graph_objs as go
import plotly.io as pio


def plot_price_paths(price_paths, num_paths_to_display=10):
    fig = go.Figure()
    for i in range(num_paths_to_display):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(price_paths))),
                y=price_paths[:, i],
                mode="lines",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Simulations Monte Carlo des trajectoires de prix",
        xaxis_title="Jours de bourse",
        yaxis_title="Prix de l'actif sous-jacent ($)",
    )
    return fig


def plot_payoff_distribution(payoffs, option_type):
    fig = go.Figure(
        data=[
            go.Histogram(
                x=payoffs, nbinsx=50, name=f"{option_type.capitalize()} Payoffs"
            )
        ]
    )
    fig.update_layout(
        title=f"Distribution des payoffs de l'option {option_type}",
        xaxis_title="Payoff à l'échéance ($)",
        yaxis_title="Fréquence",
        legend_title_text="Payoffs",
    )
    return fig


def plot_greeks(price_paths, greeks, option_type):
    fig = go.Figure()
    for greek, values in greeks.items():
        fig.add_trace(
            go.Scatter(x=price_paths[:, 0], y=values, mode="lines", name=greek)
        )
    fig.update_layout(
        title=f"Greeks pour les options {option_type}",
        xaxis_title="Prix du sous-jacent ($)",
        yaxis_title="Greeks",
        legend_title_text="Greeks",
    )
    return fig
