import signal
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import webbrowser
from threading import Timer
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from datetime import datetime
import os
from modules.calculations import (
    call_price,
    put_price,
    call_delta,
    put_delta,
    gamma,
    vega,
    call_theta,
    put_theta,
    call_rho,
    put_rho,
)
from modules.simulations import (
    generate_scenarios,
    calculate_call_payoffs,
    calculate_put_payoffs,
    simulate_scenario,
)
from modules.plots import (
    plot_price_paths,
    plot_payoff_distribution,
    plot_greeks,
)
import pandas as pd
from fpdf import FPDF
import plotly.io as pio
import socket

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Rapport de Simulation d'Options", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, "L", 1)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_graph(self, image_path, title):
        self.chapter_title(title)
        self.image(image_path, w=170)
        self.ln()


def generate_pdf_report(simulation_data, graphs):
    pdf = PDFReport()
    pdf.add_page()

    # Title page
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "", 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.ln(20)

    # Summary
    pdf.chapter_title("Résumé de la Simulation")
    for key, value in simulation_data.items():
        pdf.chapter_body(f"{key}: {value}")

    # Add graphs
    for title, image_path in graphs.items():
        if os.path.exists(image_path):
            pdf.add_graph(image_path, title)

    pdf_output = "rapport_simulation.pdf"
    pdf.output(pdf_output)
    return pdf_output


def serve_layout():
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H1(
                                    "Générateur de Scénarios pour Portefeuille d'Options Européennes",
                                    className="text-center",
                                ),
                                className="mb-5 mt-5",
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H3("Paramètres", className="text-center"),
                                    dbc.Form(
                                        [
                                            dbc.Label(
                                                "Prix initial de l'actif sous-jacent"
                                            ),
                                            dbc.Input(
                                                id="S0", type="number", value=100
                                            ),
                                            dbc.Label("Prix d'exercice"),
                                            dbc.Input(id="K", type="number", value=100),
                                            dbc.Label(
                                                "Temps jusqu'à l'échéance (en mois)"
                                            ),
                                            dbc.Input(
                                                id="T",
                                                type="number",
                                                value=12,
                                                min=1,
                                                max=48,
                                            ),
                                            dbc.Label("Taux d'intérêt sans risque"),
                                            dbc.Input(
                                                id="r", type="number", value=0.05
                                            ),
                                            dbc.Label("Volatilité"),
                                            dbc.Input(
                                                id="sigma", type="number", value=0.2
                                            ),
                                            dbc.Label(
                                                "Nombre de simulations Monte Carlo"
                                            ),
                                            dbc.Input(
                                                id="num_simulations",
                                                type="number",
                                                value=10000,
                                                step=1000,
                                            ),
                                            dbc.Label("Nombre de pas de temps"),
                                            dbc.Input(
                                                id="num_steps", type="number", value=252
                                            ),
                                        ]
                                    ),
                                    html.H3("Widgets", className="text-center mt-4"),
                                    dcc.Checklist(
                                        id="widget-checklist",
                                        options=[
                                            {
                                                "label": "Trajectoires de Prix",
                                                "value": "price_paths",
                                            },
                                            {
                                                "label": "Distribution des Payoffs (Call)",
                                                "value": "call_payoff",
                                            },
                                            {
                                                "label": "Distribution des Payoffs (Put)",
                                                "value": "put_payoff",
                                            },
                                            {
                                                "label": "Greeks (Call)",
                                                "value": "call_greeks",
                                            },
                                            {
                                                "label": "Greeks (Put)",
                                                "value": "put_greeks",
                                            },
                                        ],
                                        value=[
                                            "price_paths",
                                            "call_payoff",
                                            "put_payoff",
                                            "call_greeks",
                                            "put_greeks",
                                        ],
                                        labelStyle={"display": "block"},
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dcc.Graph(id="price-paths-graph"),
                                    dcc.Graph(id="call-payoff-distribution-graph"),
                                    dcc.Graph(id="put-payoff-distribution-graph"),
                                    dcc.Graph(id="call-greeks-graph"),
                                    dcc.Graph(id="put-greeks-graph"),
                                    html.Div(id="simulation-results"),
                                    dbc.Button(
                                        "Exporter Rapport",
                                        id="export-button",
                                        color="secondary",
                                        className="mt-4",
                                    ),
                                    html.Div(id="export-result"),
                                ],
                                width=8,
                            ),
                        ]
                    ),
                ]
            )
        ]
    )


app.layout = serve_layout


@app.callback(
    [
        Output("price-paths-graph", "figure"),
        Output("call-payoff-distribution-graph", "figure"),
        Output("put-payoff-distribution-graph", "figure"),
        Output("call-greeks-graph", "figure"),
        Output("put-greeks-graph", "figure"),
        Output("simulation-results", "children"),
    ],
    [
        Input("S0", "value"),
        Input("K", "value"),
        Input("T", "value"),
        Input("r", "value"),
        Input("sigma", "value"),
        Input("num_simulations", "value"),
        Input("num_steps", "value"),
        Input("widget-checklist", "value"),
    ],
)
def update_graphs(S0, K, T, r, sigma, num_simulations, num_steps, selected_widgets):
    T_years = T / 12  # Convert months to years
    price_paths = generate_scenarios(S0, r, sigma, T_years, num_steps, num_simulations)
    call_payoffs = calculate_call_payoffs(price_paths, K)
    put_payoffs = calculate_put_payoffs(price_paths, K)
    call_price_mc = np.exp(-r * T_years) * np.mean(call_payoffs)
    put_price_mc = np.exp(-r * T_years) * np.mean(put_payoffs)

    call_deltas = [call_delta(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    call_gammas = [gamma(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    call_vegas = [vega(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    call_thetas = [call_theta(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    call_rhos = [call_rho(s, K, T_years, r, sigma) for s in price_paths[:, 0]]

    put_deltas = [put_delta(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    put_gammas = [gamma(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    put_vegas = [vega(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    put_thetas = [put_theta(s, K, T_years, r, sigma) for s in price_paths[:, 0]]
    put_rhos = [put_rho(s, K, T_years, r, sigma) for s in price_paths[:, 0]]

    option_portfolio = [
        {"type": "call", "S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
        {"type": "put", "S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
    ]

    market_conditions = [
        {"price_change": 0.1, "volatility_change": 0.05},
        {"price_change": -0.1, "volatility_change": -0.05},
    ]

    simulated_values = simulate_scenario(option_portfolio, market_conditions)

    figures = {
        "price_paths": plot_price_paths(price_paths),
        "call_payoff": plot_payoff_distribution(call_payoffs, "call"),
        "put_payoff": plot_payoff_distribution(put_payoffs, "put"),
        "call_greeks": plot_greeks(
            price_paths,
            {
                "Delta": call_deltas,
                "Gamma": call_gammas,
                "Vega": call_vegas,
                "Theta": call_thetas,
                "Rho": call_rhos,
            },
            "call",
        ),
        "put_greeks": plot_greeks(
            price_paths,
            {
                "Delta": put_deltas,
                "Gamma": put_gammas,
                "Vega": put_vegas,
                "Theta": put_thetas,
                "Rho": put_rhos,
            },
            "put",
        ),
    }

    return (
        figures.get("price_paths") if "price_paths" in selected_widgets else {},
        figures.get("call_payoff") if "call_payoff" in selected_widgets else {},
        figures.get("put_payoff") if "put_payoff" in selected_widgets else {},
        figures.get("call_greeks") if "call_greeks" in selected_widgets else {},
        figures.get("put_greeks") if "put_greeks" in selected_widgets else {},
        f"Valeurs simulées du portefeuille pour différents scénarios de marché: {simulated_values}",
    )


@app.callback(
    Output("export-result", "children"),
    Input("export-button", "n_clicks"),
    State("S0", "value"),
    State("K", "value"),
    State("T", "value"),
    State("r", "value"),
    State("sigma", "value"),
    State("num_simulations", "value"),
    State("num_steps", "value"),
)
def export_report(n_clicks, S0, K, T, r, sigma, num_simulations, num_steps):
    if n_clicks:
        T_years = T / 12  # Convert months to years
        price_paths = generate_scenarios(
            S0, r, sigma, T_years, num_steps, num_simulations
        )
        call_payoffs = calculate_call_payoffs(price_paths, K)
        put_payoffs = calculate_put_payoffs(price_paths, K)
        call_price_mc = np.exp(-r * T_years) * np.mean(call_payoffs)
        put_price_mc = np.exp(-r * T_years) * np.mean(put_payoffs)

        simulation_data = {
            "Prix Initial": S0,
            "Prix d'Exercice": K,
            "Temps jusqu'à l'Échéance (mois)": T,
            "Taux d'Intérêt": r,
            "Volatilité": sigma,
            "Nombre de Simulations": num_simulations,
            "Nombre de Pas de Temps": num_steps,
            "Prix Call Monte Carlo": round(call_price_mc, 2),
            "Prix Put Monte Carlo": round(put_price_mc, 2),
        }

        figures = {
            "Trajectoires de Prix": plot_price_paths(price_paths),
            "Distribution des Payoffs (Call)": plot_payoff_distribution(
                call_payoffs, "call"
            ),
            "Distribution des Payoffs (Put)": plot_payoff_distribution(
                put_payoffs, "put"
            ),
        }

        graph_files = {}
        for title, fig in figures.items():
            filename = f"{title.replace(' ', '_').lower()}.png"
            pio.write_image(fig, filename)
            graph_files[title] = filename

        pdf_output = generate_pdf_report(simulation_data, graph_files)

        # Clean up image files after generating the PDF
        for file in graph_files.values():
            if os.path.exists(file):
                os.remove(file)

        return f"Rapport exporté: {pdf_output}"
    return ""


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    port = find_free_port()
    Timer(1, open_browser, args=[port]).start()
    app.run_server(debug=True, port=port)


def handle_exit(*args):
    print("Stopping server...")
    app.server.shutdown()


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
