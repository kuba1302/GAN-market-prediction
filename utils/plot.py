import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from numpy import single


def plot_all_prices(
    data_dict, x_col, y_col, one_figsize: list, minticks=20, maxticks=30
):

    number_of_plots = len(data_dict)
    _, axes = plt.subplots(
        number_of_plots, 1, figsize=(one_figsize[0], one_figsize[1] * number_of_plots)
    )

    for ax, (company, data) in zip(axes, data_dict.items()):
        ax.plot(data[x_col], data[y_col])
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        )
        ax.set_title(
            f"{company.capitalize()} close stock price in past 3 years.", size=25
        )
        ax.set_xlabel("Time", size=15)
        ax.set_ylabel("Price in $", size=15)

    plt.show()


def plot_bb(close, up, down):
    plt.title("Bollinger Bands", size=25)
    plt.xlabel("Days")
    plt.ylabel("Closing Prices")
    plt.plot(close, label="Closing Prices")
    plt.plot(up, label="Bollinger Up", c="y")
    plt.plot(down, label="Bollinger Down", c="b")
    plt.legend()
    plt.show()


def plot_multiple_lines(title, close, list_of_lines):
    plt.title(title, size=25)
    plt.xlabel("Days")
    plt.ylabel("Closing Prices")
    plt.plot(close, label="Closing Prices")

    for line_tuple in list_of_lines:
        plt.plot(line_tuple[0], label=line_tuple[1], c=line_tuple[2])

    plt.legend()
    plt.show()


def plot_macd(macd, signal):
    plt.title("MACD", size=25)
    plt.xlabel("Days")
    plt.ylabel("Indicators value")
    plt.plot(macd, label="MACD", c="g")
    plt.plot(signal, label="Signal line", c="r")
    plt.axhline(y=0, color="w", linestyle="-")
    plt.legend()
    plt.show()
