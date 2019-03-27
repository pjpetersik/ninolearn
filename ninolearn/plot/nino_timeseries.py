import matplotlib.pyplot as plt


def nino_background(nino_data, nino_treshold=0.8):
    """
    Generates a plotbackground based on values of the Nino3.4 Index
    """
    nino_color = nino_data.copy()
    nino_color[abs(nino_color) < nino_treshold] = 0

    ninomax = nino_data.abs().max()
    start = True
    end = False

    for i in range(len(nino_color)):
        if nino_color[i] != 0 and start:
            start_date = nino_color.index[i]
            start = False
            end = True

        if nino_color[i] == 0 and end:
            end_date = nino_color.index[i-1]
            start = True
            end = False

            windowmax = abs(nino_data.loc[start_date:end_date]).max()
            sign = nino_color.loc[start_date]

            alpha = (windowmax - nino_treshold)/(ninomax-nino_treshold) * 0.8

            if sign > 0:
                plt.axvspan(start_date, end_date, facecolor='red', alpha=alpha)
            else:
                plt.axvspan(start_date,
                            end_date,
                            facecolor='blue',
                            alpha=alpha)
