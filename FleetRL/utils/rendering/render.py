import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ParkingLotRenderer:

    @staticmethod
    def render(there, kw, soc):
        fig, ax = plt.subplots()

        # Number of parking spots is determined by the length of the 'there' array
        num_spots = len(there)

        # Create a line representing the parking lot
        for i in range(num_spots):
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, edgecolor='black', facecolor='none'))

            # If a car is present in the spot, draw a car rectangle
            if there[i] == 1:
                car_rect = patches.Rectangle((i + 0.1, 0.1), 0.8, 0.8, edgecolor='blue', facecolor='lightblue')
                ax.add_patch(car_rect)
                ax.text(i + 0.5, 0.5, f'{kw[i]} kW', horizontalalignment='center', verticalalignment='center')
                ax.text(i + 0.5, 0.3, f'SOC: {round(soc[i]*100, 1)}%', horizontalalignment='center', verticalalignment='center')

        # Set plot limits and show the plot
        ax.set_xlim(0, num_spots)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.show()
