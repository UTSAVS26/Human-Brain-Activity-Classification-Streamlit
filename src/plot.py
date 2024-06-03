import matplotlib.pyplot as plt

def plot(image):
    if image.empty:
        return None  # Return None if DataFrame is empty

    fig, ax = plt.subplots(len(image.columns), figsize=(10, 35))

    # Generate a line plot for each column in the DataFrame
    for i, column in enumerate(image.columns):
        ax[i].plot(image.index, image[column], label=column)
        ax[i].grid(True)
        ax[i].set_title(str(column))
    
    plt.legend()
    plt.title('Simulated Data Line Chart')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(True)

    # Return the figure object
    return fig, ax