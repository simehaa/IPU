import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(i):
	# Called by matplotlib.animation.FuncAnimation for each new frame
	ax1.imshow(e[i])
	ax2.imshow(r[i])
	print(f'\rCreating gif: {(i+1)/frames*100:1.1f}%', end='')

if __name__ == '__main__':
	frames = 200
	e = [pd.read_csv(f"./data/e{i}.csv", header=None) for i in range(frames)]
	r = [pd.read_csv(f"./data/r{i}.csv", header=None) for i in range(frames)]
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.set_title(f'e')
	ax2.set_title(f'r')
	im1 = ax1.imshow(e[0])
	im2 = ax2.imshow(r[0])
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.29, 0.05, 0.41]) # colorbar right to the subplots
	plt.colorbar(im2, cax=cbar_ax)
	ani = animation.FuncAnimation(fig, animate, frames)
	ani.save('AlievPanfilov.gif', fps=16)
	print(", Done!")