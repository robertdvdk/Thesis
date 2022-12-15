import matplotlib.pyplot as plt
import os

dirs = ['1ediff', '2EDIFF_N', '3ediff_lang', '3EDIFF_CONSTANTN', 'REAL_ALL', 'REAL_ONLY_ARATH']
xs, ys = {}, {}
for dir in dirs:
	subdir = os.listdir(f'./{dir}')[0]
	if len(os.listdir(f'./{dir}')) > 1:
		subdir = os.listdir(f'./{dir}')[5]
	with open(f"./{dir}/{subdir}/valid_disc_diff.csv", "r") as fopen:
		lines = fopen.readlines()
		xs[dir], ys[dir] = [], []
		for line in lines:
			if len(xs[dir]) < 250:
				point = line.strip()
				x, y = point.split(',')
				xs[dir].append(float(x)), ys[dir].append(float(y))


fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_figheight(8)
fig.set_figwidth(8)
fig.suptitle('Performance of GAN on various datasets', fontsize=14)
fig.text(0.5, 0.02, 'Iteration', ha='center', fontsize=14)
fig.text(0.02, 0.5, 'Mean generated score - real score', va='center', rotation='vertical', fontsize=14)
axs[0, 0].plot(xs['1ediff'], ys['1ediff'], label='Dataset 1')
axs[0, 0].plot(xs['2EDIFF_N'], ys['2EDIFF_N'], label='Dataset 2')
axs[0, 0].legend()
axs[0, 0].set_title('Easy artificial datasets', fontsize=14)
axs[0, 1].plot(xs['3EDIFF_CONSTANTN'], ys['3EDIFF_CONSTANTN'], label='Dataset 3')
axs[0, 1].plot(xs['3ediff_lang'], ys['3ediff_lang'], label='Dataset 4')
axs[0, 1].set_title('Hard artificial datasets', fontsize=14)
axs[0, 1].legend()
axs[1, 0].plot(xs['REAL_ALL'], ys['REAL_ALL'])
axs[1, 0].set_title('Entire biological dataset', fontsize=14)
axs[1, 1].plot(xs['REAL_ONLY_ARATH'], ys['REAL_ONLY_ARATH'])
axs[1, 1].set_title('Arabidopsis dataset', fontsize=14)
plt.savefig('gan_perf_large')
