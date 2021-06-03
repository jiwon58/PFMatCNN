import os, sys, numpy as np, argparse, shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import numba

def mean_nonzero(array):
    return np.sum(array)/np.count_nonzero(array)

def std_nonzero(array):
    return array[array != 0].std()

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:<10d}{}'.format(int(num), ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def addnoise(args):
    particle_std =  8.81 - 5.39/2
    #particle_std = 6.9
    #background_std = 6.01 #- 4.45/2
    background_std = 6.0
    time_step = 500000
    time_step2 = 500000
    master = './data/' + args.dataset
    _time_step = human_format(time_step).replace(" ","")
    if args.random:
        if args.shape:
            if args.background and args.particle:
                newmaster = master + '_rsbp' + _time_step
            elif args.particle:
                newmaster = master + '_rsp' + _time_step
            elif args.background:
                newmaster = master + '_rsb' + _time_step
            elif args.particle_not_background:
                # random shape on particle and just random noise on background
                newmaster = master + '_rsprb' + _time_step
        else:
            if args.particle and args.background:
                newmaster = master + '_rpb'
            elif args.particle:
                newmaster = master + '_rp'
            elif args.background:
                newmaster = master + '_rb'
    else:
        if args.shape:
            if args.background and args.particle:
                newmaster = master + '_nsbp' + _time_step
            elif args.particle:
                newmaster = master + '_nsp' + _time_step
            elif args.background:
                newmaster = master + '_nsb' + _time_step
        elif args.custom:
            newmaster = master +'_custom' + _time_step
        elif args.saltnpepper:
            newmaster = master +'_snp'
        else:
            print('Please choose one of mode')
            sys.exit(1)
    print('\t'+newmaster)
    noise_path = './noise/'+str(time_step)+'/npy'
    bg_noises = [f for f in os.listdir(noise_path) if not f[0] == '.']
    noise_path2 = './noise/'+str(time_step2)+'/npy'
    bg_noises2 = [f for f in os.listdir(noise_path2) if not f[0] == '.']


    if not os.path.isdir(master):
        print('There is no dataset!')
        sys.exit(1)
    if not os.path.exists(newmaster):
        os.mkdir(newmaster)
    content_folders = ['/images/', '/labels/', '/labels_npy/']

    folder_list = ['/train', '/validate']
    for subfolders in folder_list:
        if not os.path.exists(newmaster+subfolders):
            os.mkdir(newmaster+subfolders)
            os.mkdir(newmaster+subfolders+'/images')
            os.mkdir(newmaster+subfolders+'/labels')
            os.mkdir(newmaster+subfolders+'/labels_npy')
        print('-----> {} folder'.format(subfolders[1:]))
        np.random.seed(1119)
        random_int = np.random.randint(len(bg_noises),size=15)
        FileList = sorted([f for f in os.listdir(master+subfolders+'/labels_npy') if not f[0] == '.'])
        for i in tqdm(range(len(FileList))):
            file = FileList[i]
            # Particles from labels
            particles = np.load(master+subfolders+'/labels_npy/'+file)
            N = particles.shape[0]
            # ~particle = background
            background = 1 - particles

            img = Image.open(os.path.join(master+subfolders+'/images/'+file[:-6]+'.png')).convert('L')
            img = np.asarray(img)
            if args.random:
                if args.shape: # Random normal noise with simulation images
                    if args.particle and args.background:
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        particle_noise = np.multiply(particles, np.random.normal(0, particle_std, size=(N, N)))
                        particle_noise = particle_noise + np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))

                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        background_noise = np.multiply(background, np.random.normal(0, background_std , size=(N, N)))
                        background_noise = background_noise + np.multiply(background, background_std * np.load(os.path.join(noise_path,bg_noises[random_int])))

                        average_shift_particle = mean_nonzero(particle_noise)
                        average_shift_background = mean_nonzero(background_noise)
                        img_noise = img + particle_noise + background_noise - np.tile(average_shift_particle, (N, N)) * particles - np.tile(average_shift_background, (N, N)) * background

                    elif args.particle:
                        # Pick random images from image folder
                        particle_noise = np.multiply(particles, np.random.normal(0, particle_std, size=(N, N)))
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        particle_noise = particle_noise + np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        average_shift = mean_nonzero(particle_noise)
                        img_noise = img + particle_noise - np.tile(average_shift, (N, N)) * particles

                    elif args.background:
                        # Pick random images from image folder
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        background_noise = np.multiply(background, np.random.normal(0, background_std , size=(N, N)))
                        background_noise = background_noise + np.multiply(background, background_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        average_shift = mean_nonzero(background_noise)
                        img_noise = img + background_noise - np.tile(average_shift, (N, N)) * background

                    elif args.particle_not_background:
                        # Case study option
                        # Brightness of dendrites: 90.7
                        # Contrast of dendrites: 8.8
                        # Brightness of background: 133.1
                        # Contrast of background: 6.0 (without mask 9.9)

                        particle_noise = np.multiply(particles, np.random.normal(1.2, particle_std, size=(N, N)))

                        np.random.seed(1119) # for same phase field noises
                        random_int = np.random.randint(len(bg_noises),size=1)[0]

                        particle_noise = particle_noise + np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        #particle_noise = np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        average_shift = mean_nonzero(np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))) 
                        img_noise = img + particle_noise - np.tile(average_shift, (N, N)) * particles 

                else: # Just random normal noises
                    if args.particle and args.background:
                        particle_noise = np.multiply(particles, np.random.normal(0.3, particle_std, size=(N, N)))
                        background_noise = np.multiply(background, np.random.normal(1.75, background_std, size=(N, N)))
                        img_noise = img + particle_noise + background_noise

                    elif args.particle:

                        particle_noise = np.multiply(particles, np.random.normal(0, particle_std, size=(N, N)))
                        img_noise = img + particle_noise

                    elif args.background:
                        background_noise = np.multiply(background, np.random.normal(0, background_std, size=(N, N)))
                        img_noise = img + background_noise

            else: # Not having random noise, Just simulation images as noise
                if args.shape:
                    if args.particle and args.background:
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        particle_noise = np.multiply(particles, particle_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        background_noise = np.multiply(background, background_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        average_shift_particle = mean_nonzero(particle_noise)
                        average_shift_background = mean_nonzero(background_noise)
                        img_noise = img + particle_noise + background_noise - np.tile(average_shift_particle, (N, N)) * particles -np.tile(average_shift_background, (N, N)) * background

                    elif args.particle:
                        # Pick random images from image folder
                        np.random.seed(1119)
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        particle_noise = np.multiply(particles, (particle_std)* np.load(os.path.join(noise_path,bg_noises[random_int[i]])))
                        average_shift = mean_nonzero(particle_noise)
                        img_noise = img + particle_noise - np.tile(average_shift, (N, N)) * particles

                    elif args.background:
                        # Pick random images from image folder
                        random_int = np.random.randint(len(bg_noises),size=1)[0]
                        background_noise = np.multiply(background, background_std * np.load(os.path.join(noise_path,bg_noises[random_int])))
                        average_shift = mean_nonzero(background_noise)
                        img_noise = img + background_noise - np.tile(average_shift, (N, N)) * background
                else: # There is no case of this option
                    continue

            plt.imsave('{0}{1}/images/{2}.png'.format(newmaster, subfolders,file[:-6]), img_noise,
            cmap='gray', vmin=0, vmax=255)
            shutil.copy(master+subfolders+'/labels/'+file[:-6]+'_L.png',newmaster+subfolders+'/labels/'+file[:-6]+'_L.png')
            shutil.copy(master+subfolders+'/labels_npy/'+file,newmaster+subfolders+'/labels_npy/'+file)
        print('{} folder: Done!\n'.format(subfolders[1:]))
    #print(newmaster)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add noise images')
    parser.add_argument('dataset', help='Name of Dataset in which Conversion will Take Place')
    parser.add_argument('-p', '--particle', action='store_true', help='Put noise on particle side')
    parser.add_argument('-b', '--background', action='store_true', help='Put noise on background side')
    parser.add_argument('-ppnb', '--particle_not_background', action='store_true', help='Random & Phase-field on particle / Only random on background')
    parser.add_argument('-s', '--shape', action='store_true', help='Put noise using images')
    args = parser.parse_args()
    addnoise(args)
