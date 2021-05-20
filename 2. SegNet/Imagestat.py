import os, sys, numpy as np, argparse, json
from PIL import Image, ImageStat, ImageChops
from skimage.measure import regionprops, label, perimeter
from tqdm import tqdm
from pyprnt import prnt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
"""
Stat_matrix column headers
% % This is what is stored in each column:
% 01 - Number of pixels in image / Number_pix [#]
% 02 - ?Interface thickness? [#]
% 03 - Image luminance / Image_luminance [#]
% 04 - Image RMS contrast / Image_RMS_contrast [#]
% 05 - Dendrite RMS contrast / Dendrite_RMS_contrast [#]
% 06 - Background RMS contrast / Background_RMS_contrast [#]
% 07 - Dendrite area fraction / Dendrite_areafraction [#]
% 08 - Number of dendrite bodies / Number_denritebody [#]
% 09 - Avg body size / Avgbodysize [pixels]
% 10 - Standard deviation of body size / StdDev_bodysize [pixels]
% 11 - Curvature (dendrite area / perimeter) S_v^-1 / Curvature [pixels]
% 12 - Dendrite brightness/luminance / Dendrite_brightness [#]
% 13 - Background brightness/luminance / Background_brightness [#]
"""

def mean_nonzero(array):
    return np.sum(array)/np.count_nonzero(array)
def std_nonzero(array):
    return array[array != 0].std()

def gen_default(dataset):
    Stat_matrix = {
        'root': './data/' + dataset,
        'Number_pix':[], 'Interface_thickness':[], 'Image_luminance': [],
        'Image_RMS_contrast': [], 'Dendrite_RMS_contrast': [], 'Background_RMS_contrast': [],
        'Dendrite_areafraction': [], 'Number_denritebody': [], 'Avgbodysize': [],
        'StdDev_bodysize': [], 'Curvature': [], 'Dendrite_brightness': [],
        'Background_brightness': [], 'area_all': [], 'perimeter_all': []
    }
    return Stat_matrix

def make_stat(dir, Stat_matrix, args):
    #area_all, perimeter_all = 0, 0

    dir_im, dir_gt, dir_out = dir + '/images/', dir + '/labels/', dir + '/Stats/'

    im_names = sorted([f for f in os.listdir(dir_im) if not f[0] == '.'])
    gt_names = sorted([f for f in os.listdir(dir_gt) if not f[0] == '.'])
    if not len(im_names) == len(gt_names):
        print('Number of images and ground truth is different')
        sys.exit(1)

    for image in tqdm(im_names):

        img = Image.open(os.path.join(dir_im+image)).convert('L')

        np_img = np.asarray(img)
        ## Calculate values using native numpy arrays
        Stat_matrix['Number_pix'].append(np.size(np_img))
        Stat_matrix['Image_luminance'].append(np.mean(np_img))
        Stat_matrix['Image_RMS_contrast'].append(np.std(np_img))

        gt = Image.open(dir_gt+image[:-4]+'_L.png').convert('L')
        stat_gt = ImageStat.Stat(gt)
        np_gt = np.asarray(gt)
        np_gt_l = label(np_gt)

        stat_dend = ImageStat.Stat(img, gt)

        if stat_dend.count[0] == 0:
            Stat_matrix['Dendrite_brightness'].append(0)
            Stat_matrix['Dendrite_RMS_contrast'].append(0)
        else:
            ## Calculate dendrites avg, std using true and false
            Stat_matrix['Dendrite_brightness'].append(mean_nonzero(np_img[np_gt != 0]))#np.mean(np_img[np_gt != 0]))
            Stat_matrix['Dendrite_RMS_contrast'].append(np.std(np_img[np_gt != 0]))


        stat_back = ImageStat.Stat(img, ImageChops.invert(gt))

        ## Calculate dendrites avg, std using true and false
        Stat_matrix['Background_brightness'].append(mean_nonzero(np_img[np_gt == 0]))#np.mean(np_img[np_gt == 0]))
        Stat_matrix['Background_RMS_contrast'].append(np.std(np_img[np_gt == 0]))

        # Calculate area fraction by number of true element in ground truth
        Stat_matrix['Dendrite_areafraction'].append(np.count_nonzero(np_gt > 0)/ np.size(np_gt))

        rp = regionprops(np_gt_l)
        areas = [r.area for r in rp if r.area >= 5]
        perimeters = [r.perimeter for r in rp if r.perimeter >= 5]
        if len(areas) == 0:
            Stat_matrix['Number_denritebody'].append(0)
            Stat_matrix['Avgbodysize'].append(0)
            Stat_matrix['StdDev_bodysize'].append(0)
            Stat_matrix['Curvature'].append(0)
        else:
            Stat_matrix['Number_denritebody'].append(len(areas))
            Stat_matrix['Avgbodysize'].append(sum(areas)/len(areas))
            Stat_matrix['StdDev_bodysize'].append(np.std(areas))
            Stat_matrix['area_all'].append(float(sum(areas)))
            Stat_matrix['perimeter_all'].append(float(sum(perimeters)))
            Stat_matrix['Curvature'].append(sum(areas)/sum(perimeters))
        if args.overlay:
            im = np.array(Image.open(os.path.join(dir_im,image)), dtype=np.uint8)
            plt.imshow(im, cmap='gray', vmin=0, vmax=255)
            ax = plt.gca()
            for region in rp:
                if region.area >= 5:
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
            ax.set_axis_off()
            if not os.path.exists(os.path.join(dir,'ROI')):
                os.makedirs(os.path.join(dir,'ROI'))
            plt.savefig(os.path.join(dir,'ROI',image), bbox_inches= 'tight', pad_inches=0)
            [p.remove() for p in reversed(ax.patches)]
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    with open(dir_out+'stat_dict.json', 'w') as fp:
        json.dump(Stat_matrix, fp)

    return Stat_matrix

def main(args):
    Stat_matrix = gen_default(args.dataset)
    if args.view:
        try:
            with open(Stat_matrix['root']+'/Stat/dataset_dict.json') as fp:
                Dataset_dict = json.load(fp)
        except IOError:
            print('File dose not exist!')
            sys.exit()
    else:
        if args.exclude_test:
            subsets = ['/train', '/validate']
        elif args.only_test:
            subsets = ['/test']
        else:
            subsets = ['/test', '/train', '/validate']
        for subset in subsets:
            print('>',subset[1:])
            Stat_matrix = make_stat(Stat_matrix['root'] + subset, Stat_matrix, args)
            #prnt(Stat_matrix)

        Dataset_dict = {
                        'Dataset Name' : Stat_matrix['root'][7:],
                        'Number of Images' : len(Stat_matrix['Curvature']),
                        'Number of pixels in image' : int(sum(Stat_matrix['Number_pix']) / len(Stat_matrix['Number_pix'])),
                        'Average Curvature': round(sum(Stat_matrix['area_all']) / sum(Stat_matrix['perimeter_all']), 3),
                        #'Average Curvature_2': round(sum(Stat_matrix['Curvature']) / len([c for c in Stat_matrix['Curvature'] if c != 0]), 3),
                        'Average Background Brightness': round(sum(Stat_matrix['Background_brightness'])/len(Stat_matrix['Background_brightness']), 3),
                        'Average Background RMS contrast': round(sum(Stat_matrix['Background_RMS_contrast'])/len(Stat_matrix['Background_RMS_contrast']), 3),
                        'Average Dendrite Brightness' : round(sum(Stat_matrix['Dendrite_brightness'])/len([c for c in Stat_matrix['Dendrite_brightness'] if c != 0]), 3),
                        'Average Dendrite RMS contrast' : round(sum(Stat_matrix['Dendrite_RMS_contrast'])/len([c for c in Stat_matrix['Dendrite_RMS_contrast'] if c != 0]), 3),
                        'Average Brightness' : round(sum(Stat_matrix['Image_luminance'])/len(Stat_matrix['Image_luminance']), 3),
                        'Image RMS contrast' : round(sum(Stat_matrix['Image_RMS_contrast'])/len(Stat_matrix['Image_RMS_contrast']), 3),
                        'Total Number of Dendrite Bodies' : sum(Stat_matrix['Number_denritebody']),
                        'Average Dendrite Area' : round(sum(Stat_matrix['area_all']) / sum(Stat_matrix['Number_denritebody']), 3),
                        'Standard Deviation Dendrite Area' : round(sum(Stat_matrix['StdDev_bodysize'])/len([c for c in Stat_matrix['StdDev_bodysize'] if c != 0]), 3),
                        'Average Area Fraction' : round(sum(Stat_matrix['Dendrite_areafraction'])/ len(Stat_matrix['Dendrite_areafraction']), 3),
                        'Average Body size' : round(sum(Stat_matrix['Avgbodysize'])/len([c for c in Stat_matrix['Avgbodysize'] if c != 0]), 3)
        }
        if not os.path.exists(Stat_matrix['root']+'/Stat'):
            os.makedirs(Stat_matrix['root']+'/Stat')
        with open(Stat_matrix['root']+'/Stat/stat_dict.json', 'w') as fp:
            json.dump(Stat_matrix, fp)
        with open(Stat_matrix['root']+'/Stat/dataset_dict.json', 'w') as fp:
            json.dump(Dataset_dict, fp)
    prnt(Dataset_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make statistics', usage = 'python Imagestat.py [dataset]')
    parser.add_argument('dataset', help='name of the dataset folder')
    #parser.add_argument('view', help='name of the subset folder')
    parser.add_argument('-v', '--view', action='store_true', help='Show the saved stat_dict')
    parser.add_argument('-o', '--overlay', action='store_true', help='Make overlay images that have particles of interest')
    parser.add_argument('-ex', '--exclude-test', action='store_true', help='Exclude test folder')
    parser.add_argument('-ot', '--only-test', action='store_true', help='Test on only test folder')
    args = parser.parse_args()
    main(args)
