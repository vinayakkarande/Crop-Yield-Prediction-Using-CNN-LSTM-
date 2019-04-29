import ee
import datetime
import time
import numba as nb
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from tqdm import tnrange
from collections import Counter

# Authenticate to the Earth Engine servers.
ee.Initialize()

def timestamp_to_datetime(timestamp, time_format = '%Y-%m-%d'):
	return datetime.datetime.fromtimestamp(timestamp/1000).strftime(time_format)

def dates_available(geCollection):
    """
    Returns a list of the dates available for this collection.
    geCollection: ee.ImageCollection object
    Returns a list of date strings in YYYY-MM-DD format.
	Author: https://github.com/jldowns
    """
		
    timestamps =  geCollection.aggregate_array('system:time_start').getInfo()
    dateformat_array = [timestamp_to_datetime(t) for t in timestamps]
    return  dateformat_array
	
def available_bands(image_collection):
    """
    Returns a dictionary in format
    { "band1": { "number_available" : number of images that contain band1,
                 "percent_available" : percent of all images in collection that contain band1 }
                },
      "band2": {...
    }
	Author: https://github.com/jldowns
    """
	
    band_ids = [band_info['id'] for band_info in image_collection.first().getInfo()['bands']]
    collection_size = image_collection.size().getInfo()
    availability_dict = {}
    for b in band_ids:
        imgs_available = image_collection.select(b).size().getInfo()
        percent_available = imgs_available/collection_size*100
        availability_dict[b] = {
            "number_available": imgs_available,
            "percent_available": percent_available
        }
        # print( "'"+b+"' available in "+ str(imgs_available) + " images. ("+str(percent_available)+"%)")

    return availability_dict


def export_image(img, loc, name, scale, crs):
	"""
	Export an image from Earth Engine to my Google Drive
	"""
	task = ee.batch.Export.image.toDrive(img, name, folder=loc, scale=scale, crs=crs)
	task.start()
	while task.status()['state'] == 'RUNNING':
		print('Running...')		
		# Perhaps task.cancel() at some point.
		time.sleep(10)
	print('Done.', task.status())

def appendBand(current, previous):
	"""
	Transforms an Image Collection with 1 band per Image into a single Image with items as bands
	Author: Jamie Vleeshouwer
	"""
	# Rename the band
	previous = ee.Image(previous)
	current = current
    # Append it to the result (Note: only return current item on first element/iteration)
	accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
	return accum

def merge_img_array(arr_1, nband_1, arr_2, nband_2):
    """
    Merge images from difference arrays of 2D images. (ex. one has 7, the other has 2, both of them has 500 temporal images, then 
    we merge to a list of 500 temporal images with 9 bands.) The variable arr_1 is the main one.
    """
    m = arr_1.shape[0]
    n = arr_1.shape[1]
    l1 = arr_1.shape[2]
    l2 = arr_2.shape[2]
    merged_arr = np.empty((m, n, l1+l2))
    
    
    for i in range(int(l1/nband_1)):
        position_1 = i*nband_1 + i*nband_2
        position_2 = position_1 + nband_1
        merged_arr[:, :, position_1 : position_1+nband_1] = arr_1[:, :, i*nband_1 : i*nband_1+nband_1]
        merged_arr[:, :, position_2 : position_2+nband_2] = arr_2[:, :, i*nband_2 : i*nband_2+nband_2]
        
    # Check before done
    clause_1 = np.all(np.equal(merged_arr[:, :, 0:nband_1], arr_1[:, :, 0:nband_1]))
    clause_2 = np.all(np.equal(merged_arr[:, :, -nband_2:], arr_2[:, :, -nband_2:]))
    if (clause_1 == True) and (clause_2 == True):
        #print('done')
        pass
    else:
        raise ValueError('A very specific bad thing happened. Maybe the number of given band is wrong?')
    return merged_arr
	
	
#@nb.jit
def get_img_per_yr(dates_of_images):
    
    """ Return a dictionary of year with counts of images in that year.
    The input must be a list of date string in a formart of 'yyyy-mm-dd'"""
    
    years = [date[0:4] for date in dates_of_images]
    unique_year = list(np.unique(years))
    years = {year:years.count(year) for year in unique_year}
    
    return years


def get_year_dim(years, nband):
    
    """ Get a list of [start, stop] for dimensions of bands in each year for a given amount of years.
    Input years is a dictionary from get_img_per_yr"""

    year_list = sorted(years.keys())
    dimensions = [[years[year]*nband*year_list.index(year), years[year]*nband*year_list.index(year) + years[year]*nband] for year in years]
    
    return dimensions 


def filter_year(year_range, img, nband, dates_of_images):
    
    """ Return the image with specify year range and the year is [inclusive, exclusive]"""
    
    # need to do this cause we use dictionary keys to reference
    year_range[1] = str(int(year_range[1]) - 1)
    
    years = get_img_per_yr(dates_of_images)
    year_dim = get_year_dim(years, nband)
    year_list = sorted(years.keys())
    
    if np.all(year_range[0] not in year_list or year_range[1] not in year_list):
        raise(ValueError('The given year is out of bound for the given set of image.'))
        
    start = year_dim[year_list.index(year_range[0])][0]
    stop = year_dim[year_list.index(year_range[1])][1] #<-- compensate for the -1 earlier
    new_img = img[:, :, start: stop]
    new_dates_of_images = [date for date in dates_of_images if date[0:4] in \
                          [str(year) for year in range(int(year_range[0]), int(year_range[1]) + 1)]]

    return new_img, new_dates_of_images

def split_img_by_year(img, nband, dates_of_images):
    
    """Return a list of image separated by year from a single stacked-by-year image"""
    
    years = get_img_per_yr(dates_of_images)
    year_dim = get_year_dim(years, nband)
    img_list = [img[:, :, dim[0]:dim[1]] for dim in year_dim]
    return img_list
    
def count_img_nan(img):
    
    """ Check and count the number of nan in the image, will return nothing if 
    there is no nan, but will retur number if there is a nan"""
    
    return np.count_nonzero(np.isnan(img))

def norm_to_zero_one(arr):
    return (arr - np.nanmin(arr)) * 1.0 / (np.nanmax(arr) - np.nanmin(arr))

def normalize_impute_image(img):
    
    #plt.imshow(img[:, :, band])
    #plt.show()
    #img_df = norm_to_zero_one(pd.DataFrame(img))
    img = norm_to_zero_one(img)
    #print(count_img_nan(test_img))
    img = SoftImpute(max_iters=300, verbose=0).complete(img)
    if count_img_nan(img) > 0:
        print('There is a problem with band {}'.format(band))
    return img

@nb.jit
def zero_fill(img):
    for band in range(img.shape[2]):
        img[:, :, band] = np.nan_to_num(img[:, :, band])
    return img

@nb.jit
def mask_img(img, mask):
    for band in range(len(img.shape[2])):
        img[:, :, band] = img[:, :, band] * mask[:, :, :]
		
		
def plt_img_dist(file_dir, nband, nsample, yield_df):
    import random
    """
    Return data for plotting the distribution of the number in each band of a set of images in a folder.
    Image is in a form of npy format. 
    """
    
    fips = yield_df[['YEAR', 'STATE', 'DISTRICT']][yield_df['YEAR'] >= 2010].values
    rand_idx = random.sample(range(0, len(fips)), nsample)
    data = {band:[] for band in range(0, nband)}
    for i, _ in enumerate(tqdm(fips[rand_idx])):
        #file = random.choice(os.listdir("D://projectII_temp_data//MODIS_LAND"))
        try:
            test = np.load(file_dir + str(fips[i][0]) + '_' + str(fips[i][1]) + '_' + str(fips[i][2]) + '.npy')
            for band in range(nband):
                data[band] = np.append(data[band], test[:, :, band].ravel())
                data[band] = data[band][np.nonzero(data[band])]
        except: # In case it hits [51,131] or [46, 102]
            pass
    return data
	
def get_stats(file_dir, nband, nsample, yield_df):
    
    import random

    avg_MODIS_LAND = {band:[] for band in range(0, nband)}  # average per images (38 images per year for MODIS)
    std_MODIS_LAND = {band:[] for band in range(0, nband)}
    count_MODIS_LAND = {band:[] for band in range(0, nband)}
    
    avg_year = {band:[] for band in range(0, nband)}   # yearly average (for plotting)
    var_year = {band:[] for band in range(0, nband)}
    max_year = {band:[] for band in range(0, nband)}
    count_year = {band:[] for band in range(0, nband)}
    
    avg_total = {band:[] for band in range(0, nband)}     # Total average (for statistical analysis)
    std_total = {band:[] for band in range(0, nband)} 
    
    yield_ = []
    nband = nband
    nsample = nsample

    fips = yield_df[['YEAR', 'STATE', 'DISTRICT']][yield_df['YEAR'] >= 2010].values
    rand_idx = random.sample(range(0, len(fips)), nsample)
    for i, _ in enumerate(fips[rand_idx]):
        try:
            test = np.load(file_dir + str(fips[i][0]) + '_' + str(fips[i][1]) + '_' + str(fips[i][2]) + '.npy')
            yield_ += [yield_df['YIELD'][(yield_df['STATE'] == int(fips[i][1])) & (yield_df['DISTRICT'] == int(fips[i][2])) \
                                          & (yield_df['YEAR'] == int(fips[i][0]))].values]
            for band in range(nband):
                bands = list(np.arange(band, test.shape[2], nband)) 
                non_zero_index = test[:, :, bands].nonzero()
                count_year[band] += [len(non_zero_index[0])]
                avg_year[band] += [test[:, :, bands][non_zero_index].mean()]
                var_year[band] += [test[:, :, bands][non_zero_index].var()]
                max_year[band] += [test[:, :, bands].max()]

        except:
            pass

    for band in range(nband):
        # Delete NaN (i.e. samples that do not have any corn in that year)
        nan_idx = np.argwhere(np.isnan(avg_year[band]))
        avg_year[band] = np.delete(avg_year[band], nan_idx)
        var_year[band] = np.delete(var_year[band], nan_idx)
        count_year[band] = np.delete(count_year[band], nan_idx)
        

        avg_total[band] = np.sum(avg_year[band]*np.array(count_year[band]))/ np.sum(count_year[band])
        std_total[band] = np.sqrt(np.sum(var_year[band]*np.array(count_year[band]))/ np.sum(count_year[band]))
        
    yield_ = np.delete(yield_, nan_idx)       
    return avg_year, count_year, max_year, yield_, avg_total, std_total
	
def get_bin(img, lims, nband, resolution):
    """Making an image taken for a number of time in a year and convert 
    it to bins. The result has a dimension of 
    (nbands, resolution of bin, imgs per year)"""
    
    y = np.zeros((nband, resolution, int(img.shape[2]/nband)))
    for time in range(int(img.shape[2]/nband)):
        for lim, band in zip(lims, range(nband)):
            bins = np.array(np.linspace(lim[0], lim[1], resolution))
            x = img[:, :, band + time*nband][np.nonzero(img[:, :, band + time*nband])]
            x = np.digitize(x.ravel(), bins)
            count = Counter(x)
            y[band, :, time] = np.array([count[i+1] for i in range(resolution)])/len(x)
    return y