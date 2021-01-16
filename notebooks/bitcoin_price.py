# %% [markdown]
'''
# The hidden secrets of the bitcoin price
'''
# %% [markdown]
'''
Bitcoin is a digital currency created in 2009 by Satoshi Nakamoto, he describes it as a "peer-to-peer version of electronic cash" <cite> nakamoto2019bitcoin </cite>.
One big advantage of bitcoin (and other cryptocurrencies) is that all the data is open and immutable, residing inside the blockchain.
The openness and immutability of the data has made the research behind blockchain really active, mostly on the price forecasting (<cite> jang2017empirical </cite>, <cite> mudassir2020time </cite>).
Many, rightfully, rush into the blockchain data (such as addresses, transactions etc..), but I will show in this post that the bitcoin price itself is already really informative.
Understanding how the price behaves will make a substantial difference in the choice of models and parameters for predicting price.

The behaviour of the price is best understood via two main properties of time series: stationarity and seasonality. For example, a stationary time series can be much easier to model than a non-stationary.

In what’s coming, I will share with you my thought process in looking at the price, using statistical tools and python programming.
'''
# %% [markdown]
'''
<binder></binder>
'''
# %% [markdown]
'''
## tl;dr
1. Two important properties for a time-serie: stationnarity (distribution does not depend on the time) and seasonnality (recurrent patterns in the data).
2. Auto-correlation to check if a data is non-stationnary; derivative or data filtering/substraction to remove the non-stationnary component.
3. FFT and short FFT to analyse the seasonnality.
'''
# %% [markdown]
'''
## 1. A quick primer on time series
'''
# %% [markdown]
'''
As I said, there are two important properties attached to time series: seasonality and stationarity.  
A stationnary process means that the distribution (statistical properties) of the data does not changes over time, this is why it is much easier to model.
Seasonnality represents how frequently the data change (for the bitcoin price, we can express it in cycles per day), and also when it starts.

We will first focus on the analysis of the stationarity, and after the seasonality.
'''
# %% [markdown]
'''
### 1.1. Stationnarity
'''
# %% [markdown]
'''
One way to detect if a data is stationnay is to compute the autocorrelation of the data, if it degrades quickly it is stationnary.  
There are many different types of non-stationnary data in the litterature, so I suggest you to read the [following post](https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322)
if you want to learn more on it. Check also [this figure](https://otexts.com/fpp2/stationarity.html) and try to guess which time-serie is stationary!
'''
# %% [markdown]
'''
### 1.2. Seasonnality
'''
# %% [markdown]
'''
To analyse the seasonality of the bitcoin, we can make a [fourier analysis](https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/) to extract the most proeminent frequencies.  
The magnitude of the FFT inform us how the given frequency component affect the price. In the other hand, the phase of the FFT is interresting to watch when does the dynamic of the price starts.
If the magnitude or phase has a random white noise trend, then there is no evidence of principal component.

Check this nice [blog post](https://machinelearningmastery.com/time-series-seasonality-with-python/) if you want to learn more on seasonnality.
'''
# %% [markdown]
'''
## 2. Code
'''
# %% [markdown]
'''
## 2.1. Loading the data
'''
# %% [markdown]
'''
The hourly USD price for the bitcoin can be collected using [glassnode](https://studio.glassnode.com/pricing), with their advanced subscription.
If you don’t want to pay for it, the 24-hour data comes free of charge.
Here we will use hourly data to get a more precise analysis.
'''
# %% [code]
### imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
# %% [markdown]
'''
First, we want to create a function to load the data.
'''
# %% [code]
# function to read the data
def read_data(filepath):
    price = []
    time = []
    
    with open(filepath) as f:
        lines = f.readlines()[1:]

    for data in lines:
        curr_time = float(data.split(",")[0])
        curr_price = -1
        if data.split(",")[1][:-1]:
            curr_price = float(data.split(",")[1][:-1])
        time += [curr_time]
        price += [curr_price]
        
    return np.array(price, dtype=np.float32), np.array(time, dtype=int)
# %% [markdown]
'''
Now we will load the data by skipping the first year.
'''
# %% [code]
# define paths
filepath = "data/market/price_usd_close_BTC_1h"
figure_dir = ""

# loading the hourly data, to avoid unimformative data, we skip the first year (8760 h)
price, time = read_data(filepath)
time_shifted = time - time[0]
price = price[8760:]
time_shifted = time_shifted[8760:]
# %% [markdown]
'''
Let's look at the bitcoin price over the time,
'''
# %% [code]
### plot
plt.figure()
plt.plot(time_shifted, price)
plt.title("Bitcoin price over time (USD)")
plt.ylabel("price (USD)")
plt.xlabel("time (h)")
if figure_dir:
    plt.savefig(os.path.join(figure_dir, "price.png"))
plt.show()
plt.close()
# %% [markdown]
'''
The non-stationnary behaviour of the data is obvious when looking at the bitcoin price.
We can also see clearly the big rises of Dec 2017/2020 there.
'''
# %% [markdown]
'''
## 2.2. Stationnarity
'''
# %% [markdown]
'''
One way to remove the non-stationnary component on the data is to compute its derivative.
Another way is to filter the data with a gaussian kernel, and substract it to the original price data.
'''
# %% code
# derivative
price_dt = price[1:] - price[:-1]

# filter
filter_width = 12
def gaussian_kernel_1d(filter_width):
    #99% of the values
    sigma = (filter_width)/2.33
    norm = 1.0 / (np.sqrt(2*np.pi) * sigma)
    kernel = [norm * np.exp((-1)*(x**2)/(2 * sigma**2)) for x in range(-filter_width, filter_width + 1)]
    return np.float32(kernel / np.sum(kernel))
f = tf.reshape(gaussian_kernel_1d(filter_width), [-1, 1, 1])
tf_price = tf.reshape(tf.constant(price, dtype=tf.float32), [1, -1, 1])
tf_price = tf.reshape(tf.nn.conv1d(tf_price, filters=f, stride=1, padding='VALID'), [-1])
# padding is necessary to keep same dim
tf_price = tf.concat([ tf.constant(tf_price[0].numpy(), shape=filter_width), tf_price ], axis=0)
filt_price = tf.concat([ tf_price,tf.constant(tf_price[-1].numpy(), shape=filter_width) ], axis=0).numpy()
price_centered = price - filt_price
# %% [markdown]
'''
By comparing the two methods (derivative and filetring), we see that the resulting prices are now zero-centered.
They are shown with the orange colour in the below charts:
'''
# %% code
### plot
fig, axes = plt.subplots(2, figsize=(12, 8))
axes[0].plot(time_shifted, price, label="non-stationnary bitcoin price")
axes[0].plot(time_shifted[:-1], price_dt, label="stationnary bitcoin price")
axes[0].set_title('Derivative method')
axes[0].legend(loc="upper left")
axes[1].plot(time_shifted, price, label="non-stationnary bitcoin price")
axes[1].plot(time_shifted, price_centered, label="stationnary bitcoin price")
axes[1].plot(time_shifted, filt_price, label="filtered bitcoin price")
axes[1].set_title('Filtering and substraction method')
axes[1].legend(loc="upper left")
if figure_dir:
    plt.savefig(os.path.join(figure_dir, "price_stationnarity.png"))
plt.show()
plt.close()
# %% [markdown]
'''
In order to verify the quality of the process, one can check the auto-correlation for both the raw price data (blue line), and stationnary price data with the filtering method (green line).
This will inform us about how well the data is stationnary after the process.

We will compute the auto-correlations with different delays of up to 2 days every hours.
'''
# %% [code]
### auto-correlation function
def autocorr(input, delay):
    input = tf.constant(input, dtype=tf.float32)
    input_delayed = tf.roll(input, shift=delay, axis=0)

    x1 = tf.reshape(input, [1, -1, 1])
    x2 = tf.reshape(input_delayed, [-1, 1, 1])
    return tf.reshape(tf.nn.conv1d(x1, filters=x2, stride=1, padding='VALID'), [-1])
# %% [code]
# autocorrelation of the price for different delays
delays = np.arange(0, 48)
# raw price data
autocorr_price = []
for hour in delays:
    autocorr_price += [autocorr(price, hour)]
# stationnary data
autocorr_centered_price = []
for hour in delays:
    autocorr_centered_price += [autocorr(price_centered, hour)]
# %% [markdown]
'''
Looking at the plot, it is clear that the auto-correlation for the stationnary data degrades much faster than for the raw price data.
This means that we successfully removed the non-stationnary component for the price!
'''
# %% [code]
### plot
fig, axes = plt.subplots(2, figsize=(12, 8))
axes[0].stem(delays, autocorr_centered_price, linefmt='b-', markerfmt='bo', basefmt='', use_line_collection=True)
axes[0].set_title('stationnary bitcoin price auto-correlation')
axes[1].stem(delays, autocorr_price, linefmt='b-', markerfmt='bo', basefmt='', use_line_collection=True)
axes[1].set_title('raw bitcoin price auto-correlation')
axes[1].set(xlabel='delay (h)', ylabel='amplitude')
if figure_dir:
    plt.savefig(os.path.join(figure_dir, "check_stationnarity.png"))
plt.show()
plt.close()
# %% [markdown]
'''
Looking into the stationarity component also allows us to determine the window of prediction that is most suitable for the data.
For example by checking how fast, for a given timestamp, the distribution of the raw price differ with its neighbors.

By comparing the histogram (i.e. computing the correlation) for each timestamp with its neighbors, one can get an overview of what would be the acceptable range for a prediction.
With the idea that if the distributions are close to each other, it is obviously easier to predict (because they are closed to each other).
'''
# %% code
### histogram function
def data_distribution(inp):
    return np.histogram(inp, range=(0, 20000), bins=500, density=True)[0]
# %% code    
win_size = 2*24 #distribution of the data is calculated within 2 days (in hours)
slide = 5*24 #we slide up to -/+ 5 days
corr = []

# loop through al timestamps
timestamps_range = np.arange(slide + int(win_size/2), len(price) - slide - int(win_size/2), 72)
sliding_range = np.arange(-slide, slide + 1)
for i in timestamps_range:
    idx = i-int(win_size/2)
    # distribution of the price (over price from day -7.5 to day +7.5), the fixed distributioin
    fixed_price = price[idx:int(idx + win_size)]
    fixed_distrib = data_distribution(fixed_price)
    curr_corr = []
     # compare to each distribution at different timestamps (sliding from -30 to +30), the moving distribution 
    for offset in sliding_range:
        idx = offset + i - int(win_size/2)
        moving_price = price[idx:(idx + win_size)]
        moving_distrib = data_distribution(moving_price)
        curr_corr += [np.correlate(fixed_distrib, moving_distrib)]
    curr_corr = curr_corr / np.max(curr_corr)    
    corr += [curr_corr]    
    if i%7992 == 0:
        print("day {}/{}".format(i/24, len(price)/24))

output = np.array(corr)[:, :, 0]
# %% [markdown]
'''
In the following plot, the y-axis describes some samples taken at different timestamps of the bitcoin price.
From up to down, it follows the chronological order, but this is not important since each sample can be taken independently.
The x-axis desribes the different offsets to compute the histogramms (from -120 hours to +120 hours).
And the color is the resulting correlation between these distributions and the distribution at timestamp $h_0$ (the current timestamp of the sample).
'''
# %% [code]
### plot
plt.imshow(output, cmap="gray")
plt.axis("tight")
idx_sliding_range = np.arange(0, len(sliding_range), 30)
plt.xticks([i for i in idx_sliding_range], ["h{}".format(sliding_range[i]) for i in idx_sliding_range])
plt.xlabel("time offset (h)")
plt.ylabel("samples")
cbar = plt.colorbar()
cbar.set_label('correlation value')
if figure_dir:
    plt.imsave(os.path.join(figure_dir, "range_accuracy.png"), output, cmap="gray")
plt.show()
plt.close()
# %% [markdown]
'''
Looking at it, we can say that the acceptable range for prediction is around +/-15 hours.

>**Note**  
>The range for the color is verry granular, and sometimes constant.
>This is because of the number of bins in the histogramm (500) and price values ranging from 0 to 20k\$, meaning the precision is about ~40\$.
>So if the price moves inside the 40\$ range within a certain period, the histogramms will have a perfect match.
'''
# %% [markdown]
'''
## 2.3. Seasonnality
'''
# %% [markdown]
'''
Let's now switch the seasonality analysis by computing the FFT, and extract its magnitude and phase components.  
As explained before, the FFT will be used here to understand the redundant patterns in the data.
Because the FFT works better on LTI (linear and time invariant) systems, it cannot be applied with the raw bitcoin price (which is not stationnary!). Therefore we will apply it on the stationnary bitcoin price.
'''
# %% code
# fft
price_fouried = tf.signal.fft(price_centered)
T = 1/24 # sampling interval in days
N = price_fouried.shape[0]
frequencies = np.linspace(0, 1 / T, N)
# %% [markdown]
'''
In the below figure, there is no clear evidence of a pattern there, although we see evidence for important frequency ranging from 1 to 1.9 cycles per day, with a little peak at 1.52.
This means that the bitcoin price can "generally" be explained by a sinusoid with a period of ~15.8 hours.
'''
# %% code
### plot
fig, axes = plt.subplots(2, figsize=(12, 8))
axes[0].plot(frequencies[:N // 2], tf.abs(price_fouried)[:N // 2] * 1 / N)
axes[0].set_title('FFT magnitude')
axes[1].plot(frequencies[:N // 2], tf.math.angle(price_fouried)[:N // 2])
axes[1].set_title('FFT phase')
axes[1].set(xlabel='cycles per day', ylabel='amplitude')
if figure_dir:
    plt.savefig(os.path.join(figure_dir, "fft.png"))
plt.show()
plt.close()
# %% [markdown]
'''
Another way to analyse seasonnality on a non-stationnary data is to compute its spectogramm (derived from a time-frequency analysis).  
A spectrogram is a visual representation during time of a signal's spectrum of frequencies. It is commonly used (for example by [spleeter](https://github.com/deezer/spleeter)) to exctract voice from audio signals.
The spectrogram can be computed using a short-fourier transform, which basically runs a fourier transform on a short window, sliding through all the data.

Here, we will use a window size of 48 samples (hours), with a step of 1 and 62 frequency components.
'''
# %% [code]
# tensorflow provides a fast implementation of the fast fourier transform.
stft = tf.signal.stft(price, frame_length=48, frame_step=1, fft_length=125, pad_end=True)
spectrogram = tf.abs(stft).numpy()

# %% [code]
### plot
# inspired from https://www.tensorflow.org/tutorials/audio/simple_audio
# convert to log scale and transpose so that the time is represented in the x-axis (columns).
fig, axes = plt.subplots(2, figsize=(12, 8))
max_time = np.max(time_shifted)
axes[0].plot(time_shifted, price)
axes[0].set_xlim([0, max_time])
axes[0].set_title('non-stationnary bitcoin price')
log_spec = np.log(spectrogram.T)
axes[1].pcolormesh(time_shifted, np.arange(log_spec.shape[0]), log_spec)
axes[1].set_xlim([0, max_time])
axes[1].set_title('Spectrogram (short-fft)')
axes[1].set(xlabel='time (h)', ylabel='frequencies')
if figure_dir:
    plt.savefig(os.path.join(figure_dir, "spectrogram.png"))
plt.show()
plt.close()

# %% [markdown]
'''
Looking at the figure, whenever there are big changes in the data (for example Dec. 2017), there is a much higher magnitude response.
Generally speaking, it seems that the FFT looks like a white noise whenever the time.
'''
# %% [markdown]
'''
## Conclusion
'''
# %% [markdown]
'''
In the light of the properties that we saw above, one thing can be said with certainty; predicting bitcoin price is no easy task because of its time dependency.  
Hopefully we found a way to simplify the process, by removing the non-stationnary component of the data (so it does not depend on time anymore).
This allowed us to analyse redundant patterns in the data and we found that such a pattern exists.
The reccurent patterns are interresting, because they can be latter used as a new feature into a predictive model (think of adding the time of day into a weather prediction model for example).

These findings oppened to us new ways to get an accurate predictive model for the bitcoin price, but this is another story...
'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
I really suggest you to read the book from Hyndman <cite> hyndman2018forecasting </cite>, it covers all best practices for time-series forecasting as well as coding examples.
The online version is available [here](https://otexts.com/fpp2/index.html).
'''
# %% [markdown]
'''
## Acknowledgement
'''
# %% [markdown]
'''
Thanks to [Vahid Zarifpayam](https://twitter.com/Vahidzarif1) for the review of this post.  
Credits goes to [Bitprobe](https://bitprobe.io/).
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Data-Science; Cryptocurrency; Statistics;
'''